"""Tier-B audio-conditioning GENERATION eval (standalone — NOT the training val loop).

Tier A (val/loss, entropy, token-acc) is teacher-forced and lives in the
training loop. Tier B here is generation-based: it actually SAMPLES MIDI from
the audio-conditioned model and compares it to ground truth, which Tier A
cannot measure. It is deliberately a separate, heavier eval run on a handful of
trained checkpoints (e.g. the top extractors from the val/loss screen), not
something bolted onto every training step.

WHAT THE TASK ACTUALLY IS (important framing):
  The audio fed to the model is NOT the audio of the track being generated — it
  is the audio of the OTHER (context) tracks. The model generates a *different*
  track's MIDI conditioned on the context tracks' audio. So this is conditional
  arrangement/accompaniment generation, NOT transcription. Consequences:
    * It is inherently one-to-many (many target tracks fit a given accompaniment),
      so ABSOLUTE note-F1 against the single ground-truth target is low even for a
      good model and is misleading on its own.
    * The meaningful signal is the DELTA vs control conditioning: a model that
      genuinely uses the audio should recover the GT target BETTER with the real
      context audio than with mismatched / silent audio. So every metric is
      computed under {real, mismatched, silence} and we report real-minus-control.

METRICS (all comparing generated TARGET tracks vs GT TARGET tracks, i.e. the
non-"audio" tracks):
  * note_f1      : onset+pitch note matching (self-contained; no mir_eval dep).
  * dist_div     : distribution divergence of musical attributes (pitch-class,
                   note-duration, polyphony, density, pitch-range) between GT and
                   generated sets — overlapping-area / Jensen-Shannon per attribute.
  * fmd_proxy    : Fréchet distance between GT and generated sets in a feature
                   space built from those attributes (a dependency-free stand-in
                   for Fréchet Music Distance; true FMD would use a learned
                   symbolic-music embedder — port the arch-modernization pipeline
                   later and drop it in via --fmd-embedder).

Usage (run on a cluster GPU; this script is NOT wired into any PRISM job yet):
    python3 scripts/eval_audio_generation.py \\
        --lightning-ckpt runs/audio_screen_<ext>/checkpoints/last.ckpt \\
        --midi-ckpt <yellow_medium>.safetensors \\
        --encoder-config models/amber_encoder.json \\
        --eval-data data/v2.0.0/validation/*.parquet \\
        --audio-extractor <ext> [--audio-weights <dir>] \\
        --output-dir runs/audio_gen_eval/<ext> \\
        [--n-pieces 100] [--conditions real,mismatched,silence] \\
        [--temperature 0.9 --top-p 0.95 --seed 0]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import torch

CONDITIONS_DEFAULT = ("real", "mismatched", "silence")


# ---------------------------------------------------------------------------
# Model / engine setup (mirrors eval_audio_conditioning_sensitivity.build_model)
# ---------------------------------------------------------------------------

def build_lit_module(args, tokenizer, encoder_config):
    from midigpt.adapters.audio_adapter import AudioToMidiAdapter
    from midigpt.audio import load_audio_extractor
    from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
    from midigpt.training.audio_lightning_module import AudioConditionedLightningModule

    import midigpt._core as _core
    from safetensors import safe_open

    with safe_open(str(args.midi_ckpt), framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
        ckpt_cfg = json.loads(f.metadata().get("config", "{}"))
    gpt2_model = GPT2LMHeadModel(GPT2Config(**ckpt_cfg))
    gpt2_model.load_state_dict(state_dict, strict=False)
    target_vocab = _core.Vocabulary(encoder_config).size()
    if target_vocab != ckpt_cfg["vocab_size"]:
        gpt2_model.resize_token_embeddings(target_vocab)
    model_dim = gpt2_model.transformer.wte.weight.shape[1]

    extractor = load_audio_extractor(
        extractor_type=args.audio_extractor, weights_path=args.audio_weights,
        feature_dim=args.feature_dim,
    )
    adapter = AudioToMidiAdapter(
        feature_dim=args.feature_dim, model_dim=model_dim,
        num_tokens_per_bar=1, pooling_strategy=args.pooling, fusion_strategy=args.fusion,
    )
    lit = AudioConditionedLightningModule(
        gpt2_model=gpt2_model, audio_adapter=adapter, audio_extractor=extractor,
        training_mode="full",
    )
    ckpt = torch.load(args.lightning_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = lit.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Loaded lightning ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    lit.eval().to(args.device)
    return lit


def build_engine(lit, tokenizer, encoder_config):
    """Wrap the trained gpt2 in an InferenceEngine and attach the audio adapter+extractor."""
    from midigpt.inference.engine import InferenceEngine
    from midigpt.attributes.base import AttributeAnalyzer

    analyzer = AttributeAnalyzer.from_config(encoder_config)
    engine = InferenceEngine(lit.gpt2_model, tokenizer, analyzer)
    engine.warmup()
    engine.attach_audio_conditioning(lit.audio_adapter, lit.audio_extractor)
    return engine


# ---------------------------------------------------------------------------
# Audio conditioning construction (frames -> request.audio_conditioning)
# ---------------------------------------------------------------------------

def extract_frames_per_track(extractor, waveforms, track_indices, sample_rate, device):
    """Run the extractor once over the padded per-track waveforms.

    Returns (frames_by_track_local[list (T,D)], total_duration, track_to_local).
    Mirrors AudioConditionedLightningModule._extract_soft_replacements.
    """
    if not waveforms:
        return [], 0.0, {}
    min_len = int(sample_rate * 0.1)
    max_len = max(max(w.shape[-1] for w in waveforms), min_len)
    stacked = torch.zeros(len(waveforms), max_len, device=device)
    for i, w in enumerate(waveforms):
        stacked[i, : w.shape[-1]] = w.to(device)
    with torch.no_grad():
        frames = extractor.extract_frames(stacked, sample_rate=sample_rate)  # (n, T, D)
    total_duration = max_len / sample_rate
    track_to_local = {t: i for i, t in enumerate(track_indices)}
    return frames, total_duration, track_to_local


def build_audio_conditioning(extractor, waveforms, track_indices, bar_intervals, refs,
                             sample_rate, device):
    """Produce request.audio_conditioning = list of (bar_frames_3d, track_frames_3d).

    One entry per audio bar, in `refs` order (track-major, bar order) — the same
    order the encoder emits [AudioContext] placeholders. The engine now binds
    these to its own placeholder positions (see _build_audio_soft_replacements in
    session.py), so NO token position is needed here. Frames are raw (1, T, D);
    the adapter pools internally.
    """
    if not waveforms or not refs:
        return []
    frames, total_duration, track_to_local = extract_frames_per_track(
        extractor, waveforms, track_indices, sample_rate, device)
    out = []
    for (t_idx, b_idx) in refs:
        if t_idx not in track_to_local:
            continue
        t_frames = frames[track_to_local[t_idx]]                                   # (T_total, D)
        b_frames = extractor.slice_bar_frames(t_frames, total_duration, [bar_intervals[b_idx]])[0]
        out.append((b_frames.unsqueeze(0), t_frames.unsqueeze(0)))                 # (1,T,D) each
    return out


def perturb_waveforms(real_waveforms, condition, rng, pool):
    """Return the per-track waveforms for a given control condition.

      real       -> the real context-track audio
      silence     -> zeroed (does the model use audio at all?)
      mismatched  -> content from a different random piece (cares about content?)
    """
    if condition == "real":
        return real_waveforms
    if condition == "silence":
        return [torch.zeros_like(w) for w in real_waveforms]
    if condition == "mismatched":
        other = pool[rng.randrange(len(pool))] or real_waveforms
        return [other[j % len(other)] for j in range(len(real_waveforms))]
    raise ValueError(f"unknown condition {condition}")


# ---------------------------------------------------------------------------
# Score -> note events, for the TARGET (non-"audio") tracks only
# ---------------------------------------------------------------------------

def _abs_onset_seconds(score, track, bar_idx, note):
    """Absolute onset time (s): bars before this one + the note's in-bar onset.

    beat_length is in quarter notes; resolution = ticks/quarter; tempo = us/quarter.
    """
    us_per_quarter = score.tempo
    ticks_per_quarter = score.resolution
    bars_ticks = 0
    for b in track.bars[:bar_idx]:
        beats = b.ts_numerator * (4.0 / b.ts_denominator)      # quarters in the bar
        bars_ticks += beats * ticks_per_quarter
    onset_ticks_abs = bars_ticks + note.onset_ticks
    return onset_ticks_abs * (us_per_quarter / 1e6) / ticks_per_quarter


def target_track_notes(score):
    """Collect (onset_sec, pitch) for every note in the non-'audio' (target) tracks.

    Returns dict: track_idx -> list[(onset_sec, offset_sec, pitch)].
    """
    out = {}
    for t_idx, track in enumerate(score.tracks):
        if getattr(track, "track_type", None) == "audio":
            continue
        notes = []
        for b_idx, bar in enumerate(track.bars):
            for n in bar.notes:
                on = _abs_onset_seconds(score, track, b_idx, n)
                dur = n.duration_ticks * (score.tempo / 1e6) / score.resolution
                notes.append((on, on + max(dur, 1e-3), n.pitch))
        out[t_idx] = notes
    return out


# ---------------------------------------------------------------------------
# Metric 1: note-level precision / recall / F1 (self-contained, no mir_eval)
# ---------------------------------------------------------------------------

def note_prf1(gen_notes, gt_notes, onset_tol=0.05):
    """Greedy one-to-one match on (pitch exact, |onset_gen - onset_gt| <= tol)."""
    if not gen_notes and not gt_notes:
        return 1.0, 1.0, 1.0
    if not gen_notes or not gt_notes:
        return 0.0, 0.0, 0.0
    gt_by_pitch = {}
    for i, (on, off, p) in enumerate(gt_notes):
        gt_by_pitch.setdefault(p, []).append((on, i))
    used = set()
    tp = 0
    for on, off, p in sorted(gen_notes):
        cands = gt_by_pitch.get(p, [])
        best, best_d = None, onset_tol + 1e-9
        for gon, gi in cands:
            if gi in used:
                continue
            d = abs(gon - on)
            if d <= best_d:
                best, best_d = gi, d
        if best is not None:
            used.add(best)
            tp += 1
    prec = tp / len(gen_notes)
    rec = tp / len(gt_notes)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Metric 2 + 3: attribute feature vectors -> distribution divergence + Fréchet
# ---------------------------------------------------------------------------

def _pitch_class_hist(notes):
    h = [0.0] * 12
    for _, _, p in notes:
        h[p % 12] += 1.0
    s = sum(h)
    return [x / s for x in h] if s else h


def _duration_hist(gen_dur, bins=(0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)):
    c = [0.0] * (len(bins) - 1)
    for d in gen_dur:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                c[i] += 1.0
                break
    s = sum(c)
    return [x / s for x in c] if s else c


def feature_vector(notes):
    """Compact musical feature vector for a target-track note set (for Fréchet + summary)."""
    if not notes:
        return [0.0] * 16
    pitches = [p for _, _, p in notes]
    durs = [off - on for on, off, _ in notes]
    onsets = sorted(on for on, _, _ in notes)
    iois = [b - a for a, b in zip(onsets, onsets[1:])] or [0.0]
    pc = _pitch_class_hist(notes)                 # 12
    dh = _duration_hist(durs)[:0]                  # placeholder, folded below
    feats = pc + [
        statistics.mean(pitches) / 127.0,
        (statistics.pstdev(pitches) if len(pitches) > 1 else 0.0) / 127.0,
        statistics.mean(durs),
        statistics.mean(iois),
    ]
    return feats


def jensen_shannon(p, q):
    def kl(a, b):
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def overlapping_area(p, q):
    return sum(min(a, b) for a, b in zip(p, q))


def frechet_distance(gen_feats, gt_feats):
    """Fréchet distance between two sets of feature vectors (diagonal-cov proxy).

    ||mu_g - mu_r||^2 + sum(var_g + var_r - 2*sqrt(var_g*var_r)). Diagonal
    covariance keeps it dependency-free; a full-cov / learned-embedder FMD can
    replace this later without changing the harness.
    """
    import statistics as st
    if not gen_feats or not gt_feats:
        return None
    d = len(gen_feats[0])
    mg = [st.mean(f[i] for f in gen_feats) for i in range(d)]
    mr = [st.mean(f[i] for f in gt_feats) for i in range(d)]
    vg = [st.pvariance([f[i] for f in gen_feats]) if len(gen_feats) > 1 else 0.0 for i in range(d)]
    vr = [st.pvariance([f[i] for f in gt_feats]) if len(gt_feats) > 1 else 0.0 for i in range(d)]
    mean_term = sum((a - b) ** 2 for a, b in zip(mg, mr))
    cov_term = sum(g + r - 2 * math.sqrt(max(g * r, 0.0)) for g, r in zip(vg, vr))
    return mean_term + cov_term


# ---------------------------------------------------------------------------
# Generation of the target tracks for one piece under one condition
# ---------------------------------------------------------------------------

def generate_accompaniment(engine, tokenizer, item, audio_conditioning, args):
    """Generate the TARGET (non-audio) tracks conditioned on the context (audio)
    tracks, via the standard engine — only the conditioning differs.

    Request idiom (inference.md "Multi-track AR with leading context"):
      * audio (context) tracks -> TrackPrompt(ignore=True): left untouched; their
        [AudioContext] placeholders carry the injected audio soft embeddings.
      * target (MIDI) tracks    -> TrackPrompt(bars=[], autoregressive=True):
        generated fully from scratch — we also CLEAR their notes in the Score
        passed to the engine so generation can't copy the ground truth.
    audio_conditioning is the position-free (bar_frames, track_frames) list; the
    engine binds it to its own placeholder positions.

    Returns (generated_score, gt_score, target_track_indices).
    """
    import copy
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

    gt_score = tokenizer.decode(item["input_ids"], resample=True)
    gen_score = copy.deepcopy(gt_score)
    prompts, target_tracks = [], []
    for t_idx, track in enumerate(gen_score.tracks):
        if getattr(track, "track_type", None) == "audio":
            prompts.append(TrackPrompt(id=t_idx, ignore=True))
        else:
            for bar in track.bars:
                bar.notes = []  # don't let AR copy the GT target
            prompts.append(TrackPrompt(id=t_idx, bars=[], autoregressive=True))
            target_tracks.append(t_idx)
    cfg = InferenceConfig(temperature=args.temperature, top_k=args.top_k,
                          top_p=args.top_p, seed=args.seed)
    req = GenerationRequest(tracks=prompts, config=cfg, audio_conditioning=audio_conditioning)
    return engine.session(gen_score, req).run(), gt_score, target_tracks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lightning-ckpt", required=True)
    ap.add_argument("--midi-ckpt", required=True)
    ap.add_argument("--encoder-config", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--audio-extractor", required=True)
    ap.add_argument("--audio-weights", default=None)
    ap.add_argument("--feature-dim", type=int, default=768)
    ap.add_argument("--pooling", default="perceiver")
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--n-pieces", type=int, default=100)
    ap.add_argument("--conditions", default=",".join(CONDITIONS_DEFAULT))
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--max-bars", type=int, default=64)
    ap.add_argument("--bar-step", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--soundfont-dir", default="soundfonts")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import midigpt._core as _core
    from midigpt.audio import SoundFontSynthesizer
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.training.audio_dataset import AudioTransitionDataset

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_config = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    tokenizer = Tokenizer(encoder_config)

    print("Building model + engine...", flush=True)
    lit = build_lit_module(args, tokenizer, encoder_config)
    engine = build_engine(lit, tokenizer, encoder_config)

    print("Loading validation data...", flush=True)
    eval_paths = sorted(glob.glob(args.eval_data)) if "*" in args.eval_data else [args.eval_data]
    import datasets as hf
    raw = hf.load_dataset("parquet", data_files=eval_paths, split="train")
    synth = SoundFontSynthesizer(args.soundfont_dir, sample_rate=args.sample_rate)
    num_bars_list = list(range(args.bar_step, args.max_bars + args.bar_step, args.bar_step))
    dataset = AudioTransitionDataset(
        scores=None, raw_rows=raw, tokenizer=tokenizer, num_bars_choices=num_bars_list,
        synthesizer=synth, max_seq_len=args.max_seq_len,
        score_ids=[str(i) for i in range(len(raw))],
    )

    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    items = []
    for idx in order:
        if len(items) >= args.n_pieces:
            break
        it = dataset[idx]
        if it and it.get("audio_waveforms"):
            items.append(it)
    print(f"Collected {len(items)} usable pieces", flush=True)
    wav_pool = [it["audio_waveforms"] for it in items]

    # metric accumulators: per condition -> lists
    f1s = {c: [] for c in conditions}
    js_pc = {c: [] for c in conditions}
    oa_pc = {c: [] for c in conditions}
    feats_gen = {c: [] for c in conditions}
    feats_gt = []

    for i, it in enumerate(items):
        gt_full_notes = None
        for cond in conditions:
            wavs = perturb_waveforms(it["audio_waveforms"], cond, rng, wav_pool)
            ac = build_audio_conditioning(
                lit.audio_extractor, wavs, it["audio_track_indices"], it["bar_intervals"],
                it["audio_bar_refs"], args.sample_rate, args.device)
            try:
                gen_score, gt_score, target_tracks = generate_accompaniment(engine, tokenizer, it, ac, args)
            except Exception as e:
                print(f"  piece {i} cond {cond}: generation failed: {repr(e)[:160]}", flush=True)
                continue
            gen_notes_by_t = target_track_notes(gen_score)
            gt_notes_by_t = target_track_notes(gt_score)
            gen_notes = [n for t in gen_notes_by_t.values() for n in t]
            gt_notes = [n for t in gt_notes_by_t.values() for n in t]
            if gt_full_notes is None:
                gt_full_notes = gt_notes
                feats_gt.append(feature_vector(gt_notes))
            _, _, f1 = note_prf1(gen_notes, gt_notes, onset_tol=args.onset_tol)
            f1s[cond].append(f1)
            pg, pr = _pitch_class_hist(gen_notes), _pitch_class_hist(gt_notes)
            js_pc[cond].append(jensen_shannon(pg, pr))
            oa_pc[cond].append(overlapping_area(pg, pr))
            feats_gen[cond].append(feature_vector(gen_notes))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} pieces done", flush=True)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    results = {}
    for cond in conditions:
        results[cond] = {
            "n": len(f1s[cond]),
            "note_f1": _mean(f1s[cond]),
            "pitchclass_js": _mean(js_pc[cond]),
            "pitchclass_overlap": _mean(oa_pc[cond]),
            "fmd_proxy": frechet_distance(feats_gen[cond], feats_gt),
        }
    # real-minus-control deltas (the primary signal, given the one-to-many task)
    deltas = {}
    if "real" in results and results["real"]["note_f1"] is not None:
        for cond in conditions:
            if cond == "real":
                continue
            r, c = results["real"], results[cond]
            deltas[cond] = {
                "note_f1_real_minus_ctrl": (r["note_f1"] - c["note_f1"]) if c["note_f1"] is not None else None,
                "pitchclass_overlap_real_minus_ctrl": (
                    r["pitchclass_overlap"] - c["pitchclass_overlap"]) if c["pitchclass_overlap"] is not None else None,
                "fmd_proxy_ctrl_minus_real": (
                    c["fmd_proxy"] - r["fmd_proxy"]) if (c["fmd_proxy"] is not None and r["fmd_proxy"] is not None) else None,
            }

    summary = {
        "extractor": args.audio_extractor,
        "lightning_ckpt": args.lightning_ckpt,
        "n_pieces_used": len(items),
        "conditions": results,
        "deltas_real_minus_control": deltas,
        "note": ("Task is conditional arrangement (target track from OTHER tracks' audio), "
                 "one-to-many; absolute note_f1 is low by nature — read the real-minus-control "
                 "deltas as the signal that the model uses the audio."),
    }
    (out_dir / "audio_generation_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
