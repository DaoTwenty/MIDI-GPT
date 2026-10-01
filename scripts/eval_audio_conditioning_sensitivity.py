"""Audio-conditioning sensitivity eval: does the model actually use its audio
conditioning, or has it learned to ignore it?

The single biggest risk in any conditional generation model is that it learns
to ignore the conditioning signal entirely (a known failure mode -- see this
project's own conditioning-slakh sibling study, whose control_adherence sweep
exists for exactly this reason on the symbolic-attribute side). This script
is the audio-conditioning analog: computes NLL on real MIDI targets under 4
conditions per validation piece, reusing the exact model forward path
training uses (AudioConditionedLightningModule._compute_loss's
soft_token_replacements injection mechanism), just swapping which audio
feeds those positions:

  (a) real      -- the real, correctly-aligned audio context (baseline)
  (b) silence   -- every audio position's waveform zeroed out before feature
                   extraction (does the model use audio at all?)
  (c) mismatched -- waveforms swapped in from a different, unrelated
                   validation piece, same position count via cycling (does
                   the model care about audio CONTENT, or just presence?)
  (d) shuffled  -- the SAME piece's own real waveforms cyclically permuted
                   among their own positions (does the model care about
                   correct track/bar ALIGNMENT, or just "some real audio
                   from this piece exists somewhere"?)

A genuinely audio-conditioned model should show real << shuffled < mismatched
~ silence in NLL. Convergence of any non-real condition toward real's NLL
indicates the model is ignoring that specific distinction. Bootstrap 95% CIs
on both each condition's mean and its delta from real, mirroring
eval_ablation_nll.py's methodology (arch-modernization) and
reeval_control_adherence.py's (conditioning-slakh).

Usage:
    python3 scripts/eval_audio_conditioning_sensitivity.py \\
        --lightning-ckpt runs/audio_medium_full/checkpoints/last.ckpt \\
        --midi-ckpt <base pretrained yellow_medium checkpoint>.safetensors \\
        --encoder-config <path to encoder config json> \\
        --eval-data data/.../validation/*.parquet \\
        --output-dir runs/audio_conditioning_eval/full \\
        [--n-pieces 200] [--audio-extractor muscriptor] [--pooling perceiver]
        [--fusion concat] [--seed 0] [--n-bootstrap 2000] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

N_BOOTSTRAP = 2000
CONDITIONS = ("real", "silence", "mismatched", "shuffled")


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, rng=None):
    rng = rng or random.Random(0)
    n = len(values)
    if n == 0:
        return None
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return lo, hi


def build_model(args, tokenizer, encoder_config):
    from midigpt.adapters.audio_adapter import AudioToMidiAdapter
    from midigpt.audio import load_audio_extractor
    from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
    from midigpt.training.audio_lightning_module import AudioConditionedLightningModule

    if args.midi_ckpt:
        # Mirrors train_audio_conditioned.py's _load_midi_ckpt exactly: the
        # base checkpoint's own vocab can differ from the target encoder
        # config's (amber_encoder.json adds audio-placeholder tokens on top
        # of whatever yellow_medium's base encoder had) -- resize before the
        # trained lightning checkpoint's state_dict gets loaded on top,
        # which is already sized for the target (resized) vocab.
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
    else:
        gpt2_model = GPT2LMHeadModel(GPT2Config(
            vocab_size=tokenizer.vocab_size(), n_positions=args.max_seq_len,
            n_embd=256, n_layer=4, n_head=4,
        ))
    model_dim = gpt2_model.transformer.wte.weight.shape[1]

    extractor = load_audio_extractor(
        extractor_type=args.audio_extractor, weights_path=args.audio_weights,
        feature_dim=args.feature_dim,
    )
    adapter = AudioToMidiAdapter(
        feature_dim=args.feature_dim, model_dim=model_dim,
        num_tokens_per_bar=args.tokens_per_bar,
        pooling_strategy=args.pooling, fusion_strategy=args.fusion,
    )

    lit_module = AudioConditionedLightningModule(
        gpt2_model=gpt2_model, audio_adapter=adapter, audio_extractor=extractor,
        training_mode="full",  # irrelevant for eval -- only affects requires_grad
    )

    ckpt = torch.load(args.lightning_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = lit_module.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Loaded lightning checkpoint. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing keys (first 10): {missing[:10]}", flush=True)

    lit_module.eval()
    lit_module.to(args.device)
    return lit_module


def build_soft_replacements(
    lit_module, sample_idx, waveforms, track_indices, bar_intervals, refs, positions, device, sample_rate
):
    """Mirrors AudioConditionedLightningModule._extract_soft_replacements
    exactly: `waveforms`/`track_indices` are PER-TRACK (one full-track
    waveform per audio track -- NOT one per position/reference; a single
    track's waveform can be sliced at multiple different bar intervals for
    multiple different refs). `refs` is a list of (t_idx, b_idx) pairs,
    `positions` the matching token positions -- len(refs) == len(positions),
    which can exceed len(waveforms).
    """
    if not waveforms or not refs:
        return []
    min_len = int(sample_rate * 0.1)
    max_len = max(max(w.shape[-1] for w in waveforms), min_len)
    stacked = torch.zeros(len(waveforms), max_len, device=device)
    for i, w in enumerate(waveforms):
        stacked[i, : w.shape[-1]] = w.to(device)

    with torch.no_grad():
        frames = lit_module.audio_extractor.extract_frames(stacked, sample_rate=sample_rate)
    total_duration = max_len / sample_rate
    track_to_local = {t_idx: i for i, t_idx in enumerate(track_indices)}

    out = []
    for (t_idx, b_idx), pos in zip(refs, positions):
        local = track_to_local[t_idx]
        t_frames = frames[local]
        b_frames = lit_module.audio_extractor.slice_bar_frames(t_frames, total_duration, [bar_intervals[b_idx]])[0]
        soft_emb = lit_module.audio_adapter(b_frames.unsqueeze(0), track_frames=t_frames.unsqueeze(0))
        out.append((sample_idx, int(pos), soft_emb))
    return out


def nll_under_condition(
    lit_module, input_ids, labels, waveforms, track_indices, bar_intervals, refs, positions, device, sample_rate
):
    soft_replacements = build_soft_replacements(
        lit_module, 0, waveforms, track_indices, bar_intervals, refs, positions, device, sample_rate
    )
    with torch.no_grad():
        logits = lit_module.forward(input_ids.unsqueeze(0).to(device), soft_token_replacements=soft_replacements)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[1:].unsqueeze(0).contiguous().to(device)
    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
        ignore_index=-100, reduction="sum",
    )
    n_valid = (shift_labels != -100).sum().item()
    return per_tok.item(), n_valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lightning-ckpt", required=True)
    ap.add_argument("--midi-ckpt", default=None)
    ap.add_argument("--encoder-config", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-pieces", type=int, default=200)
    ap.add_argument("--audio-extractor", default="muscriptor", choices=["yourmt3", "yourmt3+", "muscriptor"])
    ap.add_argument("--audio-weights", default=None)
    ap.add_argument("--feature-dim", type=int, default=768)
    ap.add_argument("--pooling", default="perceiver", choices=["mean_conv", "perceiver", "beat_grid"])
    ap.add_argument("--fusion", default="concat", choices=["concat", "film", "residual", "none"])
    ap.add_argument("--tokens-per-bar", type=int, default=1)
    ap.add_argument("--max-bars", type=int, default=64)
    ap.add_argument("--bar-step", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--soundfont-dir", default="soundfonts")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import glob

    import midigpt._core as _core
    from midigpt.audio import SoundFontSynthesizer
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.training.audio_dataset import AudioTransitionDataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_config = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    tokenizer = Tokenizer(encoder_config)

    print("Building model...", flush=True)
    lit_module = build_model(args, tokenizer, encoder_config)

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
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    items = []
    for idx in indices:
        if len(items) >= args.n_pieces:
            break
        item = dataset[idx]
        if item is None or not item.get("audio_waveforms"):
            continue
        items.append(item)
    print(f"Collected {len(items)} usable pieces (audio_waveforms non-empty)", flush=True)

    per_condition_nats: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    per_condition_tokens: dict[str, list[float]] = {c: [] for c in CONDITIONS}

    for i, item in enumerate(items):
        input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
        labels = torch.as_tensor(item["labels"], dtype=torch.long)
        real_waveforms = item["audio_waveforms"]  # per-TRACK, not per-position
        track_indices = item["audio_track_indices"]
        bar_intervals = item["bar_intervals"]
        refs = item["audio_bar_refs"]  # (t_idx, b_idx) pairs, one per position
        positions = item["audio_bar_positions"]
        if not real_waveforms or not refs:
            continue

        def _run(waveforms, tag):
            nats, ntok = nll_under_condition(
                lit_module, input_ids, labels, waveforms, track_indices, bar_intervals, refs, positions,
                args.device, args.sample_rate,
            )
            per_condition_nats[tag].append(nats)
            per_condition_tokens[tag].append(ntok)

        # (a) real
        _run(real_waveforms, "real")

        # (b) silence -- same per-track structure, zeroed content
        _run([torch.zeros_like(w) for w in real_waveforms], "silence")

        # (c) mismatched -- this piece's track/bar/position structure unchanged,
        # but the per-track waveform CONTENT borrowed from a different,
        # randomly chosen piece (cycled to match this piece's track count).
        other = items[rng.randrange(len(items))]
        other_waveforms = other["audio_waveforms"] or real_waveforms
        _run([other_waveforms[j % len(other_waveforms)] for j in range(len(real_waveforms))], "mismatched")

        # (d) shuffled -- this piece's own real per-track waveforms, cyclically
        # permuted among their own track slots (track_indices/bar_intervals/
        # refs/positions all unchanged, so refs still resolve to the same
        # local slots -- just holding a different track's real audio now).
        if len(real_waveforms) > 1:
            _run(real_waveforms[1:] + real_waveforms[:1], "shuffled")
        else:
            _run(real_waveforms, "shuffled")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)} pieces done", flush=True)

    rng2 = random.Random(args.seed)
    results = {}
    for cond in CONDITIONS:
        nats = per_condition_nats[cond]
        toks = per_condition_tokens[cond]
        if not nats:
            results[cond] = None
            continue
        total_nats = sum(nats)
        total_toks = sum(toks)
        per_piece_mean = [n / t for n, t in zip(nats, toks) if t > 0]
        ci = bootstrap_mean_ci(per_piece_mean, n_boot=args.n_bootstrap, rng=rng2)
        results[cond] = {
            "n_pieces": len(nats),
            "nats_per_token": total_nats / total_toks if total_toks else None,
            "nats_per_token_ci95": list(ci) if ci else None,
        }

    if results["real"] and results["real"]["nats_per_token"] is not None:
        real_val = results["real"]["nats_per_token"]
        for cond in CONDITIONS:
            if cond == "real" or not results[cond]:
                continue
            results[cond]["delta_from_real"] = results[cond]["nats_per_token"] - real_val

    summary = {
        "lightning_ckpt": args.lightning_ckpt,
        "n_pieces_requested": args.n_pieces,
        "n_pieces_used": len(items),
        "seed": args.seed,
        "conditions": results,
    }
    (out_dir / "audio_conditioning_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
