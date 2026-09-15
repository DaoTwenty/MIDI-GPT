"""Full smoke test: GigaMIDI + Slakh2100 + SoundFont + Yellow_Small."""

import os, json, random, time, copy, sys
os.environ.setdefault("DYLD_LIBRARY_PATH", "/usr/local/lib")

import torch
import midigpt._core as _core
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt._types import Score, Bar
from safetensors import safe_open

# ── Token name map ──────────────────────────────────────────────────

ALL_TOKEN_TYPES = [
    _core.TokenType.PieceStart, _core.TokenType.PieceEnd,
    _core.TokenType.NumBars, _core.TokenType.Track, _core.TokenType.TrackEnd,
    _core.TokenType.Bar, _core.TokenType.BarEnd,
    _core.TokenType.TimeSig, _core.TokenType.Instrument,
    _core.TokenType.NotePitch, _core.TokenType.NoteOnset, _core.TokenType.NoteDuration,
    _core.TokenType.PitchClass, _core.TokenType.Velocity,
    _core.TokenType.AudioContext,
    _core.TokenType.FillIn, _core.TokenType.FillInPlaceholder,
    _core.TokenType.FillInStart, _core.TokenType.FillInEnd,
    _core.TokenType.MaskBar, _core.TokenType.Genre,
    _core.TokenType.UseVelocity, _core.TokenType.UseMicrotiming,
    _core.TokenType.Delta,
]


def build_token_names(vocab):
    names = {}
    for tt in ALL_TOKEN_TYPES:
        if vocab.has(tt):
            for v in range(vocab.domain_size(tt)):
                tid = vocab.encode_val(tt, v)
                names[tid] = f"[{tt.name}={v}]" if v > 0 else f"[{tt.name}]"
    return names


def compact_log(token_ids, names, max_notes=3):
    """Collapse runs of note tokens into [NOTES xN]."""
    parts = []
    i = 0
    while i < len(token_ids):
        tid = token_ids[i]
        name = names.get(tid, f"?{tid}")
        is_structural = name.startswith("[") and name.endswith("]")
        if not is_structural:
            run_start = i
            while i < len(token_ids) and not names.get(token_ids[i], "").startswith("["):
                i += 1
            run_len = i - run_start
            if run_len <= max_notes:
                for j in range(run_start, i):
                    parts.append(names.get(token_ids[j], f"?{token_ids[j]}"))
            else:
                parts.append(f"[NOTES x{run_len}]")
        else:
            parts.append(name)
            i += 1
    return " ".join(parts)


# ── Dummy audio extractor for smoke test ───────────────────────────

def main():
    # ── 1. Load tokenizer ──────────────────────────────────────────
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Metacreation--MIDI-GPT/"
        "snapshots/3aa5748ba68e32a43c70ddc66b5e595069484532/"
        "yellow_small-final.safetensors"
    )
    with safe_open(cache_path, framework="pt") as f:
        ckpt_cfg = json.loads(f.metadata().get("config", "{}"))

    enc_json = {
        "num_bars_map": [4, 8],
        "supports_infill": ckpt_cfg.get("supports_infill", True),
        "time_signatures": ["4/4", "3/4", "2/4", "6/8", "3/2", "2/2", "1/4", "5/4", "7/8"],
    }
    encoder_config = _core.EncoderConfig.from_json(json.dumps(enc_json))
    tokenizer = Tokenizer(encoder_config)
    vocab = tokenizer._vocab
    NAMES = build_token_names(vocab)

    print("=" * 70)
    print("FULL SMOKE TEST: GigaMIDI + Slakh2100 + SoundFont + Yellow_Small")
    print("=" * 70)

    # ── 2. Load GigaMIDI ───────────────────────────────────────────
    print("\n[1/6] Loading GigaMIDI parquet...")
    import datasets as hf
    parquet_path = "tmp/gigamidi/v2.0.0/train/00000.parquet"
    data = hf.load_dataset("parquet", data_files=parquet_path, split="train")
    giga_scores = []
    for i in range(min(50, len(data))):
        try:
            score = Score.from_bytes(data[i]["music"])
            giga_scores.append(score)
        except Exception:
            continue
    print(f"  Loaded {len(giga_scores)} scores from GigaMIDI")
    s = giga_scores[0]
    print(f"  Score 0: {len(s.tracks)} tracks, tempo={s.tempo}, res={s.resolution}")
    for ti, t in enumerate(s.tracks):
        n_notes = sum(len(b.notes) for b in t.bars)
        print(f"    Track {ti}: type={t.track_type}, {len(t.bars)} bars, {n_notes} notes")

    # ── 3. Load Slakh2100 ──────────────────────────────────────────
    print("\n[2/6] Loading Slakh2100 test tracks...")
    slakh_root = "/Users/paultriana/creative_labs/data/slakh2100/test"
    slakh_samples = []
    for track_dir in sorted(os.listdir(slakh_root)):
        track_path = os.path.join(slakh_root, track_dir)
        if not os.path.isdir(track_path):
            continue
        midi_path = os.path.join(track_path, "all_src.mid")
        stems_dir = os.path.join(track_path, "stems")
        if not os.path.exists(midi_path) or not os.path.exists(stems_dir):
            continue
        try:
            score = Score.from_bytes(open(midi_path, "rb").read())
            stems = sorted(
                os.path.join(stems_dir, f)
                for f in os.listdir(stems_dir) if f.endswith(".flac")
            )
            if stems and score.tracks:
                slakh_samples.append({
                    "score": score,
                    "audio_stems": stems[: len(score.tracks)],
                })
        except Exception:
            continue
    print(f"  Loaded {len(slakh_samples)} Slakh tracks")
    if slakh_samples:
        s0 = slakh_samples[0]
        print(f"  Sample 0: {len(s0['score'].tracks)} tracks, {len(s0['audio_stems'])} stems")

    # ── 4. Encode GigaMIDI — AR mode ───────────────────────────────
    print("\n[3/6] Encoding GigaMIDI sample (AR mode)...")
    from midigpt.augmentation.score_window import select_window
    score0 = giga_scores[0]
    window = select_window(score0, n_bars=4, n_tracks=3, min_fill_ratio=0.75)
    if window:
        opts = _core.EncodeOptions()
        tokens = tokenizer.encode(window, opts=opts)
        print(f"  Tokens ({len(tokens)}): {compact_log(tokens, NAMES)}")
    else:
        print("  Could not select window")

    # ── 5. Encode GigaMIDI — Infill mode ───────────────────────────
    print("\n[4/6] Encoding GigaMIDI sample (Infill mode)...")
    if window:
        infill_bars = {(0, 1), (0, 2), (1, 1), (1, 2)}
        opts = _core.EncodeOptions()
        opts.multi_fill = infill_bars
        tokens = tokenizer.encode(window, opts=opts)
        print(f"  Tokens ({len(tokens)}): {compact_log(tokens, NAMES)}")

    # ── 6. Encode with AudioContext track ───────────────────────────
    print("\n[5/6] Encoding sample with AudioContext track...")
    if window:
        audio_window = copy.deepcopy(window)
        n_bars = len(audio_window.tracks[0].bars)
        audio_window.tracks[2].track_type = "audio"
        audio_window.tracks[2].bars = []
        for _ in range(n_bars):
            empty_bar = Bar()
            empty_bar.ts_numerator = 4
            empty_bar.ts_denominator = 4
            audio_window.tracks[2].bars.append(empty_bar)

        opts = _core.EncodeOptions()
        tokens = tokenizer.encode(audio_window, opts=opts)
        print(f"  AR+Audio ({len(tokens)}): {compact_log(tokens, NAMES)}")

        opts2 = _core.EncodeOptions()
        opts2.multi_fill = {(0, 1), (0, 2)}
        tokens2 = tokenizer.encode(audio_window, opts=opts2)
        print(f"  Infill+Audio ({len(tokens2)}): {compact_log(tokens2, NAMES)}")

    # ── 7. Build datasets + 5 training steps ───────────────────────
    print("\n[6/6] Building datasets + training 5 steps...")
    from midigpt.audio.synth import SoundFontSynthesizer
    from midigpt.training.audio_dataset import (
        AudioTransitionDataset, SlakhPairedStemDataset, MixedMidiGPTDataset,
    )
    from midigpt.training.audio_collator import AudioConditionedCollator
    from torch.utils.data import DataLoader

    sf_synth = SoundFontSynthesizer("soundfonts", sample_rate=16000)
    print(f"  SoundFont pool: {len(sf_synth._sf2_paths)} SF2s")

    giga_dataset = AudioTransitionDataset(
        scores=giga_scores[:20],
        tokenizer=tokenizer,
        synthesizer=sf_synth,
        num_bars_choices=[4, 8],
        infill_prob=0.5,
        audio_track_prob=0.7,
        min_tracks=2,
        max_tracks=6,
        max_seq_len=2048,
    )

    slakh_dataset = SlakhPairedStemDataset(
        samples=slakh_samples[:10],
        tokenizer=tokenizer,
        num_bars_choices=[4, 8],
        infill_prob=0.5,
        audio_track_prob=0.7,
        min_tracks=2,
        max_tracks=6,
        max_seq_len=2048,
    )

    mixed = MixedMidiGPTDataset(giga_dataset, slakh_dataset)
    collator = AudioConditionedCollator(pad_token_id=0, max_seq_len=2048)
    loader = DataLoader(mixed, batch_size=2, shuffle=True, collate_fn=collator, num_workers=0)

    print(f"  Mixed dataset: {len(mixed)} (GigaMIDI: {len(giga_dataset)}, Slakh: {len(slakh_dataset)})")

    for step, batch in enumerate(loader):
        if step >= 5:
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        is_infill = batch["is_infill"]
        audio_pos = batch["audio_bar_positions"]

        print(f"\n  ── Step {step} ──")
        print(f"  is_infill: {is_infill}")
        print(f"  input_ids shape: {input_ids.shape}")

        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            # Find last non-zero token
            last_non_pad = len(ids) - 1
            while last_non_pad > 0 and ids[last_non_pad] == 0:
                last_non_pad -= 1
            ids = ids[: last_non_pad + 1]

            lab = labels[b].tolist()[: len(ids)]
            active = sum(1 for x in lab if x != -100)
            masked = sum(1 for x in lab if x == -100)

            print(f"  Sample {b} ({len(ids)} tokens):")
            print(f"    SEQ: {compact_log(ids, NAMES)}")
            print(f"    LABELS: {active} active, {masked} masked (-100)")

            if audio_pos[b]:
                print(f"    AUDIO_POS: {audio_pos[b]}")
            else:
                print(f"    AUDIO: none")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
