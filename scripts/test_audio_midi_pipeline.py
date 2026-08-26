"""Verification script for GigaMIDI on-the-fly synthesis and Slakh2100 pre-rendered stem training pipelines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from midigpt._types import Bar, Note, Score, Track
from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.audio import FastAudioSynthesizer, load_audio_extractor
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.audio_dataset import (
    AudioTransitionDataset,
    SlakhPairedStemDataset,
)


ENCODER_JSON = "models/yellow_encoder.json"
DEFAULT_SLAKH_ROOT = Path(__file__).resolve().parents[2] / "data" / "slakh2100" / "test"


def create_mock_score(num_bars: int = 16, num_tracks: int = 3) -> Score:
    tracks = []
    tracks.append(Track(
        bars=[Bar(notes=[
            Note(pitch=60+(b%4)*2, velocity=80, onset_ticks=0, duration_ticks=120),
            Note(pitch=64+(b%4)*2, velocity=85, onset_ticks=120, duration_ticks=120),
            Note(pitch=67+(b%4)*2, velocity=75, onset_ticks=240, duration_ticks=120),
            Note(pitch=71+(b%4)*2, velocity=90, onset_ticks=360, duration_ticks=120),
        ], ts_numerator=4, ts_denominator=4) for b in range(num_bars)],
        instrument=0, track_type="melodic",
    ))
    tracks.append(Track(
        bars=[Bar(notes=[
            Note(pitch=36+(b%4)*2, velocity=95, onset_ticks=0, duration_ticks=240),
            Note(pitch=43+(b%4)*2, velocity=90, onset_ticks=240, duration_ticks=240),
        ], ts_numerator=4, ts_denominator=4) for b in range(num_bars)],
        instrument=33, track_type="melodic",
    ))
    tracks.append(Track(
        bars=[Bar(notes=[
            Note(pitch=36, velocity=100, onset_ticks=0, duration_ticks=60),
            Note(pitch=38, velocity=90, onset_ticks=120, duration_ticks=60),
            Note(pitch=42, velocity=80, onset_ticks=240, duration_ticks=60),
            Note(pitch=38, velocity=95, onset_ticks=360, duration_ticks=60),
        ], ts_numerator=4, ts_denominator=4) for b in range(num_bars)],
        instrument=0, track_type="drum",
    ))
    return Score(tracks=tracks, resolution=480, tempo=500000)


def build_tokenizer() -> tuple[Tokenizer, list[int]]:
    """Load tokenizer and return (tokenizer, num_bars_map) from JSON config."""
    import midigpt._core as _core
    raw = Path(ENCODER_JSON).read_text()
    num_bars_map = json.loads(raw).get("num_bars_map", [4, 8])
    enc_cfg = _core.EncoderConfig.from_json(raw)
    return Tokenizer(enc_cfg), num_bars_map


def compute_loss_manually(
    model: GPT2LMHeadModel,
    adapter: AudioToMidiAdapter,
    sample: dict,
) -> torch.Tensor:
    """Forward pass + loss without Lightning."""
    input_ids = sample["input_ids"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    soft_replacements = None
    bar_frames_list = sample.get("context_bar_frames", [])
    track_frames_list = sample.get("track_frames", [])

    if bar_frames_list:
        soft_replacements = []
        track_frames = track_frames_list[0] if track_frames_list else None
        for b_idx, b_frames in enumerate(bar_frames_list):
            if b_frames.ndim == 2:
                b_frames = b_frames.unsqueeze(0)
            t_frames_in = (
                track_frames.unsqueeze(0)
                if (track_frames is not None and track_frames.ndim == 2)
                else track_frames
            )
            with torch.no_grad():
                soft_emb = adapter(b_frames, track_frames=t_frames_in)
            soft_replacements.append((b_idx, soft_emb))

    with torch.no_grad():
        logits, _ = model(input_ids=input_ids, soft_token_replacements=soft_replacements)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ---------------------------------------------------------------------------
# Test 1: GigaMIDI on-the-fly synthesis
# ---------------------------------------------------------------------------
def test_gigamidi_on_the_fly_pipeline(extractor_type: str, weights_path: str | None) -> None:
    print("\n" + "="*70)
    print(f"1. GigaMIDI ON-THE-FLY SYNTHESIS  [extractor={extractor_type}]")
    print("="*70)

    test_midi_path = "tmp/midi_examples/Summer.mid"
    if os.path.exists(test_midi_path):
        try:
            score = Score.from_midi(test_midi_path)
            print(f"Loaded real MIDI: {len(score.tracks)} tracks, {len(score.tracks[0].bars)} bars")
        except Exception as e:
            print(f"Fallback to mock score ({e})")
            score = create_mock_score()
    else:
        print("Using mock score (16 bars, 3 tracks).")
        score = create_mock_score()

    tokenizer, num_bars_map = build_tokenizer()
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size}  |  Valid bar lengths: {num_bars_map}")

    extractor = load_audio_extractor(extractor_type, weights_path=weights_path, feature_dim=768)
    print(f"Extractor: {type(extractor).__name__}")

    # Use only bar lengths <= total bars in score (and valid per config)
    total_bars = min((len(t.bars) for t in score.tracks), default=0)
    valid_choices = [b for b in num_bars_map if b <= total_bars] or [num_bars_map[0]]
    print(f"Bar length choices for this score: {valid_choices}")

    dataset = AudioTransitionDataset(
        scores=[score],
        tokenizer=tokenizer,
        audio_extractor=extractor,
        num_bars_choices=valid_choices,
        infill_prob=0.5,
        audio_track_prob=0.6,
    )

    print("Generating on-the-fly training excerpt...")
    sample = dataset[0]

    print(f"  Input token length   : {len(sample['input_ids'])}")
    print(f"  Active loss tokens   : {(sample['labels'] != -100).sum().item()}")
    print(f"  Context bar frames   : {len(sample['context_bar_frames'])}")
    if sample["context_bar_frames"]:
        print(f"  Bar frame shape      : {list(sample['context_bar_frames'][0].shape)}")
    print(f"  Mode                 : {'Infill' if sample['is_infill'] else 'Autoregressive'}")

    cfg = GPT2Config(vocab_size=vocab_size, n_embd=512, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(cfg).eval()
    adapter = AudioToMidiAdapter(feature_dim=768, model_dim=512, pooling_strategy="perceiver", fusion_strategy="concat").eval()

    loss = compute_loss_manually(model, adapter, sample)
    print(f"  Loss                 : {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    print("✓ GigaMIDI on-the-fly pipeline PASSED!")


# ---------------------------------------------------------------------------
# Test 2: Slakh2100 pre-rendered stems
# ---------------------------------------------------------------------------
def test_slakh_pre_rendered_stem_pipeline(
    extractor_type: str,
    weights_path: str | None,
    slakh_root: Path,
) -> None:
    print("\n" + "="*70)
    print(f"2. Slakh2100 PRE-RENDERED STEMS  [extractor={extractor_type}]")
    print("="*70)

    track_dirs = []
    for midi_path in sorted(slakh_root.glob("Track*/all_src.mid")):
        candidate = midi_path.parent
        if not (candidate / "stems").is_dir():
            continue
        candidate_score = Score.from_midi(str(midi_path))
        if all((candidate / "stems" / f"S{index:02d}.flac").is_file()
               for index in range(len(candidate_score.tracks))):
            track_dirs.append(candidate)
    if not track_dirs:
        raise FileNotFoundError(
            f"No Slakh tracks with all_src.mid and complete aligned stems found under {slakh_root}. "
            "Pass --slakh-root to a Slakh2100 test split."
        )

    track_dir = track_dirs[0]
    score = Score.from_midi(str(track_dir / "all_src.mid"))
    stems = [track_dir / "stems" / f"S{index:02d}.flac" for index in range(len(score.tracks))]
    print(f"Using {track_dir.name}: {len(score.tracks)} MIDI tracks with aligned FLAC stems")

    tokenizer, num_bars_map = build_tokenizer()
    vocab_size = tokenizer.vocab_size()
    extractor = load_audio_extractor(extractor_type, weights_path=weights_path, feature_dim=768)

    total_bars = min((len(t.bars) for t in score.tracks), default=0)
    valid_choices = [b for b in num_bars_map if b <= total_bars] or [num_bars_map[0]]

    dataset = SlakhPairedStemDataset(
        samples=[{"score": score, "audio_stems": [str(path) for path in stems]}],
        tokenizer=tokenizer,
        audio_extractor=extractor,
        num_bars_choices=valid_choices,
        transpose_range=(-4, 4),
    )

    print("Generating training excerpt with synchronized MIDI+audio pitch augmentation...")
    sample = dataset[0]

    print(f"  Input token length   : {len(sample['input_ids'])}")
    print(f"  Context bar frames   : {len(sample['context_bar_frames'])}")
    if sample["context_bar_frames"]:
        print(f"  Bar frame shape      : {list(sample['context_bar_frames'][0].shape)}")

    cfg = GPT2Config(vocab_size=vocab_size, n_embd=512, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(cfg).eval()
    adapter = AudioToMidiAdapter(feature_dim=768, model_dim=512).eval()

    loss = compute_loss_manually(model, adapter, sample)
    print(f"  Loss                 : {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    print("✓ Slakh2100 pre-rendered stem pipeline PASSED!")


# ---------------------------------------------------------------------------
# Test 3: Multitrack checkpoint on single-track audio
# ---------------------------------------------------------------------------
def test_multitrack_checkpoint_single_track(weights_path: str | None) -> None:
    print("\n" + "="*70)
    print("3. MULTITRACK CHECKPOINT → SINGLE TRACK  [yourmt3 vs yourmt3+]")
    print("="*70)

    score = create_mock_score(num_bars=8)
    synth = FastAudioSynthesizer(sample_rate=16000)
    wav_single = synth.render_track(score, 0)
    print(f"Single-track waveform shape: {list(wav_single.shape)}")

    from midigpt.audio.synth import get_bar_intervals
    bar_intervals = get_bar_intervals(score)

    extractor_single = load_audio_extractor("yourmt3", weights_path=None, feature_dim=768)
    extractor_multi  = load_audio_extractor("yourmt3+", weights_path=weights_path, feature_dim=768)

    print("  Single-track extractor: extracting bar frames...")
    frames_s = extractor_single.extract_bar_frames(wav_single, sample_rate=16000, bar_intervals=bar_intervals)
    print("  Multi-track extractor (single-track input): extracting bar frames...")
    frames_m = extractor_multi.extract_bar_frames(wav_single, sample_rate=16000, bar_intervals=bar_intervals)

    print(f"  Single ckpt: {len(frames_s)} bar frames, shape {list(frames_s[0].shape) if frames_s else 'N/A'}")
    print(f"  Multi  ckpt: {len(frames_m)} bar frames, shape {list(frames_m[0].shape) if frames_m else 'N/A'}")

    if frames_s and frames_m:
        f_s = frames_s[0].mean(dim=0)
        f_m = frames_m[0].mean(dim=0)
        cos_sim = F.cosine_similarity(f_s.unsqueeze(0), f_m.unsqueeze(0)).item()
        print(f"  Cosine similarity (bar 0, single vs multi): {cos_sim:.4f}")

    print("✓ Multi-track checkpoint single-track comparison PASSED!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify audio+MIDI training pipeline.")
    parser.add_argument("--extractor", default="yourmt3",
                        choices=["yourmt3", "yourmt3+", "muscriptor"],
                        help="Audio extractor type (default: yourmt3).")
    parser.add_argument("--weights", default=None,
                        help="Path to extractor checkpoint (optional).")
    parser.add_argument("--skip-multitrack", action="store_true",
                        help="Skip the multitrack checkpoint comparison test.")
    parser.add_argument("--slakh-root", type=Path, default=DEFAULT_SLAKH_ROOT,
                        help="Path to the Slakh2100 test split containing Track*/ directories.")
    args = parser.parse_args()

    test_gigamidi_on_the_fly_pipeline(args.extractor, args.weights)
    test_slakh_pre_rendered_stem_pipeline(args.extractor, args.weights, args.slakh_root)
    if not args.skip_multitrack:
        test_multitrack_checkpoint_single_track(args.weights)

    print("\n" + "="*70)
    print("ALL PIPELINE VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
