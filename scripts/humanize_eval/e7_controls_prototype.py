"""E7 -- controls prototype (EXPERIMENT_PLAN.md section E7 / sec:C-H).
Informal, cheap, last, and explicitly NOT a metric: produces a small
alpha x tau grid of audio-ready MIDI files for listening.

Because Humanize output is a pure residual on a fixed skeleton, a "strength"
control doesn't need a sampling trick -- it's a post-hoc alpha-blend between
a mechanical anchor and the generated (humanized) notes:

  mechanical anchor: constant velocity (80) + onset snapped to the nearest
  of the 12 model-resolution grid cells per bar (i.e. no microtiming --
  "robotic"). This is a real quantization of onset_ticks, not just zeroing
  the `delta` attribute (delta alone doesn't affect exported MIDI; only
  onset_ticks does -- confirmed by reading resample_delta()).

  generated: the model's actual sampled velocity/onset at temperature tau.

  blended_velocity = round((1-alpha)*80 + alpha*generated_velocity)
  blended_onset    = round((1-alpha)*mechanical_onset + alpha*generated_onset)

alpha=0 -> fully mechanical/robotic, alpha=1 -> full generated expressiveness,
continuous and monotone in between. tau is a separate axis (variety), so the
full grid is alpha x tau.

Context bars (outside the humanized target) are left exactly as GT throughout
-- only the target bars are blended -- so each file is a real, playable
excerpt, not an isolated snippet.

Usage:
    python3 e7_controls_prototype.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny-20260807-035822/model_final.safetensors \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e7_controls \\
        [--n-pieces 4] [--temperatures 0.7,1.0] [--alphas 0.0,0.25,0.5,0.75,1.0] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from e2_sampling_calibration import pick_window_and_targets  # noqa: E402


def mechanical_onset(note, bar, resolution: int, n_cells: int = 12) -> float:
    bar_len = bar.beat_length * resolution
    if bar_len <= 0:
        return float(note.onset_ticks)
    cell_width = bar_len / n_cells
    cell = round(note.onset_ticks / cell_width)
    return cell * cell_width


def blend_target_bars(context_score, generated_score, target_bars: set[int], alpha: float, resolution: int):
    """context_score supplies the mechanical anchor's pitch/duration/bar
    structure (typically the GT-decoded window); generated_score supplies
    the sampled velocity/onset for the same target bars. Returns a new
    Score with target bars blended and all other bars taken from
    context_score untouched."""
    out = copy.deepcopy(context_score)
    for b, (out_bar, gen_bar) in enumerate(zip(out.tracks[0].bars, generated_score.tracks[0].bars)):
        if b not in target_bars:
            continue
        gen_by_pitch_onset = {(n.pitch, n.duration_ticks): n for n in gen_bar.notes}
        for note in out_bar.notes:
            mech_onset = mechanical_onset(note, out_bar, resolution)
            mech_vel = 80
            gen_note = gen_by_pitch_onset.get((note.pitch, note.duration_ticks))
            gen_onset = gen_note.onset_ticks if gen_note is not None else note.onset_ticks
            gen_vel = gen_note.velocity if gen_note is not None else mech_vel
            note.onset_ticks = max(0, round((1 - alpha) * mech_onset + alpha * gen_onset))
            note.velocity = max(1, min(127, round((1 - alpha) * mech_vel + alpha * gen_vel)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-pieces", type=int, default=4)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--temperatures", default="0.7,1.0")
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
    from midigpt.inference.engine import InferenceEngine

    temperatures = [float(x) for x in args.temperatures.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    tok = engine._tokenizer

    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
            if len(raw_pieces) >= 300:
                break
        if len(raw_pieces) >= 300:
            break
    rng.shuffle(raw_pieces)

    manifest_lines = ["# E7 -- controls prototype (informal listening set, not a metric)\n",
                       "| file | piece | tau | alpha |", "|---|---|---|---|"]
    n_written = 0
    for raw in raw_pieces:
        if n_written >= args.n_pieces:
            break
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue
        try:
            picked = pick_window_and_targets(score, args.n_bars, args.coverage, rng)
        except Exception:
            continue
        if picked is None:
            continue
        window, target_bars = picked

        try:
            gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
            gt_decoded = tok.decode(gt_tokens)
        except Exception:
            continue
        if not any(gt_decoded.tracks[0].bars[b].notes for b in target_bars):
            continue

        piece_idx = n_written
        for tau in temperatures:
            req = GenerationRequest(
                tracks=[TrackPrompt(id=0, bars=sorted(target_bars), humanize=True)],
                config=InferenceConfig(
                    temperature=tau, top_p=args.top_p, bars_per_step=max(1, args.n_bars),
                    tracks_per_step=1, model_dim=args.n_bars, mask_mode="remove",
                    seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
                ),
            )
            try:
                gen_decoded = engine.session(copy.deepcopy(window), req).run()
            except Exception as exc:
                print(f"  [skip] generation failed piece={piece_idx} tau={tau}: {exc}", flush=True)
                continue

            for alpha in alphas:
                blended = blend_target_bars(gt_decoded, gen_decoded, target_bars, alpha, gt_decoded.resolution)
                fname = f"piece{piece_idx:02d}_tau{tau:.2f}_alpha{alpha:.2f}.mid"
                blended.to_midi(str(out_dir / fname))
                manifest_lines.append(f"| {fname} | {piece_idx} | {tau} | {alpha} |")
        n_written += 1
        print(f"  wrote grid for piece {piece_idx} ({n_written}/{args.n_pieces})", flush=True)

    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines))
    print(f"Done: {n_written} pieces, files under {out_dir}", flush=True)
    print(f"Manifest: {out_dir / 'MANIFEST.md'}", flush=True)


if __name__ == "__main__":
    main()
