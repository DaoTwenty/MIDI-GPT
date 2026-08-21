"""E10 -- diverse before/after listening examples for the unconditioned
(base) Humanize checkpoint. Informal, not a metric: produces real, playable
MIDI files so a human can actually listen, not just read numbers.

For each selected piece, writes THREE files sharing the same pitch/duration/
bar structure:

  before    -- every track mechanized (constant velocity=80, onset snapped
              to the nearest of the 12 model-resolution grid cells per bar --
              same "robotic input" anchor E7 uses, a real quantization of
              onset_ticks, not just zeroing an attribute).
  after     -- the model's humanized regeneration of every bar/track, given
              the mechanized version as input (the actual whole_piece
              deployment scenario -- zero real context, matching E8's
              hardest/most realistic cold-start case).
  reference -- the ORIGINAL, unmodified ground-truth performance (never
              touched) -- lets a listener judge how close the model's
              stylistic choices land relative to a real human performance,
              not just whether "after" sounds more alive than "before".

Piece selection is deliberately spread across track count (solo up through
full ensemble) and `music_style_scraped` (a raw GigaMIDI metadata column,
never used for conditioning -- see CLAUDE.md/RESULTS_LOG's genre-unusable
finding -- but real, informative data for picking a genuinely diverse set).

Also writes a quick expressive-token analysis per piece/condition: mean
velocity, velocity std (dynamic variability), and mean onset deviation from
the nearest grid cell in ticks (a "looseness" measure, 0 by construction for
`before`) -- so the listening set comes with numbers, not just audio.

Usage:
    python3 e10_listening_examples.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny_e-20260807-223945/model_final.safetensors \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e10_listening_e \\
        [--n-pieces 8] [--n-bars 8] [--temperature 0.9] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from e7_controls_prototype import mechanical_onset  # noqa: E402
from e8_deployment_scenarios import mechanize_tracks, pick_multitrack_window  # noqa: E402


# (min_tracks, max_tracks) targets to sample across, for real instrumentation
# spread rather than whatever the random shuffle happens to hand back.
TRACK_COUNT_BUCKETS = [(1, 1), (1, 1), (2, 3), (2, 3), (3, 5), (3, 5), (4, 8), (4, 8)]


def token_stats(score, resolution: int) -> dict:
    """Mean velocity / velocity std / mean |onset deviation from the
    nearest 12-cell grid point| across every note in every track/bar --
    the same expressive dimensions Humanize actually regenerates."""
    velocities: list[int] = []
    deviations: list[float] = []
    for track in score.tracks:
        for bar in track.bars:
            for note in bar.notes:
                velocities.append(note.velocity)
                grid_onset = mechanical_onset(note, bar, resolution)
                deviations.append(abs(note.onset_ticks - grid_onset))
    if not velocities:
        return {"mean_velocity": None, "velocity_std": None, "mean_onset_deviation_ticks": None}
    return {
        "mean_velocity": round(statistics.mean(velocities), 2),
        "velocity_std": round(statistics.pstdev(velocities), 2) if len(velocities) > 1 else 0.0,
        "mean_onset_deviation_ticks": round(statistics.mean(deviations), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-pieces", type=int, default=8)
    parser.add_argument("--n-bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
    from midigpt.inference.engine import InferenceEngine

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    tok = engine._tokenizer

    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    rows: list[tuple[bytes, str, int]] = []
    for batch in pf.iter_batches(columns=["music", "music_style_scraped", "num_tracks"], batch_size=500):
        for music, style, n_tracks in zip(
            batch.column("music"), batch.column("music_style_scraped"), batch.column("num_tracks"), strict=True
        ):
            rows.append((bytes(music.as_py()), style.as_py() or "unknown", int(n_tracks.as_py() or 0)))
        if len(rows) >= 4000:
            break
    rng.shuffle(rows)

    manifest_lines = [
        "# E10 -- before/after listening examples (informal, not a metric)\n",
        f"Checkpoint: `{args.checkpoint}`\n",
        "| piece | style | n_tracks | file | condition |",
        "|---|---|---|---|---|",
    ]
    analysis: list[dict] = []
    seen_styles: set[str] = set()

    n_written = 0
    bucket_i = 0
    for raw, style, approx_n_tracks in rows:
        if n_written >= args.n_pieces:
            break
        min_t, max_t = TRACK_COUNT_BUCKETS[bucket_i % len(TRACK_COUNT_BUCKETS)]
        if not (min_t <= approx_n_tracks <= max_t + 2):  # +2 slack since select_window may trim
            continue
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue

        window = pick_multitrack_window(score, args.n_bars, min_t, max_t, rng)
        if window is None:
            continue

        try:
            gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
            gt_decoded = tok.decode(gt_tokens)
        except Exception:
            continue
        if not any(bar.notes for track in gt_decoded.tracks for bar in track.bars):
            continue

        mechanized = copy.deepcopy(gt_decoded)
        mechanize_tracks(mechanized, list(range(len(mechanized.tracks))))

        n_bars_actual = len(mechanized.tracks[0].bars)
        req = GenerationRequest(
            tracks=[
                TrackPrompt(id=t, bars=list(range(n_bars_actual)), humanize=True)
                for t in range(len(mechanized.tracks))
            ],
            config=InferenceConfig(
                temperature=args.temperature, top_p=args.top_p, bars_per_step=n_bars_actual,
                tracks_per_step=len(mechanized.tracks), model_dim=n_bars_actual,
                mask_mode="remove", seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
            ),
        )
        try:
            humanized = engine.session(copy.deepcopy(mechanized), req).run()
        except Exception as exc:
            print(f"  [skip] generation failed piece={n_written} style={style}: {exc}", flush=True)
            continue

        piece_id = f"piece{n_written:02d}"
        files = {
            "before_mechanical": mechanized,
            "after_humanized": humanized,
            "reference_ground_truth": gt_decoded,
        }
        for cond, sc in files.items():
            fname = f"{piece_id}_{cond}.mid"
            sc.to_midi(str(out_dir / fname))
            manifest_lines.append(
                f"| {piece_id} | {style} | {len(gt_decoded.tracks)} | {fname} | {cond} |"
            )

        stats = {cond: token_stats(sc, gt_decoded.resolution) for cond, sc in files.items()}
        analysis.append({"piece": piece_id, "style": style, "n_tracks": len(gt_decoded.tracks), "stats": stats})

        seen_styles.add(style)
        n_written += 1
        bucket_i += 1
        print(f"  wrote {piece_id} (style={style}, n_tracks={len(gt_decoded.tracks)}) [{n_written}/{args.n_pieces}]", flush=True)

    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines))
    (out_dir / "expressive_token_analysis.json").write_text(json.dumps(analysis, indent=2))

    print(f"\nDone: {n_written} pieces ({len(seen_styles)} distinct styles: {sorted(seen_styles)})", flush=True)
    print(f"Files under: {out_dir}", flush=True)
    print(f"Manifest: {out_dir / 'MANIFEST.md'}", flush=True)
    print(f"Analysis: {out_dir / 'expressive_token_analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
