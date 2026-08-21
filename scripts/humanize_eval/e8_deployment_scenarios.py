"""E8 -- realistic deployment scenarios, multi-track (EXPERIMENT_PLAN.md is
single-track only; this generalizes the harness so the checkpoint-matrix
comparison (A/B/C/D/E) has a real cross-track surface to run against).

Scenarios, each scored the same way (Wasserstein vs GT, generated dispersion,
degeneracy rate -- reusing E2's exact definitions):

  whole_track     -- target every bar of ONE track; every other track is
                     real context (today's training/eval regime, generalized
                     to multi-track windows).
  mechanical_ctx  -- same target, but every OTHER track is fully mechanized
                     (the from-scratch-context OOD probe, generalized to
                     multiple context tracks instead of E4's single track).
  mixed_ctx       -- same target, but only SOME other tracks are mechanized
                     (a fraction drawn per piece) -- the realistic partial
                     state a real user's project would actually be in.
  vertical_slice  -- target the same bar index across several tracks (the
                     "polish this bar across the ensemble" pattern).
  whole_piece     -- target every bar of every track (the full from-scratch
                     humanization request, zero real context anywhere).
  track_by_track  -- the actual iterative deployment procedure: start fully
                     mechanical, humanize track 1 (context = other tracks
                     still mechanical), then track 2 (context = track 1's
                     OWN GENERATED output + remaining tracks still
                     mechanical), etc. Reports whether error COMPOUNDS
                     across the sequence (each step's Wasserstein-to-GT vs.
                     its position in the sequence) -- the key open question
                     from the mechanical-context design discussion.

No gates defined (descriptive, cross-checkpoint comparison is the point).

Usage:
    python3 e8_deployment_scenarios.py \\
        --checkpoint <path> \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir <dir> \\
        [--limit N] [--min-tracks 2] [--max-tracks 6] [--temperature 0.85] [--seed 0]
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
from e2_sampling_calibration import (  # noqa: E402
    DEGENERACY_STDEV_THRESHOLD,
    canonical_note_order,
    model_res_values,
    wasserstein_1d,
)

MECH_KWARGS = dict(infill_bars=set())


def pick_multitrack_window(score, n_bars: int, min_tracks: int, max_tracks: int, rng: random.Random):
    from midigpt.augmentation.score_window import select_window

    for n_tracks in range(min(max_tracks, len(score.tracks)), min_tracks - 1, -1):
        window = select_window(score, n_bars, n_tracks, min_fill_ratio=0.0)
        if window is not None and len(window.tracks) >= min_tracks:
            return window
    return None


def eligible_cells(window) -> list[tuple[int, int]]:
    return [
        (t, b)
        for t in range(len(window.tracks))
        for b in range(len(window.tracks[t].bars))
        if window.tracks[t].bars[b].notes
    ]


def mechanize_tracks(window, track_idxs, exclude_bars: set[tuple[int, int]] = frozenset()) -> None:
    from midigpt.augmentation.mechanize import mechanize_bar

    for t in track_idxs:
        for b, bar in enumerate(window.tracks[t].bars):
            if (t, b) not in exclude_bars:
                mechanize_bar(bar, window.resolution)


def score_scenario(engine, tok, window, target_cells: set[tuple[int, int]], temperature: float, rng: random.Random, gt_window=None):
    """Returns (gt_vel, gen_vel) lists (model-resolution levels) for the
    target cells, or None if the scenario has no scoreable notes.

    `window` is what the model actually sees as input (may have mechanized
    context tracks). `gt_window` is what "ground truth" is scored against --
    defaults to `window` when the target track's own bars are untouched
    there (true for whole_track/mechanical_ctx/mixed_ctx/vertical_slice/
    whole_piece, where only OTHER tracks get mechanized). Must be passed
    explicitly whenever the target track's own bars might already be
    mechanized in `window` (track_by_track's evolving `w` starts fully
    mechanical, including bars not yet generated) -- otherwise GT is
    silently extracted from flattened data instead of the real piece.
    """
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

    if gt_window is None:
        gt_window = window
    try:
        gt_tokens = tok.encode(copy.deepcopy(gt_window), compute_attributes=False)
        gt_decoded = tok.decode(gt_tokens)
    except Exception:
        return None

    by_track: dict[int, set[int]] = {}
    for t, b in target_cells:
        by_track.setdefault(t, set()).add(b)
    if not by_track:
        return None

    # Every track in the score needs an explicit TrackPrompt -- non-target
    # tracks are marked ignore=True (pure context, use their real notes
    # as-is) rather than omitted.
    tracks = [
        TrackPrompt(id=t, bars=sorted(by_track[t]), humanize=True) if t in by_track
        else TrackPrompt(id=t, bars=[], ignore=True)
        for t in range(len(window.tracks))
    ]
    req = GenerationRequest(
        tracks=tracks,
        config=InferenceConfig(
            temperature=temperature, top_p=1.0, bars_per_step=max(1, max(len(b) for b in by_track.values())),
            tracks_per_step=max(1, len(by_track)), model_dim=len(window.tracks[0].bars),
            mask_mode="remove", seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
        ),
    )
    try:
        gen_decoded = engine.session(copy.deepcopy(window), req).run()
    except Exception:
        return None

    # model_res_values operates on a single-track-indexed score (tracks[0]);
    # here there may be multiple target tracks, so extract directly.
    from midigpt.tokenizer.tokenizer import resample_delta

    def extract(score_like):
        out = []
        resampled = copy.deepcopy(score_like)
        resample_delta(resampled, resampled.resolution, 12, use_delta=True)
        for t, bars in by_track.items():
            if t >= len(resampled.tracks):
                continue
            for b, bar in enumerate(resampled.tracks[t].bars):
                if b not in bars:
                    continue
                for note in canonical_note_order(bar.notes):
                    vel = min(127, max(0, note.velocity))
                    out.append(vel)
        return out

    gt_vel = extract(gt_decoded)
    gen_vel = extract(gen_decoded)
    if not gt_vel or not gen_vel:
        return None
    return gt_vel, gen_vel


def metrics_for(gt_vel, gen_vel) -> dict:
    gen_stdev = statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0
    return {
        "n_notes": len(gt_vel),
        "vel_wasserstein": wasserstein_1d(gen_vel, gt_vel),
        "gen_stdev": gen_stdev,
        "gt_stdev": statistics.pstdev(gt_vel) if len(gt_vel) >= 2 else 0.0,
        "degenerate": gen_stdev < DEGENERACY_STDEV_THRESHOLD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--min-tracks", type=int, default=2)
    parser.add_argument("--max-tracks", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.engine import InferenceEngine

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
            if len(raw_pieces) >= max(args.limit * 3, 60):
                break
        if len(raw_pieces) >= max(args.limit * 3, 60):
            break
    rng.shuffle(raw_pieces)

    scenario_records: dict[str, list[dict]] = {s: [] for s in
        ["whole_track", "mechanical_ctx", "mixed_ctx", "vertical_slice", "whole_piece"]}
    track_by_track_sequences: list[list[dict]] = []

    n_used = 0
    for raw in raw_pieces:
        if n_used >= args.limit:
            break
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue
        window = pick_multitrack_window(score, args.n_bars, args.min_tracks, args.max_tracks, rng)
        if window is None:
            continue
        n_tracks = len(window.tracks)
        cells = eligible_cells(window)
        if not cells:
            continue
        tracks_with_notes = sorted({t for t, _ in cells})
        if len(tracks_with_notes) < 2:
            continue

        # ---- whole_track: target one track wholly, rest real context ----
        w = copy.deepcopy(window)
        target_track = rng.choice(tracks_with_notes)
        target_cells = {(target_track, b) for b in range(len(w.tracks[target_track].bars)) if w.tracks[target_track].bars[b].notes}
        res = score_scenario(engine, tok, w, target_cells, args.temperature, rng)
        if res:
            scenario_records["whole_track"].append(metrics_for(*res))

        # ---- mechanical_ctx: same target, ALL other tracks mechanized ----
        w = copy.deepcopy(window)
        other_tracks = [t for t in range(n_tracks) if t != target_track]
        mechanize_tracks(w, other_tracks)
        res = score_scenario(engine, tok, w, target_cells, args.temperature, rng)
        if res:
            scenario_records["mechanical_ctx"].append(metrics_for(*res))

        # ---- mixed_ctx: a random subset of other tracks mechanized ------
        w = copy.deepcopy(window)
        if other_tracks:
            k = rng.randint(1, len(other_tracks))
            mixed_mech = rng.sample(other_tracks, k)
            mechanize_tracks(w, mixed_mech)
        res = score_scenario(engine, tok, w, target_cells, args.temperature, rng)
        if res:
            scenario_records["mixed_ctx"].append(metrics_for(*res))

        # ---- vertical_slice: one bar index, across the notes-bearing tracks
        w = copy.deepcopy(window)
        bar_candidates = sorted({b for t, b in cells})
        if bar_candidates:
            b = rng.choice(bar_candidates)
            vslice_cells = {(t, b) for t in tracks_with_notes if w.tracks[t].bars[b].notes}
            if vslice_cells:
                res = score_scenario(engine, tok, w, vslice_cells, args.temperature, rng)
                if res:
                    scenario_records["vertical_slice"].append(metrics_for(*res))

        # ---- whole_piece: every bar of every track (zero real context) --
        w = copy.deepcopy(window)
        res = score_scenario(engine, tok, w, set(cells), args.temperature, rng)
        if res:
            scenario_records["whole_piece"].append(metrics_for(*res))

        # ---- track_by_track: iterative, self-referential context --------
        w = copy.deepcopy(window)
        mechanize_tracks(w, list(range(n_tracks)))  # start fully mechanical
        order = list(tracks_with_notes)
        rng.shuffle(order)
        seq = []
        for step, t in enumerate(order):
            t_cells = {(t, b) for b in range(len(w.tracks[t].bars)) if w.tracks[t].bars[b].notes}
            if not t_cells:
                continue
            res = score_scenario(engine, tok, w, t_cells, args.temperature, rng, gt_window=window)
            if res is None:
                continue
            gt_vel, gen_vel = res
            m = metrics_for(gt_vel, gen_vel)
            m["step"] = step
            seq.append(m)
            # Feed this track's OWN generated output forward as context for
            # subsequent tracks (self-referential, matching real iterative
            # use). score_scenario() above only returned values, not the
            # decoded Score, so regenerate once more to get the Score object
            # to splice back into `w` for the next step.
            from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
            t_bars = sorted(b for _, b in t_cells)
            req = GenerationRequest(
                tracks=[
                    TrackPrompt(id=tt, bars=t_bars, humanize=True) if tt == t
                    else TrackPrompt(id=tt, bars=[], ignore=True)
                    for tt in range(len(w.tracks))
                ],
                config=InferenceConfig(
                    temperature=args.temperature, top_p=1.0, bars_per_step=len(t_cells),
                    tracks_per_step=1, model_dim=len(w.tracks[0].bars), mask_mode="remove",
                    seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
                ),
            )
            try:
                w = engine.session(copy.deepcopy(w), req).run()
            except Exception:
                break
        if seq:
            track_by_track_sequences.append(seq)

        n_used += 1
        if n_used % 20 == 0:
            print(f"  {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used.", flush=True)

    def agg(records: list[dict]) -> dict:
        if not records:
            return {"n": 0}
        return {
            "n": len(records),
            "mean_vel_wasserstein": statistics.mean(r["vel_wasserstein"] for r in records if r["vel_wasserstein"] is not None),
            "mean_gen_stdev": statistics.mean(r["gen_stdev"] for r in records),
            "mean_gt_stdev": statistics.mean(r["gt_stdev"] for r in records),
            "degeneracy_rate": sum(r["degenerate"] for r in records) / len(records),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "n_pieces": n_used,
        "scenarios": {name: agg(recs) for name, recs in scenario_records.items()},
    }

    # track_by_track: does Wasserstein-to-GT grow with step index? (compounding check)
    # Also track degeneracy rate by step -- Wasserstein alone can't distinguish
    # "worse but still varied" from "collapsed to a near-constant output"; a
    # checkpoint that stays flat on Wasserstein could still be collapsing more
    # (or less) at later steps, which is exactly the axis the collapse-vs-
    # precision trade-off (see RESULTS_LOG.md) needs to characterize under
    # iteration, not just single-shot.
    by_step: dict[int, list[float]] = {}
    degenerate_by_step: dict[int, list[bool]] = {}
    for seq in track_by_track_sequences:
        for m in seq:
            if m["vel_wasserstein"] is not None:
                by_step.setdefault(m["step"], []).append(m["vel_wasserstein"])
            degenerate_by_step.setdefault(m["step"], []).append(m["degenerate"])
    summary["track_by_track"] = {
        "n_sequences": len(track_by_track_sequences),
        "mean_wasserstein_by_step": {
            str(k): statistics.mean(v) for k, v in sorted(by_step.items())
        },
        "degeneracy_rate_by_step": {
            str(k): sum(v) / len(v) for k, v in sorted(degenerate_by_step.items())
        },
    }

    (out_dir / "e8_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_dir / 'e8_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
