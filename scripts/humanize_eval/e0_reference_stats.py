"""E0 -- reference stats + encoding reconstruction ceiling (no model, CPU-only).

Round-trips every held-out piece through the real production pipeline
(Tokenizer.normalize_input -> Encoder.encode -> Decoder.decode ->
Tokenizer.normalize_output) and measures:

  1. Onset reconstruction error -- the ceiling on any model's achievable
     microtiming accuracy, independent of the model. Reported in units of one
     12-TPQ cell (the encoder's `resolution`) and, for illustration only, in
     ms assuming a fixed 120 BPM reference tempo (real per-piece tempo is not
     preserved through decode, so it is not used for the ms conversion).
  2. GT marginals for VelocityLevel (`VelocityQuantizer(32).encode`, applied
     directly to every note -- not just sticky-emitted tokens) and Delta,
     both pre-clamp (the raw resample_delta residual before the encoder's
     domain-size clamp) and post-clamp (what the model actually trains on).
  3. Per-piece velocity-level dispersion (population stdev) and Shannon
     entropy.
  4. The same reconstruction-error measurement under a signed
     round-to-nearest delta instead of the current truncating resample --
     computed analytically (no retraining, no C++ changes) to check whether
     the representation itself is the binding constraint on microtiming
     quality (see scripts/humanize_eval/EXPERIMENT_PLAN.md §B).
  5. A delta-histogram breakdown by each piece's native (source) resolution,
     to check whether the second raw-delta mode is a resampling artifact
     that appears only at certain source TPQs.

Usage:
    python3 e0_reference_stats.py \\
        --parquet $SCRATCH/MIDI-GPT/data/humanize_filtered/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e0 \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def canonical_note_order(notes):
    """Group by onset_ticks, sort each onset-group by pitch.

    Mirrors encoder.cpp's own chord ordering ("emit chord notes in ascending
    pitch order"). window.tracks[*].bars[*].notes is in raw file order, which
    does NOT match that -- zipping the two directly silently pairs unrelated
    notes within any chord and reports their onset difference as "error".
    """
    by_onset: defaultdict[int, list] = defaultdict(list)
    for n in notes:
        by_onset[n.onset_ticks].append(n)
    out = []
    for onset in sorted(by_onset):
        out.extend(sorted(by_onset[onset], key=lambda n: n.pitch))
    return out

REF_TEMPO_US_PER_QUARTER = 500_000  # 120 BPM, for ms conversion only


def velocity_encode(v: int, num_levels: int = 32) -> int:
    """Direct port of VelocityQuantizer::encode (domain_transforms.h:105-109)."""
    if v <= 0:
        return 0
    if v >= 127:
        return num_levels - 1
    return min(1 + v * (num_levels - 1) // 128, num_levels - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Imports after argparse so --help works without the module stack loaded.
    import midigpt._core as _core
    from midigpt._types import Score
    from midigpt.augmentation.score_window import select_window
    from midigpt.tokenizer.tokenizer import Tokenizer

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    tok = Tokenizer(enc_cfg)
    vocab = tok._vocab
    resolution = enc_cfg.resolution  # 12
    num_levels = enc_cfg.velocity_levels  # 32
    max_delta = vocab.domain_size(_core.TokenType.Delta) - 1  # 5
    # encode() rejects any window whose bar count isn't one of these (grammar
    # constraint tied to the NumBars token) -- dataset.py's training path goes
    # through select_window() for the same reason; we must too.
    num_bars_choices = sorted(set(json.loads(enc_cfg.to_json()).get("num_bars_map") or [4]), reverse=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onset_err_cells: list[float] = []          # current pipeline, real round-trip
    onset_err_cells_signed: list[float] = []   # counterfactual, analytic
    delta_pre_clamp: Counter = Counter()       # raw resample_delta residual (0..resolution)
    delta_post_clamp: Counter = Counter()      # what the model actually sees (0..max_delta)
    delta_direction_count = 0
    delta_total_notes = 0
    vel_level_marginal: Counter = Counter()
    delta_by_tpq: defaultdict[int, Counter] = defaultdict(Counter)
    per_piece_vel_dispersion: list[float] = []
    per_piece_vel_entropy: list[float] = []
    n_pieces = 0
    n_pieces_note_mismatch = 0
    n_notes_total = 0

    pf = pq.ParquetFile(args.parquet)
    row_i = 0
    done = False
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        if done:
            break
        for music in batch.column("music"):
            row_i += 1
            if args.limit is not None and row_i > args.limit:
                done = True
                break
            raw = bytes(music.as_py())
            try:
                score = Score.from_bytes(raw)
            except Exception:
                continue
            if not score.tracks:
                continue
            n_pieces += 1
            R_native = score.resolution

            # --- pre-clamp delta + velocity marginal, computed directly on raw notes ---
            norm = tok.normalize_input(score)  # deep-copies; resamples to `resolution`, truncating
            piece_vel_levels: list[int] = []
            for track in norm.tracks:
                for bar in track.bars:
                    for note in bar.notes:
                        n_notes_total += 1
                        lvl = velocity_encode(note.velocity, num_levels)
                        vel_level_marginal[lvl] += 1
                        piece_vel_levels.append(lvl)

                        d = note.delta  # residual at `resolution`, in [0, resolution) by construction
                        delta_pre_clamp[d] += 1
                        delta_by_tpq[R_native][d] += 1
                        clamped = min(d, max_delta) if d > 0 else 0
                        delta_post_clamp[clamped] += 1
                        delta_total_notes += 1
                        # DeltaDirection would fire iff d < 0 pre-clamp (encoder.cpp:76) --
                        # structurally impossible here since resample_delta truncates to >=0.
                        if d < 0:
                            delta_direction_count += 1

            if len(piece_vel_levels) >= 2:
                per_piece_vel_dispersion.append(statistics.pstdev(piece_vel_levels))
                counts = Counter(piece_vel_levels)
                total = len(piece_vel_levels)
                h = -sum((c / total) * math.log2(c / total) for c in counts.values())
                per_piece_vel_entropy.append(h)

            # --- real round trip: true native position vs decoded position, both in
            # units of one `resolution`-cell (independent of tempo). encode()
            # requires a bar count from num_bars_map (NumBars grammar
            # constraint), so pick a valid window the same way dataset.py does.
            window = None
            for n_bars in num_bars_choices:
                window = select_window(score, n_bars, len(score.tracks), min_fill_ratio=0.75)
                if window is not None:
                    break
            if window is None:
                continue

            try:
                tokens = tok.encode(window, compute_attributes=False)
                decoded = tok.decode(tokens)
            except Exception:
                continue

            if len(decoded.tracks) != len(window.tracks):
                n_pieces_note_mismatch += 1
                continue
            mismatch = False
            for t_orig, t_dec in zip(window.tracks, decoded.tracks):
                if len(t_orig.bars) != len(t_dec.bars):
                    mismatch = True
                    break
                for b_orig, b_dec in zip(t_orig.bars, t_dec.bars):
                    if len(b_orig.notes) != len(b_dec.notes):
                        mismatch = True
                        break
                    orig_notes = canonical_note_order(b_orig.notes)
                    dec_notes = canonical_note_order(b_dec.notes)
                    for n_orig, n_dec in zip(orig_notes, dec_notes):
                        true_pos_cells = (n_orig.onset_ticks + n_orig.delta / R_native) * resolution / R_native
                        dec_pos_cells = (
                            (n_dec.onset_ticks + n_dec.delta / decoded.resolution)
                            * resolution
                            / decoded.resolution
                        )
                        onset_err_cells.append(dec_pos_cells - true_pos_cells)
                        # Counterfactual: signed round-to-nearest at `resolution`,
                        # instead of the current truncating resample -- analytic,
                        # no encode/decode needed (see EXPERIMENT_PLAN.md §B). Must
                        # quantize the residual into the *same* Delta token domain
                        # (magnitude 0..max_delta, in units of 1/resolution of one
                        # cell -- see A3) rather than comparing against the raw
                        # continuous residual, or this isn't a fair comparison
                        # against the current pipeline's real, quantized round trip.
                        nearest_cell = round(true_pos_cells)
                        raw_residual_cells = true_pos_cells - nearest_cell  # in [-0.5, 0.5)
                        residual_token_units = raw_residual_cells * resolution
                        sign = -1 if residual_token_units < 0 else 1
                        magnitude = min(round(abs(residual_token_units)), max_delta)
                        fixed_pos_cells = nearest_cell + sign * magnitude / resolution
                        onset_err_cells_signed.append(fixed_pos_cells - true_pos_cells)
                if mismatch:
                    break
            if mismatch:
                n_pieces_note_mismatch += 1

    def pctile(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        xs_sorted = sorted(xs)
        k = (len(xs_sorted) - 1) * p
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return xs_sorted[int(k)]
        return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)

    def abs_stats(xs: list[float]) -> dict:
        abs_xs = [abs(x) for x in xs]
        ms_per_cell = (REF_TEMPO_US_PER_QUARTER / 1000) / resolution
        return {
            "n": len(xs),
            "mean_cells": statistics.mean(abs_xs) if abs_xs else float("nan"),
            "median_cells": statistics.median(abs_xs) if abs_xs else float("nan"),
            "p90_cells": pctile(abs_xs, 0.90),
            "mean_ms_at_120bpm": (statistics.mean(abs_xs) if abs_xs else float("nan")) * ms_per_cell,
            "median_ms_at_120bpm": (statistics.median(abs_xs) if abs_xs else float("nan")) * ms_per_cell,
            "p90_ms_at_120bpm": pctile(abs_xs, 0.90) * ms_per_cell,
        }

    def hist_pct(counter: Counter) -> dict:
        total = sum(counter.values())
        return {str(k): round(100 * v / total, 2) for k, v in sorted(counter.items())} if total else {}

    summary = {
        "n_pieces": n_pieces,
        "n_pieces_note_count_mismatch": n_pieces_note_mismatch,
        "n_notes_total": n_notes_total,
        "reconstruction_error_current_pipeline": abs_stats(onset_err_cells),
        "reconstruction_error_signed_roundtonearest_counterfactual": abs_stats(onset_err_cells_signed),
        "delta_direction_fire_rate_pct": (
            round(100 * delta_direction_count / delta_total_notes, 4) if delta_total_notes else None
        ),
        "delta_post_clamp_histogram_pct": hist_pct(delta_post_clamp),
        "delta_pre_clamp_histogram_pct": hist_pct(delta_pre_clamp),
        "delta_by_source_tpq": {
            str(tpq): hist_pct(c) for tpq, c in sorted(delta_by_tpq.items())
        },
        "velocity_level_marginal_pct": hist_pct(vel_level_marginal),
        "per_piece_velocity_level_pstdev": {
            "mean": statistics.mean(per_piece_vel_dispersion) if per_piece_vel_dispersion else None,
            "median": statistics.median(per_piece_vel_dispersion) if per_piece_vel_dispersion else None,
        },
        "per_piece_velocity_level_entropy_bits": {
            "mean": statistics.mean(per_piece_vel_entropy) if per_piece_vel_entropy else None,
            "median": statistics.median(per_piece_vel_entropy) if per_piece_vel_entropy else None,
            "max_possible_bits": math.log2(num_levels),
        },
    }

    with open(out_dir / "e0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
