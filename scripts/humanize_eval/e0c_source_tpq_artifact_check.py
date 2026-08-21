"""E0c -- does the true source-file tpq (not the 480 the reader canonicalizes
to) explain the pre-clamp Delta bimodal mode via pure resampling arithmetic?

MidiReader is always constructed with its default resolution=480
(midi_reader.h: `explicit MidiReader(int resolution = 480)`), so
score.resolution==480 for every piece says nothing about the source files'
real native PPQ -- that value (`tpq` in midi_reader.cpp) is read straight
from each MIDI file's header and used internally, then discarded once the
Score is canonicalized to 480. This script recovers it directly from the raw
MIDI header bytes (standard MThd chunk, division field at bytes 12:14) and
checks whether it predicts each note's residual via pure arithmetic alone.

Hypothesis under test (user's): note.delta's bimodal pattern is an artifact
of specific source tpq / 480 ratios, not genuine performance. For a note
quantized exactly to the source file's native tick grid (no real performance
jitter at all), converting tick position -> 480 -> (eventually) the 144
delta micro-grid is a deterministic function of (tick mod period), where
period = tpq / gcd(tpq, 480). If real per-tpq-bucket residual histograms
closely match the all-notes-exactly-on-grid deterministic prediction, that's
strong evidence the observed pattern is arithmetic, not expressive.

Usage:
    python3 e0c_source_tpq_artifact_check.py \\
        --parquet $SCRATCH/MIDI-GPT/data/humanize_filtered/validation.parquet \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e0c \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

TARGET_RES = 480  # MidiReader's hardcoded default


def read_midi_tpq(raw: bytes) -> int | None:
    if raw[:4] != b"MThd" or len(raw) < 14:
        return None
    division = struct.unpack(">H", raw[12:14])[0]
    if division & 0x8000:
        return None  # SMPTE timecode division, not ticks-per-quarter
    return division


def deterministic_prediction(tpq: int, pos_res: int = 12) -> Counter:
    """Pre-clamp delta histogram if EVERY note sat exactly on the source
    file's native tick grid (delta_native=0), varying tick mod the full
    period needed for the tpq->480->pos_res mapping to repeat exactly.
    Weighted uniformly over one period (each native tick equally likely a
    priori) -- the null model for "no real performance jitter at all".
    """
    period = tpq  # one full source cell already covers the repeat period
    # true_pos_12 = round(k * 480/tpq) is midi_reader's rounding (k=native tick,
    # exact grid point, no jitter) -- but the SECOND resample (480->12, in
    # tokenizer.py) uses (onset_ticks_480 + delta_480/480) * (12/480), and
    # onset_ticks_480 here already includes midi_reader's own rounding. We
    # reproduce that full two-stage pipeline exactly, deterministically, for
    # k = 0..period-1.
    hist: Counter = Counter()
    for k in range(period):
        rel_onset_raw = k * TARGET_RES / tpq
        onset_480 = round(rel_onset_raw)
        onset_residual = rel_onset_raw - onset_480
        delta_480 = round(onset_residual * TARGET_RES)
        true_pos_12 = (onset_480 + delta_480 / TARGET_RES) * (pos_res / TARGET_RES)
        floor_12 = math.floor(true_pos_12)
        d = round((true_pos_12 - floor_12) * pos_res)
        hist[d] += 1
    total = sum(hist.values())
    return Counter({k: 100 * v / total for k, v in hist.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pos-res", type=int, default=12)
    args = parser.parse_args()

    from midigpt._types import Score

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_res = args.pos_res

    tpq_piece_counts: Counter = Counter()
    tpq_note_hist: defaultdict[int, Counter] = defaultdict(Counter)
    tpq_note_count: Counter = Counter()
    n_pieces = 0
    n_unreadable_tpq = 0

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
            tpq = read_midi_tpq(raw)
            if tpq is None:
                n_unreadable_tpq += 1
                continue
            try:
                score = Score.from_bytes(raw)
            except Exception:
                continue
            if not score.tracks:
                continue
            n_pieces += 1
            tpq_piece_counts[tpq] += 1

            native_res = score.resolution  # always 480 (MidiReader default)
            scale = pos_res / native_res
            for track in score.tracks:
                for bar in track.bars:
                    for note in bar.notes:
                        true_pos_12 = (note.onset_ticks + note.delta / native_res) * scale
                        floor_12 = math.floor(true_pos_12)
                        d = round((true_pos_12 - floor_12) * pos_res)
                        tpq_note_hist[tpq][d] += 1
                        tpq_note_count[tpq] += 1

    top_tpqs = [tpq for tpq, _ in tpq_piece_counts.most_common(10)]
    per_tpq_report = {}
    for tpq in top_tpqs:
        n = tpq_note_count[tpq]
        real_hist_pct = {str(k): round(100 * v / n, 2) for k, v in sorted(tpq_note_hist[tpq].items())} if n else {}
        pred_hist = deterministic_prediction(tpq, pos_res)
        pred_hist_pct = {str(k): round(v, 2) for k, v in sorted(pred_hist.items())}
        # L1 distance between real and predicted (null-model) histograms --
        # low distance means the real data looks like it has NO jitter beyond
        # what pure tpq->480->12 rounding arithmetic already predicts.
        all_bins = set(real_hist_pct) | set(pred_hist_pct)
        l1 = sum(abs(real_hist_pct.get(b, 0.0) - pred_hist_pct.get(b, 0.0)) for b in all_bins)
        gcd = math.gcd(tpq, TARGET_RES)
        per_tpq_report[str(tpq)] = {
            "n_pieces": tpq_piece_counts[tpq],
            "n_notes": n,
            "gcd_with_480": gcd,
            "exactly_divides_480": (TARGET_RES % tpq == 0) if tpq else None,
            "real_histogram_pct": real_hist_pct,
            "zero_jitter_null_model_pct": pred_hist_pct,
            "l1_distance_real_vs_null_model": round(l1, 2),
        }

    summary = {
        "n_pieces": n_pieces,
        "n_unreadable_tpq_header": n_unreadable_tpq,
        "n_distinct_source_tpqs": len(tpq_piece_counts),
        "source_tpq_piece_distribution": dict(tpq_piece_counts.most_common(20)),
        "per_tpq_report_top10": per_tpq_report,
    }

    with open(out_dir / "e0c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
