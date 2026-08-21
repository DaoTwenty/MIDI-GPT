"""E0b -- characterize the second pre-clamp Delta mode (no model, CPU-only).

E0 (e0_reference_stats.py) found the pre-clamp Delta histogram is bimodal: a
mode near 0 and a second mode around bin ~10 (13.9% of notes, full validation
set), not explained by source-TPQ diversity (every piece in this corpus is
natively 480 TPQ -- itself a data point: a corpus of genuinely varied-source
human performances would not usually share one exact native resolution).

This script asks: is that second mode genuine expressive timing (e.g. swing,
where the "and" of the beat is systematically pushed late) or a mechanical /
resampling artifact (e.g. a fixed near-constant offset applied uniformly,
independent of rhythmic position, with little or no per-note jitter)?

Computed directly from raw (un-resampled) notes rather than through
Tokenizer.normalize_input / resample_delta, so this is independent of -- and
unaffected by -- any fix to resample_delta's own rounding scheme: the true
continuous onset position at pos-resolution (encoder resolution, 12) is
``(onset_ticks + delta/native_res) * (12/native_res)``, and its residual
above ``floor()`` (scaled back to 0..12, matching E0's original pre-clamp
convention) is scheme-independent ground truth.

Per piece, among notes whose (floor-residual) delta falls in the "second
mode" band (configurable, default bins 8-12):
  1. Population stdev of the raw pre-clamp delta values in that band. Low
     stdev (few distinct values, tightly clustered) is the signature of a
     deterministic transform (e.g. a mis-assumed resample ratio, or a
     zero-humanization swing-quantize tool); real performance jitter should
     spread across many values.
  2. Mean delta broken out by eighth-note phase (onset_ticks_at_res12 % 12,
     bar-relative -- 0 = downbeat 8th, 6 = "and"/upbeat 8th). Classic swing
     shows a strong phase-dependent split (upbeats pushed late, downbeats
     not); a source-agnostic artifact should not care about phase.
  3. What fraction of ALL notes in the piece fall in the second-mode band
     (pieces where it's a small minority are less interesting than pieces
     where it dominates).

Aggregates across "mode-2-heavy" pieces (>=50% of notes in the band):
  count, genre distribution (music_style_scraped), and the distributions of
  (1) and (2) above.

Usage:
    python3 e0b_delta_mode_investigation.py \\
        --parquet $SCRATCH/MIDI-GPT/data/humanize_filtered/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e0b \\
        [--limit N] [--mode2-lo 8] [--mode2-hi 12] [--mode2-piece-frac 0.5]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode2-lo", type=int, default=8)
    parser.add_argument("--mode2-hi", type=int, default=12)
    parser.add_argument("--mode2-piece-frac", type=float, default=0.5)
    args = parser.parse_args()

    import math

    import midigpt._core as _core
    from midigpt._types import Score

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    pos_res = enc_cfg.resolution  # 12

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_piece_records: list[dict] = []
    n_pieces = 0
    native_tpqs: Counter = Counter()

    columns = ["music", "music_style_scraped"]
    pf = pq.ParquetFile(args.parquet)
    row_i = 0
    done = False
    for batch in pf.iter_batches(columns=columns, batch_size=200):
        if done:
            break
        musics = batch.column("music")
        genres = batch.column("music_style_scraped")
        for music, genre in zip(musics, genres):
            row_i += 1
            if args.limit is not None and row_i > args.limit:
                done = True
                break
            raw = bytes(music.as_py())
            genre_str = genre.as_py() or "unknown"
            try:
                score = Score.from_bytes(raw)
            except Exception:
                continue
            if not score.tracks:
                continue
            n_pieces += 1
            native_tpqs[score.resolution] += 1

            native_res = score.resolution
            scale = pos_res / native_res
            band_deltas: list[int] = []
            band_phase_delta: defaultdict[int, list[int]] = defaultdict(list)
            n_notes = 0
            for track in score.tracks:
                for bar in track.bars:
                    for note in bar.notes:
                        n_notes += 1
                        true_pos_12 = (note.onset_ticks + note.delta / native_res) * scale
                        pos12 = math.floor(true_pos_12)
                        d = round((true_pos_12 - pos12) * pos_res)  # scheme-independent floor-residual
                        if args.mode2_lo <= d <= args.mode2_hi:
                            band_deltas.append(d)
                            phase = pos12 % pos_res
                            band_phase_delta[phase].append(d)

            if n_notes == 0:
                continue
            band_frac = len(band_deltas) / n_notes
            record = {
                "genre": genre_str,
                "n_notes": n_notes,
                "band_frac": band_frac,
                "band_n": len(band_deltas),
                "band_stdev": statistics.pstdev(band_deltas) if len(band_deltas) >= 2 else None,
                "band_mean": statistics.mean(band_deltas) if band_deltas else None,
                "downbeat_8th_mean_delta": (
                    statistics.mean(band_phase_delta[0]) if band_phase_delta.get(0) else None
                ),
                "upbeat_8th_mean_delta": (
                    statistics.mean(band_phase_delta[6]) if band_phase_delta.get(6) else None
                ),
                "downbeat_8th_n": len(band_phase_delta.get(0, [])),
                "upbeat_8th_n": len(band_phase_delta.get(6, [])),
            }
            per_piece_records.append(record)

    mode2_heavy = [r for r in per_piece_records if r["band_frac"] >= args.mode2_piece_frac]

    def safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    def safe_median(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    genre_counts = Counter(r["genre"] for r in mode2_heavy)
    all_genre_counts = Counter(r["genre"] for r in per_piece_records)

    swing_gap = [
        (r["upbeat_8th_mean_delta"] - r["downbeat_8th_mean_delta"])
        for r in mode2_heavy
        if r["upbeat_8th_mean_delta"] is not None and r["downbeat_8th_mean_delta"] is not None
    ]

    summary = {
        "n_pieces": n_pieces,
        "native_tpq_distribution": dict(native_tpqs),
        "mode2_band": [args.mode2_lo, args.mode2_hi],
        "n_pieces_mode2_heavy": len(mode2_heavy),
        "mode2_heavy_frac_of_corpus": len(mode2_heavy) / len(per_piece_records) if per_piece_records else None,
        "mode2_heavy_band_stdev": {
            "mean": safe_mean([r["band_stdev"] for r in mode2_heavy]),
            "median": safe_median([r["band_stdev"] for r in mode2_heavy]),
        },
        "mode2_heavy_swing_gap_upbeat_minus_downbeat_mean_delta": {
            "mean": safe_mean(swing_gap),
            "median": safe_median(swing_gap),
            "n": len(swing_gap),
        },
        "mode2_heavy_top_genres": genre_counts.most_common(15),
        "all_pieces_top_genres": all_genre_counts.most_common(15),
        "mode2_heavy_genre_share_vs_corpus_share": {
            g: {
                "mode2_heavy_count": c,
                "mode2_heavy_pct": round(100 * c / len(mode2_heavy), 2) if mode2_heavy else None,
                "corpus_pct": round(100 * all_genre_counts[g] / len(per_piece_records), 2) if per_piece_records else None,
            }
            for g, c in genre_counts.most_common(15)
        },
    }

    with open(out_dir / "e0b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "e0b_per_piece.json", "w") as f:
        json.dump(per_piece_records, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
