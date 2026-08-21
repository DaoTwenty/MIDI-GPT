"""Materialize the filtered humanize dataset as its own parquet files.

Reads the per-shard passing (row_i, track_idx, ...) results from
filter_expressive_tracks.py, and for every passing row: loads the original
score, PRUNES it down to only the qualifying track(s) (drops every track that
didn't pass nomml==12 + concentration + non-constant-residual), recomputes
num_tracks/total_notes to match, and writes the result to new parquet
files -- one for train, one for validation.

This keeps dataset.py/preprocess.py completely untouched: since bad tracks
are physically removed at the data-prep stage, the existing min_bars/
min_tracks eligibility check, select_window, and humanize bar-density
sampling will only ever see qualifying tracks.

Usage:
    python3 build_filtered_parquet.py \\
        --results-dir /scratch/triana24/MIDI-GPT/humanize_filter_results \\
        --data-dir /scratch/triana24/MIDI-GPT/data/v2.0.0 \\
        --output-dir /scratch/triana24/MIDI-GPT/data/humanize_filtered
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from midigpt._types import Score, Track
from midigpt._converters import to_cpp
import midigpt._core as _core

SCHEMA = pa.schema(
    [
        pa.field("music", pa.binary()),
        pa.field("NOMML", pa.list_(pa.int8())),
        pa.field("music_style_scraped", pa.string()),
        pa.field("num_tracks", pa.int32()),
        pa.field("total_notes", pa.int32()),
    ]
)


def prune_and_reserialize(music_bytes: bytes, nomml_list: list, keep_indices: list[int]):
    score = Score.from_bytes(music_bytes)
    kept_tracks = [score.tracks[i] for i in keep_indices if i < len(score.tracks)]
    if not kept_tracks:
        return None
    pruned = Score(tracks=kept_tracks, resolution=score.resolution, tempo=score.tempo)
    new_bytes = bytes(_core.MidiWriter().write_bytes(to_cpp(pruned)))

    pruned_nomml = [nomml_list[i] if nomml_list and i < len(nomml_list) else -1 for i in keep_indices]
    total_notes = sum(len(bar.notes) for t in kept_tracks for bar in t.bars)
    return new_bytes, pruned_nomml, len(kept_tracks), total_notes


def load_passing_by_shard(results_dir: Path) -> dict[str, dict[int, list[int]]]:
    """Returns {parquet_path: {row_i: [track_idx, ...]}}.

    filter_expressive_tracks.py writes one "scored" row per NOMML==12
    candidate track (row_i, track_idx, conc_score, n_notes, std_ratio,
    timing_mode_frac, velocity_mode_frac, pass_old, pass_new) -- keep only
    tracks where pass_new is true (the current filter: conc_score +
    timing_mode_frac + velocity_mode_frac).
    """
    by_shard: dict[str, dict[int, list[int]]] = {}
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name == "summary.json":
            continue
        data = json.loads(jf.read_text())
        path = data["path"]
        row_tracks: dict[int, list[int]] = defaultdict(list)
        for row_i, track_idx, conc, n_notes, std_ratio, tmf, vmf, p_old, p_new in data["scored"]:
            if p_new:
                row_tracks[row_i].append(track_idx)
        by_shard[path] = dict(row_tracks)
    return by_shard


def build_split(shard_row_tracks: dict[str, dict[int, list[int]]], out_path: Path) -> int:
    rows_out = []
    for path, row_tracks in shard_row_tracks.items():
        if not row_tracks:
            continue
        needed_rows = set(row_tracks.keys())
        pf = pq.ParquetFile(path)
        offset = 0
        found = 0
        for batch in pf.iter_batches(columns=["music", "NOMML", "music_style_scraped"], batch_size=2000):
            n = batch.num_rows
            batch_needed = [r - offset for r in needed_rows if offset <= r < offset + n]
            if batch_needed:
                music_col = batch.column("music")
                nomml_col = batch.column("NOMML").to_pylist()
                style_col = batch.column("music_style_scraped").to_pylist()
                for local_i in batch_needed:
                    global_i = local_i + offset
                    keep_indices = sorted(row_tracks[global_i])
                    music_bytes = bytes(music_col[local_i].as_py())
                    nomml_list = nomml_col[local_i]
                    style = style_col[local_i]
                    result = prune_and_reserialize(music_bytes, nomml_list, keep_indices)
                    if result is None:
                        continue
                    new_bytes, pruned_nomml, num_tracks, total_notes = result
                    rows_out.append({
                        "music": new_bytes,
                        "NOMML": pruned_nomml,
                        "music_style_scraped": style,
                        "num_tracks": num_tracks,
                        "total_notes": total_notes,
                    })
                    found += 1
            offset += n
        print(f"  [{Path(path).parent.name}/{Path(path).name}] wrote {found}/{len(needed_rows)} rows", flush=True)

    table = pa.Table.from_pylist(rows_out, schema=SCHEMA)
    pq.write_table(table, out_path)
    return len(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_shard = load_passing_by_shard(results_dir)
    train_shards = {p: rt for p, rt in by_shard.items() if "/train/" in p}
    val_shards = {p: rt for p, rt in by_shard.items() if "/validation/" in p}
    test_shards = {p: rt for p, rt in by_shard.items() if "/test/" in p}
    print(f"train shards with hits: {len(train_shards)}, validation shards with hits: {len(val_shards)}, "
          f"test shards with hits: {len(test_shards)}")

    n_train = n_val = n_test = 0
    if train_shards:
        print("\nBuilding train split...")
        n_train = build_split(train_shards, out_dir / "train.parquet")
        print(f"train.parquet: {n_train} rows")

    if val_shards:
        print("\nBuilding validation split...")
        n_val = build_split(val_shards, out_dir / "validation.parquet")
        print(f"validation.parquet: {n_val} rows")

    if test_shards:
        print("\nBuilding test split...")
        n_test = build_split(test_shards, out_dir / "test.parquet")
        print(f"test.parquet: {n_test} rows")

    print(f"\nTotal: {n_train + n_val + n_test} rows written to {out_dir}")


if __name__ == "__main__":
    main()
