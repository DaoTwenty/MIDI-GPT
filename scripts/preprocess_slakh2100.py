#!/usr/bin/env python3
"""Convert Slakh2100 tracks to GigaMIDI-compatible parquet format.

Reads Slakh2100 track directories (with MIDI/ and metadata.yaml),
merges per-stem MIDI into a single multi-track Score, serializes to bytes,
and writes parquet shards matching the GigaMIDI v2.0.0 schema.

Usage:
    python scripts/preprocess_slakh2100.py \
        --slakh-root /path/to/slakh2100/test \
        --output tmp/slakh2100.parquet \
        --max-tracks 16 \
        --min-notes 50
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ── GigaMIDI-compatible schema (subset of columns we need for training) ──────

_SCHEMA = pa.schema(
    [
        pa.field("music", pa.binary()),
        pa.field("NOMML", pa.list_(pa.int32())),
        pa.field("num_tracks", pa.int32()),
        pa.field("TPQN", pa.int32()),
        pa.field("total_notes", pa.int32()),
        pa.field("avg_note_duration", pa.float64()),
        pa.field("avg_velocity", pa.float64()),
        pa.field("min_velocity", pa.int32()),
        pa.field("max_velocity", pa.int32()),
        pa.field("tempo", pa.string()),
        pa.field("loop_track_idx", pa.list_(pa.int32())),
        pa.field("loop_instrument_type", pa.list_(pa.string())),
        pa.field("loop_start", pa.list_(pa.int32())),
        pa.field("loop_end", pa.list_(pa.int32())),
        pa.field("loop_duration_beats", pa.list_(pa.float64())),
        pa.field("loop_note_density", pa.list_(pa.float64())),
        pa.field("NOMML_Instrument", pa.list_(pa.int32())),
        pa.field("Type", pa.string()),
        pa.field("instrument_category: drums-only: 0, all-instruments-with-drums: 1,no-drums: 2", pa.int32()),
        pa.field("md5", pa.string()),
        pa.field("music_styles_curated", pa.list_(pa.string())),
        pa.field("music_style_scraped", pa.string()),
    ]
)


def _midi_bytes(score_bytes: bytes) -> bytes:
    """Re-serialize a Score to MIDI bytes via symusic."""
    from symusic import Score

    score = Score.from_midi(score_bytes)
    return score.to_midi()


def _merge_stems(track_dir: Path, max_tracks: int = 16) -> bytes | None:
    """Load the merged all_src.mid from a Slakh2100 track directory.

    Returns MIDI bytes or None on failure.
    """
    from symusic import Score

    all_src = track_dir / "all_src.mid"
    if not all_src.exists():
        return None

    try:
        raw = all_src.read_bytes()
        score = Score.from_midi(raw)
        return score.dumps_midi()
    except Exception:
        return None


def _compute_stats(midi_bytes: bytes) -> dict[str, Any]:
    """Compute GigaMIDI-compatible metadata stats from MIDI bytes."""
    from symusic import Score

    try:
        score = Score.from_midi(midi_bytes)
    except Exception:
        return {}

    num_tracks = len(score.tracks)
    all_notes = []
    velocities = []
    durations = []

    for track in score.tracks:
        for note in track.notes:
            all_notes.append(note)
            velocities.append(note.velocity)
            durations.append(note.duration)

    total_notes = len(all_notes)
    if total_notes == 0:
        return {}

    avg_dur = sum(durations) / len(durations) if durations else 0
    avg_vel = sum(velocities) / len(velocities) if velocities else 0
    min_vel = min(velocities) if velocities else 0
    max_vel = max(velocities) if velocities else 0

    # Determine instrument category
    has_drum = any(getattr(t, "track_type", "") == "drum" for t in score.tracks)
    has_melodic = any(getattr(t, "track_type", "") != "drum" for t in score.tracks)
    if has_drum and has_melodic:
        cat = 1  # all-instruments-with-drums
    elif has_drum:
        cat = 0  # drums-only
    else:
        cat = 2  # no-drums

    return {
        "num_tracks": num_tracks,
        "total_notes": total_notes,
        "TPQN": score.tpq,
        "avg_note_duration": avg_dur,
        "avg_velocity": avg_vel,
        "min_velocity": min_vel,
        "max_velocity": max_vel,
        "tempo": str(round(60_000_000 / max(1, score.tempos[0].qpm if score.tempos else 500000), 1)),
        "instrument_category": cat,
    }


def process_track(track_dir: Path, max_tracks: int = 16) -> dict[str, Any] | None:
    """Process a single Slakh2100 track directory into a GigaMIDI-compatible row."""
    midi_bytes = _merge_stems(track_dir, max_tracks=max_tracks)
    if midi_bytes is None:
        return None

    # Compute MD5
    md5 = hashlib.md5(midi_bytes).hexdigest()

    stats = _compute_stats(midi_bytes)
    if not stats:
        return None

    # Read metadata for stem info
    meta_path = track_dir / "metadata.yaml"
    stem_info = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        for stem_id, stem_data in meta.get("stems", {}).items():
            if stem_data.get("midi_saved", False):
                stem_info.append({
                    "name": stem_id,
                    "inst_class": stem_data.get("inst_class", ""),
                    "is_drum": stem_data.get("is_drum", False),
                    "program_num": stem_data.get("program_num", 0),
                })

    return {
        "music": midi_bytes,
        "md5": md5,
        "NOMML": [0] * stats["num_tracks"],  # Slakh doesn't have NOMML annotations
        "NOMML_Instrument": [0] * stats["num_tracks"],
        "num_tracks": stats["num_tracks"],
        "TPQN": stats["TPQN"],
        "total_notes": stats["total_notes"],
        "avg_note_duration": stats["avg_note_duration"],
        "avg_velocity": stats["avg_velocity"],
        "min_velocity": stats["min_velocity"],
        "max_velocity": stats["max_velocity"],
        "tempo": stats["tempo"],
        "loop_track_idx": [],
        "loop_instrument_type": [],
        "loop_start": [],
        "loop_end": [],
        "loop_duration_beats": [],
        "loop_note_density": [],
        "Type": "0",
        "instrument_category": stats["instrument_category"],
        "music_styles_curated": [],
        "music_style_scraped": "slakh2100",
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Slakh2100 to GigaMIDI parquet format")
    parser.add_argument("--slakh-root", required=True, help="Path to Slakh2100 split directory (e.g. slakh2100/test)")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--max-tracks", type=int, default=16, help="Max stems per track")
    parser.add_argument("--min-notes", type=int, default=50, help="Minimum total notes to include")
    args = parser.parse_args()

    slakh_root = Path(args.slakh_root)
    if not slakh_root.exists():
        print(f"Error: {slakh_root} does not exist")
        sys.exit(1)

    track_dirs = sorted([d for d in slakh_root.iterdir() if d.is_dir() and d.name.startswith("Track")])
    print(f"Found {len(track_dirs)} track directories in {slakh_root}")

    rows = []
    skipped = 0
    for i, track_dir in enumerate(track_dirs):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(track_dirs)}...")
        row = process_track(track_dir, max_tracks=args.max_tracks)
        if row is None:
            skipped += 1
            continue
        if row["total_notes"] < args.min_notes:
            skipped += 1
            continue
        rows.append(row)

    print(f"Processed {len(rows)} tracks, skipped {skipped}")

    if not rows:
        print("Error: No valid tracks found")
        sys.exit(1)

    # Write parquet
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
