"""Row-index random access into a parquet dataset (validation/test/other),
for humanize_style_server.py's /parquet/* endpoints -- lets a caller fetch a
real MIDI piece by index instead of needing to upload their own. No existing
code in this repo does random-access-by-index (every eval script only ever
sequentially scans via `iter_batches`) -- this reads just the target row's
row group via `read_row_group`, not the whole file.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import pyarrow.parquet as pq

# Columns confirmed present in this project's parquet datasets (dataset.py,
# e6_per_instrument.py, e10_listening_examples.py) -- `music` is required,
# the rest are best-effort metadata (None if the file lacks them).
KNOWN_COLUMNS = ("music", "music_style_scraped", "num_tracks", "total_notes", "NOMML")


class ParquetIndex:
    """Opened once per resolved path, cached by the server (small LRU, same
    pattern as EngineCache)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._pf = pq.ParquetFile(str(path))
        md = self._pf.metadata
        self.num_rows = md.num_rows
        self.columns = [c for c in KNOWN_COLUMNS if c in self._pf.schema_arrow.names]
        starts = [0]
        for g in range(md.num_row_groups):
            starts.append(starts[-1] + md.row_group(g).num_rows)
        self._row_group_starts = starts  # len == num_row_groups + 1

    def read_row(self, index: int) -> dict:
        if not (0 <= index < self.num_rows):
            raise IndexError(f"row index {index} out of range [0, {self.num_rows})")
        rg = bisect_right(self._row_group_starts, index) - 1
        table = self._pf.read_row_group(rg, columns=self.columns)
        local = index - self._row_group_starts[rg]
        return {c: table.column(c)[local].as_py() for c in self.columns}


class ParquetIndexCache:
    """Small path-keyed LRU wrapping ParquetIndex, mirroring EngineCache's
    eviction pattern -- avoids re-opening/re-reading metadata for repeated
    requests against the same file."""

    def __init__(self, max_entries: int = 8) -> None:
        from collections import OrderedDict

        self._entries: OrderedDict[str, ParquetIndex] = OrderedDict()
        self._max_entries = max_entries

    def get(self, path: Path) -> ParquetIndex:
        key = str(path)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        idx = ParquetIndex(path)
        self._entries[key] = idx
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return idx


def resolve_under_root(data_root: Path, relative_path: str) -> Path:
    """Resolves `relative_path` under `data_root`, rejecting anything that
    escapes it (path traversal) or isn't a real .parquet file. Raises
    ValueError with a caller-facing message on any failure -- the server
    maps this to a 400."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("path must be a non-empty path relative to --data-root")
    resolved = (data_root / relative_path).resolve()
    root = data_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path {relative_path!r} escapes --data-root")
    if resolved.suffix != ".parquet":
        raise ValueError(f"path {relative_path!r} is not a .parquet file")
    if not resolved.is_file():
        raise ValueError(f"path {relative_path!r} not found")
    return resolved
