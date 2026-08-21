"""Fast, no-checkpoint-needed unit tests for humanize_style_server.py and
parquet_retrieval.py. Real-checkpoint smoke tests (plain AR/humanize,
per-bar mechanize, multi-window AR, steering, soft-prefix, parquet
retrieval) were run manually this session -- see RESULTS_LOG.md /
conversation history for those results; not re-encoded as automated tests
here since they need real checkpoint files and take real wall-clock time
(matching this repo's existing pytest.mark.slow precedent for other
real-checkpoint tests).

Run: cd scripts/style_prototype && python3 -m pytest tests/ -v
(not under src/python/ -- sys.path-insert style like the eval scripts,
consistent with the rest of this directory.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from humanize_style_server import FlexServer, _iter_windows
from parquet_retrieval import ParquetIndex, resolve_under_root


# ---- _iter_windows / window-target bar.notes gating ----


class _FakeBar:
    def __init__(self, notes):
        self.notes = notes


class _FakeTrack:
    def __init__(self, bars):
        self.bars = bars


class _FakeScore:
    def __init__(self, bars_per_track):
        self.tracks = [_FakeTrack(bars) for bars in bars_per_track]


def test_iter_windows_single_window():
    windows = list(_iter_windows(4, 16))
    assert windows == [(0, 0, 4)]


def test_iter_windows_multi_window_right_suffix():
    # 40 bars, 16-bar window -> every yielded new_end must equal that
    # window's own last local bar (window_bars), i.e. new_end - window_start == 16,
    # except possibly the very first if it doesn't need to shift.
    windows = list(_iter_windows(40, 16))
    for start, new_start, new_end in windows:
        assert new_end - start == 16 or new_end == 40 - start + start  # local end is window's own edge
        assert new_end <= start + 16
    # covers the whole piece with no gaps
    covered = sorted(ns for _, ns, _ in windows)
    assert covered[0] == 0
    total = 0
    for _, ns, ne in windows:
        assert ns == total
        total = ne
    assert total == 40


def test_window_target_humanize_gates_empty_bars():
    server = FlexServer(device="cpu", cache_size=1, data_root=Path("/tmp"))
    score = _FakeScore([[_FakeBar([1]), _FakeBar([]), _FakeBar([1])]])
    targets_local = {0: ("humanize", None, {}, {}, {})}
    target_cells, extra = server._resolve_window_targets(score, targets_local, (0, 3))
    assert target_cells == {0: [0, 2]}  # bar 1 (empty) excluded


def test_window_target_ar_includes_empty_bars():
    server = FlexServer(device="cpu", cache_size=1, data_root=Path("/tmp"))
    score = _FakeScore([[_FakeBar([1]), _FakeBar([]), _FakeBar([1])]])
    targets_local = {0: ("autoregressive", None, {}, {}, {})}
    target_cells, extra = server._resolve_window_targets(score, targets_local, (0, 3))
    assert target_cells == {0: [0, 1, 2]}  # empty bar 1 included -- the whole point of AR


def test_window_target_respects_wanted_subset():
    server = FlexServer(device="cpu", cache_size=1, data_root=Path("/tmp"))
    score = _FakeScore([[_FakeBar([1]), _FakeBar([1]), _FakeBar([1])]])
    targets_local = {0: ("humanize", {0, 2}, {}, {}, {})}
    target_cells, extra = server._resolve_window_targets(score, targets_local, (0, 3))
    assert target_cells == {0: [0, 2]}


# ---- mechanize-set resolution precedence ----


def test_mechanize_set_precedence():
    from humanize_style_server import _TargetSpec

    score = _FakeScore([[_FakeBar([]), _FakeBar([]), _FakeBar([])]])
    spec = _TargetSpec(track=0, mode="context", mechanize_before=True, expressive_bars=[1])
    out = FlexServer._resolve_mechanize_set(score, [spec])
    # mechanize_before=True mechanizes every bar except expressive_bars
    assert out == {(0, 0): True, (0, 2): True}
    assert (0, 1) not in out


def test_mechanize_set_explicit_bars_without_mechanize_before():
    from humanize_style_server import _TargetSpec

    score = _FakeScore([[_FakeBar([]), _FakeBar([])]])
    spec = _TargetSpec(track=0, mode="context", mechanize_bars=[1])
    out = FlexServer._resolve_mechanize_set(score, [spec])
    assert out == {(0, 1): True}


def test_mechanize_bars_expressive_bars_overlap_rejected():
    from humanize_style_server import _TargetSpec

    with pytest.raises(Exception):
        _TargetSpec(track=0, mode="context", mechanize_bars=[0], expressive_bars=[0])


def test_target_spec_context_mode_forbids_bars():
    from humanize_style_server import _TargetSpec

    with pytest.raises(Exception):
        _TargetSpec(track=0, mode="context", bars=[0])


def test_target_spec_non_context_requires_bars():
    from humanize_style_server import _TargetSpec

    with pytest.raises(Exception):
        _TargetSpec(track=0, mode="humanize")


# ---- ParquetIndex random access ----


def test_parquet_index_row_group_math(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "fixture.parquet"
    # Build a table with multiple row groups so the bisect logic is actually exercised.
    table = pa.table({"music": [f"row{i}".encode() for i in range(25)]})
    pq.write_table(table, path, row_group_size=7)  # -> 4 row groups (7,7,7,4)

    idx = ParquetIndex(path)
    assert idx.num_rows == 25
    for i in [0, 6, 7, 13, 14, 24]:  # spans every row-group boundary
        row = idx.read_row(i)
        assert row["music"] == f"row{i}".encode()

    with pytest.raises(IndexError):
        idx.read_row(25)
    with pytest.raises(IndexError):
        idx.read_row(-1)


def test_resolve_under_root_rejects_traversal(tmp_path):
    (tmp_path / "sub").mkdir()
    real = tmp_path / "sub" / "data.parquet"
    real.write_bytes(b"")

    assert resolve_under_root(tmp_path, "sub/data.parquet") == real
    with pytest.raises(ValueError):
        resolve_under_root(tmp_path, "../outside.parquet")
    with pytest.raises(ValueError):
        resolve_under_root(tmp_path, "sub/data.parquet".replace("parquet", "txt"))
    with pytest.raises(ValueError):
        resolve_under_root(tmp_path, "/etc/passwd")
