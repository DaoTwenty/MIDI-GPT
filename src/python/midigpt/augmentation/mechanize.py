"""Shared "make a track mechanical" transform: flat velocity + onset snapped
to the nearest of the model-resolution grid cells per bar.

Used both by training (dataset.py, to synthesize mechanical-context tracks)
and by eval (scripts/humanize_eval/), so the two can never drift apart the
way e1/e4's ad hoc flatten_context() did (it zeroed note.delta, which is a
no-op for the encoded tokens -- Tokenizer.encode() -> normalize_input()
always recomputes delta from the note's real onset_ticks via resample_delta,
regardless of what delta was set to beforehand; only onset_ticks itself
actually removes microtiming from what gets encoded).
"""

from __future__ import annotations

MECHANICAL_VELOCITY = 80
GRID_CELLS_PER_BAR = 12


def quantized_onset(note, bar, resolution: int, n_cells: int = GRID_CELLS_PER_BAR) -> int:
    """Nearest of the n_cells model-resolution grid cells per bar, in the
    score's current tick units."""
    bar_len = bar.beat_length * resolution
    if bar_len <= 0:
        return note.onset_ticks
    cell_width = bar_len / n_cells
    cell = round(note.onset_ticks / cell_width)
    return round(cell * cell_width)


def mechanize_bar(bar, resolution: int, const_velocity: int = MECHANICAL_VELOCITY) -> None:
    """In-place: flatten every note in `bar` to constant velocity and
    grid-quantized onset (no microtiming)."""
    for note in bar.notes:
        note.velocity = const_velocity
        note.onset_ticks = quantized_onset(note, bar, resolution)
        note.delta = 0


def mechanize_track(track, resolution: int, const_velocity: int = MECHANICAL_VELOCITY) -> None:
    """In-place: mechanize every bar of `track`."""
    for bar in track.bars:
        mechanize_bar(bar, resolution, const_velocity)
