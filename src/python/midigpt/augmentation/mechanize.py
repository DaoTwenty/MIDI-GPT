"""Shared "make a track mechanical" transform: flat velocity + onset snapped
to the nearest of the model-resolution grid cells per BEAT.

Used both by training (dataset.py, to synthesize mechanical-context tracks)
and by eval (scripts/humanize_eval/), so the two can never drift apart the
way e1/e4's ad hoc flatten_context() did (it zeroed note.delta, which is a
no-op for the encoded tokens -- Tokenizer.encode() -> normalize_input()
always recomputes delta from the note's real onset_ticks via resample_delta,
regardless of what delta was set to beforehand; only onset_ticks itself
actually removes microtiming from what gets encoded).

GRID_CELLS_PER_BEAT=12 is deliberately the SAME resolution
tokenizer.resample_delta uses for the model's own coarse/fine timing split
(target_res=12, "12 ticks per quarter note") -- mechanization is meant to
remove exactly the microtiming Delta is capable of restoring, no more, so
that Delta's +/-6-unit (+/-half-cell) range can always fully recover a
mechanized note back to its original position. Previously this quantized to
12 cells spread across the WHOLE BAR regardless of how many beats it has
(GRID_CELLS_PER_BAR, cell_width = bar_len/12) -- for a 4-beat bar that's
only 3 cells/beat, a grid 4x coarser than Delta's own 12-cells/beat
resolution, so mechanization was destroying up to ~4x more positional
information than Delta could ever restore. Fixed 2026-08-21 (see
RESULTS_LOG.md's "microtiming distribution discrepancy" entry) -- cell
width is now a fixed fraction of one beat, independent of bar length/time
signature, matching resample_delta exactly.
"""

from __future__ import annotations

MECHANICAL_VELOCITY = 80
GRID_CELLS_PER_BEAT = 12


def quantized_onset(note, bar, resolution: int, n_cells_per_beat: int = GRID_CELLS_PER_BEAT) -> int:
    """Nearest of the n_cells_per_beat model-resolution grid cells per BEAT
    (not per bar -- see module docstring), in the score's current tick
    units. `resolution` is ticks-per-beat, so cell_width is independent of
    bar.beat_length/time signature by construction."""
    if resolution <= 0:
        return note.onset_ticks
    cell_width = resolution / n_cells_per_beat
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
