"""Applying manual note edits to a single bar.

Pure and storage-agnostic — no session/lock/broadcast concerns here, just
"given this bar's current dict and a list of ops, produce the new bar dict
or an error." Field names match midigpt._types.Note/Bar exactly.
"""

from __future__ import annotations

_NOTE_FIELDS = ("pitch", "velocity", "onset_ticks", "duration_ticks", "delta")


def _validate_note(note: object) -> tuple[dict | None, str | None]:
    if not isinstance(note, dict):
        return None, "note must be an object"
    for field_name in ("pitch", "velocity", "onset_ticks", "duration_ticks"):
        if not isinstance(note.get(field_name), int):
            return None, f"note.{field_name} must be an int"
    if not (0 <= note["pitch"] <= 127):
        return None, f"note.pitch {note['pitch']} out of MIDI range [0, 127]"
    if not (0 <= note["velocity"] <= 127):
        return None, f"note.velocity {note['velocity']} out of MIDI range [0, 127]"
    if note["onset_ticks"] < 0:
        return None, f"note.onset_ticks {note['onset_ticks']} must be >= 0"
    if note["duration_ticks"] <= 0:
        return None, f"note.duration_ticks {note['duration_ticks']} must be > 0"
    delta = note.get("delta", 0)
    if not isinstance(delta, int):
        return None, "note.delta must be an int"
    return {
        "pitch": note["pitch"],
        "velocity": note["velocity"],
        "onset_ticks": note["onset_ticks"],
        "duration_ticks": note["duration_ticks"],
        "delta": delta,
    }, None


def apply_edit_ops(bar: dict, ops: list) -> tuple[dict | None, str | None]:
    """Apply a list of {"op": "add"|"delete"|"move", ...} ops to a copy of
    `bar`'s note list, all-or-nothing: the first invalid op aborts the
    whole batch and returns (None, error) with `bar` left untouched.
    Notes are addressed by `note_index` (their position in bar["notes"]),
    since Note has no persistent id.
    """
    if not isinstance(ops, list) or not ops:
        return None, "edit requires a non-empty 'ops' list"
    notes = list(bar.get("notes", []))
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return None, f"ops[{i}] must be an object"
        kind = op.get("op")
        if kind == "add":
            note, err = _validate_note(op.get("note"))
            if err is not None:
                return None, f"ops[{i}] (add): {err}"
            notes.append(note)
        elif kind == "delete":
            idx = op.get("note_index")
            if not isinstance(idx, int) or not (0 <= idx < len(notes)):
                return None, f"ops[{i}] (delete): note_index {idx!r} out of range"
            notes.pop(idx)
        elif kind == "move":
            idx = op.get("note_index")
            if not isinstance(idx, int) or not (0 <= idx < len(notes)):
                return None, f"ops[{i}] (move): note_index {idx!r} out of range"
            note, err = _validate_note(op.get("note"))
            if err is not None:
                return None, f"ops[{i}] (move): {err}"
            notes[idx] = note
        else:
            return None, f"ops[{i}]: unknown op {kind!r} (expected add/delete/move)"
    return {**bar, "notes": notes}, None
