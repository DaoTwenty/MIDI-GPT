"""Extracting a patch from a generation result, and applying a patch onto
a session's canonical score.
"""

from __future__ import annotations


def extract_patch(result: dict, track: int, bar_lo: int, bar_hi: int) -> dict:
    """Pull bars [bar_lo, bar_hi] of `track` out of a generation `result`
    (midigpt-http's full returned score) and return them as a patch:
    {"track": track, "bars": {bar_index: bar_dict, ...}}.

    Only the targeted range is extracted — the result score also contains
    every OTHER track/bar as context, and treating those as real changes
    would silently clobber any concurrent generation/edit that touched a
    different range while this one was in flight. Pure and read-only: does
    NOT mutate `result`. Pair with apply_patch to actually write it onto a
    canonical score.
    """
    src_bars = result["tracks"][track]["bars"]
    patched: dict[int, dict] = {}
    for b in range(bar_lo, bar_hi + 1):
        if b < len(src_bars):
            patched[b] = src_bars[b]
    return {"track": track, "bars": patched}


def apply_patch(canonical: dict, patch: dict) -> None:
    """Write a {"track", "bars": {bar_index: bar_dict}} patch (as produced
    by extract_patch, or built directly for an `edit`) onto `canonical` IN
    PLACE. Relies on the caller having held the matching (track, bar-range)
    lock — this does no reconciliation/diffing itself, same as before.
    """
    dst_bars = canonical["tracks"][patch["track"]]["bars"]
    for b, bar_dict in patch["bars"].items():
        if b < len(dst_bars):
            dst_bars[b] = bar_dict
