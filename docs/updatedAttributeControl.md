# Per-Bar Attribute Control (Updated)

This document specifies the design for **per-bar attribute and control
overrides** in the MIDI-GPT inference API, and the corresponding
implementation plan.

It supersedes the previous track-only behavior: today, every entry in
`TrackPrompt.attributes` is applied at the track level (one value
shared across all bars of the track). This design adds first-class
per-bar overrides for both attributes (`bar_attributes`) and
non-attribute controls (`bar_controls`), with regime-aware semantics.

---

## 1. API additions

```python
@dataclass
class TrackPrompt:
    id: int
    bars: list[int]
    autoregressive: bool = False
    ignore:         bool = False
    mask_bars:      list[int] = field(default_factory=list)

    # Track-level attribute overrides (one value per track).
    attributes:     dict[str, int] = field(default_factory=dict)
    # Track-level non-attribute controls (e.g. "time_signature").
    controls:       dict[str, Any] = field(default_factory=dict)

    # NEW: per-bar attribute overrides. Keyed by ABSOLUTE bar index.
    #   { bar_idx: { "tension": 3, "bar_pitch_class_set": 12, ... } }
    bar_attributes: dict[int, dict[str, int]] = field(default_factory=dict)
    # NEW: per-bar non-attribute controls (e.g. "time_signature").
    bar_controls:   dict[int, dict[str, Any]] = field(default_factory=dict)
```

Key rules:

- Both new dicts are keyed by **absolute bar index** (same coordinate
  space as `tp.bars` and the underlying score).
- A key set in `bar_attributes` must reference an attribute whose
  `level == "bar"` (e.g. `bar_pitch_class_set`, `tension`). Track-level
  attributes (e.g. `min_polyphony`) belong in `tp.attributes`.
- A track-level override and a per-bar override may coexist; the
  per-bar value wins at the matching bar.

---

## 2. Semantics by regime

Three regimes determine how overrides are realized:

- **Full AR** — `tp.autoregressive=True` and no prefix bars (the whole
  track is generated). No prompt tokens for the generation region;
  attributes are realized via C++ constraints on the sampler.
- **Partial AR** — `tp.autoregressive=True` with prefix bars (a
  right-suffix is generated). The prefix is encoded as part of the
  prompt; the suffix is generated.
- **Infill** — `tp.autoregressive=False` with generation targets. The
  prompt encodes all bars; generation fills in `FillInPlaceholder`
  spans only.

| Regime      | Track-level override (`tp.attributes`)                                                                                   | Bar-level override (`tp.bar_attributes`)                                                                                                                            |
|-------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Full AR** | `AttributeValueConstraint` on the TrackStart attribute slot. Any value in `[0, size)` is allowed.                        | `BarAttributeValueConstraint` (NEW C++ constraint) on the matching `(track_idx, bar_idx)` slot. `bar_idx` must be in `tp.bars`.                                     |
| **Partial AR** | Override the prompt token at the prefix's TrackStart slot. **No suffix constraint** (TrackStart is in the prefix only). Allowed iff *achievable* given the fixed prefix (warning if not). | `bar_idx` must be in the **suffix** (i.e. in `tp.bars`). Override the prompt token at the bar's attribute slot, and add a `BarAttributeValueConstraint` for that bar in the suffix. Prefix bar overrides are rejected (structural error). |
| **Infill**  | Override the prompt token at the TrackStart slot. Allowed iff *achievable* given the fixed (non-infill) bars (warning if not). | `bar_idx` must be in `tp.bars`. Override the prompt token at the bar's attribute slot. No constraint needed (the prompt fixes the attribute for that bar). Context-bar overrides are rejected. |

The same table applies to `controls` / `bar_controls`, with the
caveat that controls do not pass through the analyzer (no
achievability check) — their value is taken at face value.

### 2.1 Time-signature cross-track coherence

Because `TimeSig` describes a global property of a musical bar, the
same bar index must resolve to the same `(num, den)` across **all
generating tracks** in a step. We enforce:

- For any `bar_idx` referenced in `tp.bar_controls["time_signature"]`
  on more than one track, all values must match.
- If the same bar is already present as a context bar (e.g. on an
  ignored or non-generating track) with a fixed `ts_numerator/denominator`,
  the override must match that as well.

Violations are **errors** (structural).

---

## 3. Achievability

An override on a *track-level* attribute in **Partial AR** or **Infill**
may be physically infeasible: the realized attribute is computed over
the full track (fixed bars + generated bars), so the fixed bars may
already preclude the requested value.

For each attribute, `BaseAttribute.achievable_range(fixed_score,
track_idx, generated_bars) -> (min_q, max_q)` returns the closed
interval of quantized values that *could* still be realized.

Per-attribute monotonicity:

- `max_*` (e.g. `max_polyphony`, `max_note_duration`):
  `realized = max(F, G)` where `F` is the fixed-bar value and `G` the
  generated-bar value. Override `v` is achievable iff `v ≥ F_q`.
  → `achievable_range = (F_q, size - 1)`.
- `min_*` (e.g. `min_polyphony`, `min_note_duration`):
  `realized = min(F, G)`. Override `v` is achievable iff `v ≤ F_q`.
  → `achievable_range = (0, F_q)`.
- `note_density` (averaged over bars with notes): the realized value
  is bounded by `[q(prefix_avg), q(prefix + max_per_bar · |G|)]`.
- Generic / unknown attributes: full domain `(0, size - 1)` (no
  warning ever fires).

### 3.1 Achievability is a warning, not an error

Achievability is a *post-hoc* property of the realized attribute, not
a structural property of the request. The model will still generate
valid music; the analyzer's evaluation just won't match the requested
value. We:

- **Warn** (via `logging.warning(...)`) at request-validation time
  when a track-level override in partial-AR or infill falls outside
  `achievable_range`. The request is still accepted.
- **Error** (raise `RequestValidationError`) only on *structural*
  problems: unknown attribute name, out-of-domain value, level
  mismatch (bar-level attr in `tp.attributes` or vice versa), bar
  index not in `tp.bars`, partial-AR bar override on a prefix bar,
  TS cross-track mismatch.

This mirrors the novelty/silence-check policy: always evaluate, surface
both warnings and errors with the right severity.

---

## 4. File-change summary

### Python

- **`src/python/midigpt/inference/config.py`**
  Add `bar_attributes` and `bar_controls` fields to `TrackPrompt`.

- **`src/python/midigpt/inference/validation.py`**
  Add per-bar validation: subset of `tp.bars`, level-correctness,
  partial-AR suffix-only rule, achievability warning, TS cross-track
  coherence.

- **`src/python/midigpt/inference/session.py`**
  In `_sample_step`, after the analyzer pass:
  - For infill + partial-AR (suffix bars only), write
    `bar_{TokenType}_{bar_idx}` keys from `tp.bar_attributes` /
    `tp.bar_controls` into `score.tracks[i].attributes`. These
    overwrite analyzer-derived values.
  In `_build_constraints`:
  - For full-AR, add `BarAttributeValueConstraint(track_idx, bar_idx,
    TokenType, value)` for each `tp.bar_attributes` and
    `tp.bar_controls` entry.
  - For partial-AR suffix bars, add the same constraint (since the
    suffix bar's attribute token *is* generated).

- **`src/python/midigpt/attributes/base.py`**
  Add `BaseAttribute.achievable_range(fixed_score, track_idx,
  generated_bars) -> (int, int)` with a default of full domain. Add
  `AttributeAnalyzer.attribute_levels() -> dict[str, str]`.

- **`src/python/midigpt/attributes/quantile.py`**
  Override `achievable_range` for `PolyphonyQuantile`,
  `NoteDurationQuantile`, `NoteDensityQuantile`.

### C++ + bindings

- **`src/cpp/masking/bar_attribute_value_constraint.h`** (NEW)
  Stateful constraint that observes `Track`/`Bar` tokens, tracks its
  own `(track_idx, bar_idx)` cursor, and applies the value mask only
  inside the target bar of the target track.

- **`src/cpp/bindings/lib.cpp`**
  Bind the new class to Python.

### Studio passthrough

- **`mmm_studio/app.py`**
  Accept `bar_attributes` / `bar_controls` from the request payload
  and forward to `TrackPrompt(**)`.

---

## 5. Phasing

**Phase 1 — Python-only (this PR):**
1. `TrackPrompt` fields.
2. `achievable_range` (`base.py` + `quantile.py`) and
   `attribute_levels()`.
3. `validation.py` structural checks + achievability warnings + TS
   coherence.
4. `session.py` per-bar prompt override path (infill + partial-AR
   suffix). Full-AR per-bar overrides are accepted by validation but
   silently fall back to the track-level constraint (with a one-time
   warning) until Phase 2 lands.
5. `inference.md` documentation update.
6. `mmm_studio/app.py` payload passthrough.

**Phase 2 — C++ `BarAttributeValueConstraint`:**
1. New constraint class + binding.
2. Wire into `_build_constraints` for full-AR and partial-AR-suffix
   bar overrides.
3. Remove the Phase-1 fallback warning.

**Phase 3 — UI:** per-bar attribute editor in MMM Studio
(out of scope for this doc).
