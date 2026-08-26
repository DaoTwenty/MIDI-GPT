# Context viz walkthroughs

Hand-traced examples of the per-bar prompt state for the OSC Studio
ContextView, kept as a verification log. Each example fixes random
params + a playhead, then writes out the expected state per bar
per track. The notes section at the bottom of each example records
anything I learned or got wrong.

---

## Example 1 — warmup, a_masked, lookahead=1

### Params
- 2 tracks: T0 = human loop (conditioning), T1 = agent
- `model_dim=8`, `lookahead=1`, `num_anticipated=1`
- `buffer=4`, `adapt_buffer=true`
- `warmup_policy="a_masked"`, `mask_gap=false`
- playhead = 5

### Derived
- `steady_col = 8 - 1 - 1 = 6`; playhead 5 < 6 → WARMUP, window pinned `[0, 8)`
- `playheadCol=5`, `targetCol=6`, `toGen=[6,6]`, `wasted=[7,7]`
- `firstGenPlayhead = max(0, 4-1) = 3`; first-ever target = bar 4
- pre-history = bars `[0, 4)`

### Scenario
Warmup gens at playhead=3 (target=4) and playhead=4 (target=5) both
succeeded. Current tick (playhead=5) is firing a gen for target=6.

### Per-bar state

| col | bar | T0 (human)       | T1 (agent)                              |
|-----|-----|------------------|------------------------------------------|
| 0   | 0   | CONTEXT          | MASKED_PAST                              |
| 1   | 1   | CONTEXT          | MASKED_PAST                              |
| 2   | 2   | CONTEXT          | MASKED_PAST                              |
| 3   | 3   | CONTEXT          | MASKED_PAST                              |
| 4   | 4   | CONTEXT          | GENERATED                                |
| 5   | 5   | MASKED_FUTURE    | GENERATED (playhead bar; agent already filled) |
| 6   | 6   | MASKED_FUTURE    | GENERATING / TARGET_DONE                 |
| 7   | 7   | MASKED_FUTURE    | WASTED                                   |

### Notes / what I learned
- `MASKED_PAST` only ever covers bars `< firstGenPlayhead` — the empty
  pre-history the agent never had a chance to fill. Bars between
  `firstGenPlayhead` and `playhead` are either GENERATED or
  MASKED_FAILED depending on whether each warmup gen produced notes.
- Conditioning track at `barIdx == playhead` reads as MASKED_FUTURE,
  not CONTEXT, because the bar isn't complete yet (the rule is strict
  `<`, not `<=`).
- In warmup, the wasted tail (cols past the to-gen window) exists every
  tick — those bars are AR-sampled then discarded so the model can
  finish a full `model_dim`-bar prompt. Established mode has no waste.
- `firstGenPlayhead` with `adapt_buffer=true` is `buffer - lookahead`,
  not `buffer`. With buffer=4 and lookahead=1, first gen fires at
  playhead=3, not 4.

---

## Example 6 — seed=3 v2 (max buffer, policy=b, lookahead=1, two failed gens)

### Roll
```
tracks=2  model_dim=16  lookahead=1  num_anticip=1  buffer=15  adapt_buffer=True
policy=b  mask_gap=True   playhead=29   regime=ESTABLISHED
steady_col=14  playhead_col=14  target_col=15  toGen_cols=[15,15]
first_gen_playhead=14  first_target_bar=15  failed_gen_bars=[20, 24]
```

### Expected grid (window = [15, 31))

| col | bar | T0 agent ★      | T1 human live |
|-----|-----|-----------------|---------------|
| 0   | 15  | GENERATED       | CONTEXT       |
| 1-4 | 16-19 | GENERATED     | CONTEXT       |
| 5   | 20  | MASKED_FAILED   | CONTEXT       |
| 6-8 | 21-23 | GENERATED     | CONTEXT       |
| 9   | 24  | MASKED_FAILED   | CONTEXT       |
| 10-13 | 25-28 | GENERATED   | CONTEXT       |
| 14  | 29  | GENERATED       | MASKED_FUTURE (playhead) |
| 15  | 30  | TO_GENERATE     | MASKED_FUTURE |

### Notes / what I learned
- New constraint added: `buffer_bars < model_dim`. Server rejects
  invalid combinations at `/session/start`; scenario generator
  enforces it; studio's PARAM_META updated.
- Studio param bounds aligned with these rules:
    - model_dim ∈ {4, 8, 12, 16}
    - lookahead ∈ [1, 4]
    - num_anticipated ∈ [1, 4]
    - buffer_bars ∈ [2, model_dim - 1]
- `lookahead=1` collapses the lookahead overlay to an empty range;
  the to-gen target is the only "near future" cell. `mask_gap` is
  inert in this configuration.
- `policy=b` bootstrap span is `[0, target + num_anticip)` =
  `[0, 16)`, which nearly fills the window. After the one-shot
  bootstrap, the agent is fully primed and gens look exactly like
  established-mode AR thereafter.

---

## Example 5 — seed=3 (sliding window, but no gen yet) — OBSOLETE

> ⚠️ This roll produced `buffer_bars=20 > model_dim=12`, which is now
> rejected at `/session/start` (geometrically meaningless: the first
> gen target would be beyond the window). The scenario generator has
> been updated to constrain `buffer_bars <= model_dim`. Leaving the
> example below as historical record only.

### Roll
```
tracks=2  model_dim=12  lookahead=4  num_anticip=6  buffer=20  adapt_buffer=True
policy=a_empty  mask_gap=False   playhead=9   regime=ESTABLISHED
steady_col=2  playhead_col=2  target_col=6  toGen_cols=[6,11]  lookahead_cols=[3,5]
first_gen_playhead=16  first_target_bar=20  failed_gen_bars=∅
```

### Expected grid (window = [7, 19))

| col   | bar   | T0 agent ★ | T1 human live |
|-------|-------|------------|---------------|
| 0–2   | 7-9   | CONTEXT (committed pre-hist, a_empty) | 7,8 CONTEXT · 9 PENDING (playhead) |
| 3–11  | 10-18 | PENDING (uncommitted future, !gen_active) | PENDING |

### Notes / what I learned
- Surfaces a new state: `steady_col < first_gen_playhead`. The window
  is geometrically "established" (sliding) yet no gen has fired. The
  regime indicator collapses two distinct conditions ("window sliding"
  and "agent active") into one boolean. Worth adding a tri-state to
  the UI later: e.g. WARMUP / PRE-GEN-SLIDE / ACTIVE.
- `policy=a_empty` makes committed agent silence indistinguishable
  from played human context — both render as plain `CTX`. With
  `a_masked` the agent's pre-history would render as `M·P`,
  visually distinct.
- `lookahead_cols` and `toGen_cols` are wide (3 + 6) but invisible
  in this view because no gen is firing. They'd snap into view the
  moment playhead reaches 16.

---

## Example 4 — seed=2 (established, policy=b, num_anticip=3, two failed gens)

### Roll
```
tracks=2  model_dim=8  lookahead=2  num_anticip=3  buffer=6  adapt_buffer=True
policy=b  mask_gap=True   playhead=13   regime=ESTABLISHED
steady_col=3  playhead_col=3  target_col=5  toGen_cols=[5,7]  lookahead_cols=[4,4]
first_gen_playhead=4  first_target_bar=6  failed_gen_bars=[12, 13]
```

### Expected grid (window = [10, 18))

| col | bar | T0 agent ★      | T1 human loop |
|-----|-----|-----------------|---------------|
| 0   | 10  | GENERATED       | CONTEXT       |
| 1   | 11  | GENERATED       | CONTEXT       |
| 2   | 12  | MASKED_FAILED   | CONTEXT       |
| 3   | 13  | MASKED_FAILED   | MASKED_FUTURE |
| 4   | 14  | GENERATED       | MASKED_FUTURE |
| 5   | 15  | TO_GENERATE     | MASKED_FUTURE |
| 6   | 16  | TO_GENERATE     | MASKED_FUTURE |
| 7   | 17  | TO_GENERATE     | MASKED_FUTURE |

### Notes / what I learned
- Confirms the lookahead-vs-completed fix from Example 2: bar 14 sits
  in the lookahead overlay zone AND has notes (prior tick's target).
  `completed` now wins so it paints GEN rather than the overlay color.
- Failed-gen state (`M·X`) is independent of where the failure sits
  (before or at the playhead). Both bars 12 and 13 show the same
  state — that's correct because the encoder treats them the same.
- `policy=b` is invisible here: the bootstrap fired ages ago at
  playhead=4 with target span `[0, 6)`, all behind the current window.
  Bars 0–5 in the score still hold the bootstrap content; they just
  scrolled offscreen.
- `num_anticip > lookahead` is legal — the to-gen tail (3 bars) is
  wider than the lookahead overlay (1 bar). The script handled it
  correctly without special-casing.

---

## Example 3 — seed=1 (pre-buffer warmup, mask_gap, a_empty)

### Roll
```
tracks=2  model_dim=4  lookahead=2  num_anticip=1  buffer=5  adapt_buffer=True
policy=a_empty  mask_gap=True   playhead=0   regime=WARMUP
steady_col=1  playhead_col=0  target_col=2  toGen_cols=[2,2]  lookahead_cols=[1,1]
first_gen_playhead=3  first_target_bar=5  failed_gen_bars=∅
```

### Expected grid

|     | col 0 (bar 0) | col 1 (bar 1) | col 2 (bar 2) | col 3 (bar 3) |
|-----|---------------|---------------|---------------|---------------|
| T0 agent ★ | CONTEXT (a_empty, pre-hist) | MASKED_FUTURE (lookahead + mask_gap) | PENDING (to-gen, gen_active=false) | PENDING (wasted col, gen_active=false) |
| T1 human   | MASKED_FUTURE (bar == playhead) | MASKED_FUTURE | MASKED_FUTURE | MASKED_FUTURE |

### Notes / what I learned
- **Bug found**: when I rewrote `_getBarState` to put `completed` first,
  I dropped the `mask_gap` handling for the lookahead overlay. Empty
  lookahead bars with `mask_gap=true` were falling through to the
  policy/failed-gen branch and rendering as silent CONTEXT. Restored
  the `if (isLookahead && maskGap) return 'MASKED_FUTURE'` check
  *after* `completed` so notes still win.
- **Bug in script**: `bar_state` was returning `TO_GENERATE` for the
  to-gen col regardless of `gen_active`. The real viz gates this on
  `gen_active` and falls back to `PENDING` before the buffer fills.
  Fixed.
- Scenario also surfaces an edge case: `buffer_bars > model_dim` is
  legal; it just means `first_target_bar` (= 5 here) sits past the
  initial window. The window will catch up once the playhead advances.
- `MASKED_FUTURE` vs `PENDING` are *not* synonyms: `MASKED_FUTURE` is
  an active `MaskBar` token in the prompt right now; `PENDING` means
  no gen is firing this tick, so the slot has no prompt-level meaning
  yet. Pre-buffer is the only place where the agent track shows
  `PENDING`.

---

## Example 2 — established, lookahead=2, a_empty

### Params
- 2 tracks: T0 = human live, T1 = agent
- `model_dim=6`, `lookahead=2`, `num_anticipated=1`
- `buffer=4`, `adapt_buffer=true`
- `warmup_policy="a_empty"`, `mask_gap=false`
- playhead = 10

### Derived
- `steady_col = 6 - 1 - 2 = 3`; playhead 10 ≥ 3 → ESTABLISHED
- `playheadCol=3`, `targetCol=5`, `toGen=[5,5]`, lookahead overlay `=[4,4]`
- window = `[7, 13)`; col 0 → bar 7, col 3 → bar 10, col 5 → bar 12
- `firstGenPlayhead = max(0, 4-2) = 2`; first target ever = bar 4
- pre-history bars [0, 4) — all out of window at this playhead

### Scenario
First warmup gen (playhead=2, target=4) failed → bar 4 empty.
All later gens succeeded. By playhead=10, bars up through 11 are
filled (bar 11 was prev tick's target). Bar 12 is firing now.

### Per-bar state (current code behavior)

| col | bar | T0 human live    | T1 agent                                       |
|-----|-----|------------------|------------------------------------------------|
| 0   | 7   | CONTEXT          | GENERATED                                      |
| 1   | 8   | CONTEXT          | GENERATED                                      |
| 2   | 9   | CONTEXT          | GENERATED                                      |
| 3   | 10  | MASKED_FUTURE    | GENERATED                                      |
| 4   | 11  | MASKED_FUTURE    | CONTEXT  ← suspected bug (should be GENERATED) |
| 5   | 12  | MASKED_FUTURE    | GENERATING / TARGET_DONE                       |

### Notes / what I learned
- Pre-history slides out of the window in established mode, but the
  encoder still sees those bars on every gen (they're in the score) so
  the policy still matters until the score actually drops them.
- **Bug found**: `_getBarState` runs the lookahead-overlay branch
  before the `completed` check. A lookahead-zone bar that was
  already generated by a prior tick gets painted as CONTEXT (or
  MASKED_FUTURE under mask_gap) instead of GENERATED. The fix is to
  check `completed` first, then fall back to the lookahead branch
  when the bar is empty.
- Similarly, a lookahead bar that was NEVER generated (prior tick
  failed) can't currently be shown as MASKED_FAILED — same fix
  applies: only treat the bar as a pure lookahead-overlay when it
  has no notes AND isn't past `firstGenPlayhead` with a failed
  history.

---
