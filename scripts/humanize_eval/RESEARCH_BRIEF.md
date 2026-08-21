# Research brief: evaluating the (non-conditioned) Humanize model

**Purpose of this document.** This is a briefing + open research question, written to be
pasted into a more capable/more heavily-reasoning model to generate and prioritize a
concrete experiment plan. It is not itself a plan — it's the context + questions needed
to produce one. Everything stated as fact below has been verified against the actual
codebase/training run (file paths and numbers included); everything posed as a question
is genuinely open.

## 1. What "Humanize" is

A GPT-2-style symbolic-music model (`midigpt`) that takes a MIDI performance where
**pitch, onset (grid position), and duration are already fixed** ("the skeleton") and
regenerates only **Velocity** and **microtiming (Delta)** for chosen bars — i.e. it adds
(or replaces) *expressive performance* on top of an already-fixed note skeleton. It does
not choose what notes to play; it chooses how hard and how early/late to play them.

Encoding mechanics (`src/cpp/tokenizer/encoder.cpp`, `NoteEncodeMode` enum): a bar
selected for humanization is encoded twice — once as `SkeletonOnly` (pitch/duration,
Velocity/Delta withheld) in its normal place in the sequence, and once as `ExpressiveOnly`
(only Velocity/Delta, no pitch/duration) in an "appendix" block appended later in the same
sequence. Training loss is ordinary next-token cross-entropy over the *full* token
sequence (`labels = tokens`, see `src/python/midigpt/training/dataset.py` /
`collator.py` — `-100` is used only for batch-padding, never to mask skeleton vs.
appendix content).

## 2. Token scheme for velocity/microtiming (exact, from `models/humanize_encoder.json` + `src/cpp/`)

- **Velocity**: `VelocityLevel` token, `velocity_levels: 32` — uniform quantization of
  MIDI velocity 0-127 into 32 bins (`VelocityQuantizer`, `src/cpp/tokenizer/domain_transforms.h`).
  `velocity_sticky: true` means in *normal* encoding a new `VelocityLevel` token is only
  emitted when the level changes from the previous note — but in the Humanize
  `ExpressiveOnly` appendix it is **always** emitted per note (needed as an unambiguous
  per-note group marker when splicing the appendix back onto the skeleton).
- **Microtiming**: `resolution: 12` ticks per quarter note. Two token types per note,
  emitted only when the offset is non-zero:
  - `DeltaDirection` (domain size 2): emitted only when `note.delta < 0` (i.e. it's a
    "this note is early" marker; a note with no direction token and delta>0 is implicitly
    "late" or "on-grid" — direction is a binary earlier/not-earlier flag, not signed).
  - `Delta` (domain size `resolution/2 = 6`, values 0-5): the *magnitude* in ticks. 1 tick
    = 1/12 of a quarter note. At 120 BPM, 1 tick ≈ 4.2 ms, so the representable magnitude
    range is 0 to ~20.8 ms — **this is the ceiling of microtiming expressiveness the model
    can represent at all**, worth knowing before judging "is the microtiming expressive
    enough."
  - Per-note token order: `[DeltaDirection?] [Delta?] [VelocityLevel]`, grouped by note.

## 3. Training data and what the model has actually seen

Trained on a **filtered** subset of GigaMIDI v2.0.0 — not the full corpus. Filter
(`scripts/humanize_data_quality/filter_expressive_tracks.py`), all 3 required:
1. `NOMML == 12` (`src/python/midigpt/attributes/nomml.py` — median-quantization-depth
   metric, 0-12 ordinal; 12 means the track's note onsets don't cleanly match *any* tested
   binary/triplet subdivision grid down to 1/32 — i.e. "freely timed", not mechanically
   quantized).
2. **Onset-phase concentration** `score = 1 - H/log(12)`, where `H` is Shannon entropy of
   the histogram of `onset_tick % coarse_unit` mapped into 12 phase bins across the whole
   track. High score = onsets still cluster around *a* real metrical grid (just not a
   clean subdivision) rather than being uniformly/randomly distributed. Threshold ≥0.6.
3. **Dominant-bin residual std-ratio**: `pstdev(residuals in the dominant phase bin) /
   (coarse_unit/2)`, where `residual = onset_ticks - nearest_cell*coarse_unit`. Threshold
   ≥0.15 — rejects tracks where the "expressiveness" is actually just a constant
   quantization offset (fake humanization) rather than genuine per-note variance.

This dropped the corpus from ~1.7M rows to **43,943 qualifying pieces** (39,128
train / 4,815 validation), each pre-pruned to only its qualifying track(s)
(`humanize_tiny.json` config: `n_embd=256, n_layer=4, n_head=4,
humanize_probability=1.0, humanize_bar_fraction=0.5, max_steps=200000`). **Implication:**
the model has only ever been trained on "genuinely expressively played" reference
material — it has not seen mechanically-quantized performances as *targets* (only,
potentially, as *skeleton* input at inference time, which is an out-of-distribution
scenario worth testing explicitly, see §5).

## 4. Training status / available checkpoints

Two relevant runs exist (a third, `humanize_small`, was abandoned mid-training and should
be ignored):

- `humanize_tiny-20260804-124407` (the current, fixed-initialization run — see below):
  `val/loss` bottomed at **0.912 around step 121,999** (epoch ~100), then crept back up to
  **0.916 by step 187,999** (epoch 155) while `train/loss` kept falling (0.839 → 0.809)
  over the same span — train/val gap widened ~47% (0.073 → 0.107). This is the
  "overfitting" the user observed: **the best-generalizing checkpoint is around step
  122,000**, not the latest/final one. Both the best-val checkpoint
  (`model-step=122000.safetensors` or nearest available) and a late/overfit checkpoint
  (e.g. `model-step=188000.safetensors`) exist on disk — evaluating *both* and comparing
  is itself an interesting experiment (does overfitting show up as literal
  near-memorization of training velocity/timing patterns, or just as slightly worse
  calibration? See §6, memorization check).
- A prior run (`humanize_tiny-20260731-003724` / job 18921874, `model_final.safetensors`
  at step 200,000) was trained *before* a since-fixed bug in the shared GPT-2
  initialization (`src/python/midigpt/inference/model/gpt2.py`: embeddings/lm_head were
  falling back to PyTorch defaults instead of GPT-2's calibrated `std=0.02`, and residual
  `c_proj` layers were missing the `1/sqrt(2*n_layer)` scale-down). That run is kept only
  as a "known less-stable" baseline, not a candidate for the eval below.

## 5. What generation actually looks like today (no packaged tooling yet)

There is **no dedicated Humanize CLI or eval script yet** — this document is partly about
scoping what to build. The raw mechanism (`src/python/midigpt/inference/engine.py` +
`session.py`, doc at `src/python/midigpt/inference/inference.md`):

```python
req = GenerationRequest(
    tracks=[TrackPrompt(id=0, bars=[4, 5, 6, 7], humanize=True)],
    config=InferenceConfig(temperature=1.0, top_p=0.95),
)
score_out = engine.sample(score_in, req)
```

`TrackPrompt.humanize: bool` marks which bars get Velocity/Delta regenerated; the bars
must already contain real pitch/onset/duration (the skeleton). **Important:**
`humanize_probability` / `humanize_bar_fraction` (from the model config) are *training-time
only* knobs controlling how the training corpus was encoded — there is currently **no
inference-time control** over "how much"/"how strongly" to humanize beyond
`temperature`/`top_p` (standard sampling knobs) and which bars are selected at all. A
researcher wanting to build an eval harness will need to write the
score-in/request/score-out plumbing directly against `InferenceEngine` — nothing packaged
does this yet.

Existing, reusable attribute-computation code (don't reinvent):
`attributes/nomml.py` (quantization depth), `attributes/velocity.py`
(`VelocityRange` — bar-level max-min), and the onset-phase-concentration /
std-ratio formulas in `filter_expressive_tracks.py` (§3). Also
`augmentation/velocity.py::VelocityScale(factor: float | tuple[float,float])` —
rescales all note velocities by a sampled factor — useful for building
absolute-loudness-invariant comparisons.

## 6. The open research questions (this is what needs an experiment plan)

We want a **non-conditioned evaluation first** — the model currently has no explicit
"style" input, it only sees the note skeleton (+ whatever other bars/tracks are in its
context window) and must decide how to play it. A later phase (already scoped in a
separate design doc, see §7) adds an explicit style-conditioning vector; the eval harness
built now should anticipate that without over-building for it yet.

1. **Does it work at all, and how do we know?** What's the minimal, defensible sanity
   check that the model is doing something non-trivial and non-degenerate (not just
   emitting the sticky-default velocity level / zero delta everywhere, not just copying
   some fixed global average)? What baselines should generated output be compared
   against — e.g. (a) a "flat/mechanical" baseline (constant velocity, zero delta), (b)
   the ground-truth expressive performance for the same skeleton (held-out), (c) a
   shuffled/mismatched baseline (apply one piece's generated expressiveness pattern to a
   different piece's skeleton)?

2. **Characterizing generated expressiveness.** What does the model's *output*
   distribution of Velocity levels and Delta (direction+magnitude) look like — per-bar,
   per-piece, aggregated over many samples — compared to the *real* filtered-corpus
   distribution it was trained on? Things worth checking: variance/entropy of velocity
   across a bar or phrase (is it as varied as real performances, or regressing toward a
   narrower band — a known failure mode of models trained with plain cross-entropy on
   continuous-ish quantities), the marginal distribution of `Delta` magnitude (does the
   model use the full 0-5 tick range or cluster near 0, i.e. "safe"/underconfident
   microtiming?), and the direction bias (early vs. late — real expressive playing is
   often *not* symmetric, e.g. certain genres/instruments push notes ahead of or behind
   the beat systematically).

3. **Context-dependence without explicit conditioning — is implicit style transfer
   already happening "for free"?** Since there's no style vector yet, any style-sensitivity
   would have to come purely from attention over the surrounding token context already in
   the window (other bars of the same piece/track, or other tracks). Two concrete probes:
   - **(a) Inexpressive context → expressive target.** Take a real skeleton, but
     *replace* the context bars (bars around/before the target humanize bars, in the same
     window) with mechanically flat versions (constant mid-range velocity, zero delta —
     synthetic, never seen in training since training data was filtered to `NOMML==12`
     only) and ask the model to humanize the target bars. Does it still generate
     expressive output, or does it partially collapse toward the flat context style? This
     tests how much the model relies on "local performance style continuity" vs. a
     learned prior independent of context.
   - **(b) Expressive context → does the target match its style?** Take a real
     performance, humanize a *subset* of its bars while leaving the rest of the same
     piece as real (untouched, expressive) context. Does the generated subset's
     velocity/timing *profile* (e.g. a per-beat-position mean-velocity / mean-|offset|
     vector, not just flat aggregate stats) resemble the surrounding real bars' profile
     more than it resembles the corpus-average profile? If yes — and if effect size is
     non-trivial vs. a shuffled-context control — that's evidence the model is *already*
     doing a crude, context-implicit form of style matching, which would directly inform
     how much lift the planned explicit style-conditioning system (§7) can realistically
     add, and might reframe it as "make an already-present tendency reliable/controllable"
     rather than "add a capability from scratch."
   - Is "flat/mechanical context" or "expressive-but-different-piece context" even a
     *valid* prompt for this model, given training never included non-`NOMML==12`
     context? This itself deserves being checked as a distinct question — the model may
     behave unpredictably (not just "less expressive") on such out-of-distribution
     context, which matters both for this eval and for how the real product would ever
     encounter mixed-expressiveness inputs (e.g. a user who quantized only some tracks).

4. **Per-instrument/per-track differences.** Does generated expressiveness differ
   meaningfully by instrument or track role (e.g. drums vs. bass vs. lead vs. chordal
   accompaniment), matching the real corpus's per-instrument tendencies (drums typically
   have different velocity dynamics and much tighter timing than, say, a lead melody)? Or
   does the model homogenize expressiveness across instruments regardless of role? GigaMIDI
   rows carry per-track program/instrument metadata — worth checking what's available in
   the parquet schema and whether it's preserved through to something queryable at eval
   time (encoder config / dataset.py should confirm this).

5. **What controls exist, and what should we wish existed?** Right now: `temperature`,
   `top_p`, and *which bars* get marked `humanize=True` — that's it (see §5, no
   inference-time "strength" knob). Given that, what's a reasonable minimal experiment
   matrix over the controls that *do* exist (e.g. temperature sweep vs. expressiveness
   variance/entropy, does higher temperature actually buy more *musically plausible*
   variety or just noisier tokens)? Separately — is there a cheap, well-motivated way to
   simulate a "strength" control without retraining (e.g. resampling only some of the
   generated Velocity/Delta tokens and keeping others locked to skeleton-adjacent
   defaults; or top-k restriction to the model's per-token top choices as a "gentler"
   mode)? This connects to what the eventual conditioned system needs to expose as a
   product-level control, so it's worth thinking about now even in the non-conditioned
   phase.

6. **Memorization / overfitting check.** Given the confirmed overfitting past step
   ~122,000 (§4), does the late/overfit checkpoint show measurably higher near-copying of
   *training-set* velocity/timing sequences for a given skeleton than the best-val
   checkpoint does (e.g. nearest-neighbor search in some simple feature space between
   generated appendix tokens and the closest training-set piece with a similar skeleton)?
   This would give a concrete, checkpoint-comparable signal for "which checkpoint should
   actually be used going forward," beyond just the val-loss number.

## 7. Constraints / how this should be built

- **Prototype-first, no production changes.** Mirror the existing conventions in this
  worktree: `scripts/humanize_data_quality/` (data-quality filtering, standalone) and
  `scripts/style_prototype/` (a fuller prototype harness for the *next* phase — explicit
  style conditioning, not yet built/validated; its design doc already specifies a 5-metric
  evaluation ladder with a hard gate at metric 1 — "does true-style-vector beat
  mismatched/no-style baselines on held-out conditional loss" — which is a close cousin of
  question 3 above and should reuse the same per-beat-position velocity/timing profile
  representation if question 3's probe (b) is built). A new `scripts/humanize_eval/`
  directory, following the same flat-script-plus-`slurm_*.sh` pattern, is the natural
  place for whatever this produces.
- **Reuse existing metrics** (`nomml.py`, `velocity.py`, the onset-phase-concentration /
  std-ratio formulas) as building blocks rather than inventing new expressiveness scores
  from scratch where these already capture the relevant property.
- **No packaged generation CLI exists** — budget for writing the
  `InferenceEngine`/`TrackPrompt(humanize=True)` plumbing as part of this work, not as a
  prerequisite someone else has already solved.
- Use the best-val checkpoint (`~step 122,000`) as the primary subject of evaluation,
  with the late/overfit checkpoint (`~step 188,000`) as a secondary comparison point (§6).

## 8. What I want back

A prioritized, concrete experiment plan for the non-conditioned Humanize model covering
questions 1-6 above. For each proposed experiment: what data/inputs it needs, what exact
metric(s) it produces, what a "pass" or "interesting" result looks like, what it costs to
run (cheap sanity check vs. a heavier study), and — critically — a recommended *order*,
cheapest/most-informative-first, with explicit go/no-go gates the way the sibling
style-conditioning plan does (e.g. "don't build the per-instrument breakdown until the
basic distributional check in question 2 passes"). Flag anywhere the non-conditioned
findings would change priorities or design assumptions in the already-planned explicit
style-conditioning phase (§7), since that's the reason we're doing the non-conditioned
eval carefully now rather than skipping straight to conditioning.
