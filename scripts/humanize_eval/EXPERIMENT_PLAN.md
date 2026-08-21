# Experiment plan: evaluating the non-conditioned Humanize model

Response to `RESEARCH_BRIEF.md`. Written after reading the encoder/decoder/session code and
running one empirical probe on the held-out corpus. Structure:

- §A — corrections to the brief (things that will break a harness written from it as-is)
- §B — one measurement that changes the shape of the whole evaluation
- §C — questions I'd add to the brief's six
- §D — the prioritized ladder, with gates
- §E — implications for the style-conditioning phase

---

## §A. Corrections to the brief

Each of these was verified against the code, not inferred.

**A1. `engine.sample()` does not exist.** The brief (§5) and
`src/python/midigpt/inference/inference.md:24` both use it; `grep -rn "def sample" src/python/`
returns nothing. The real call is:

```python
score_out = engine.session(score_in, req).run()
```

**A2. The per-note appendix token order is reversed in the brief.** Brief §2 says
`[DeltaDirection?] [Delta?] [VelocityLevel]`. The code emits
**`VelocityLevel` → `[DeltaDirection]` → `[Delta]`** (`src/cpp/tokenizer/encoder.cpp:61-84`,
`src/cpp/masking/grammar_constraint.cpp:245-247`). This is load-bearing, not cosmetic:
`VelocityLevel` is the *group-start marker* that `resolve_humanize`
(`src/cpp/tokenizer/decoder.cpp:74-83`) uses to split the appendix into per-note groups. A
parser written to the brief's order will mis-associate every velocity with the wrong note.

**A3. Delta units are 1/144 of a quarter, not 1/12.** Brief §2 says "1 tick = 1/12 of a
quarter note … representable range 0 to ~20.8 ms". Actually `resample_delta`
(`src/python/midigpt/tokenizer/tokenizer.py:9-45`) defines delta as *the leftover fraction of
one target-resolution cell, scaled by target_res*. At `resolution: 12` one cell is 1/12 quarter
and one delta unit is 1/12 of that = **1/144 quarter ≈ 3.5 ms @ 120 BPM**. Max representable
offset is `Delta=5` → **≈ 17.4 ms**. The brief's headline number was roughly right by
coincidence; the derivation wasn't.

**A4. `DeltaDirection` never fires. The model has never seen it.** Measured over 124,076
notes from 150 held-out pieces: **0.0000%**. Cause: `Score.from_bytes` reads at TPQ 480, and
`resample_delta` then computes `new_onset = int(true_pos)` — *truncation*, so the residual is
always in `[0, 1)` and delta is always ≥ 0. (`MidiReader` itself computes a signed
round-to-nearest residual at `midi_reader.cpp:194`, but that sign is destroyed by the 480→12
resample that every training and inference path goes through.)

> **Brief §6.2's "direction bias — early vs late asymmetry" question is not answerable with
> this model.** There is no "early" token in the training distribution. Drop the question or
> reframe it as §B below.

**A5. `Delta=5` is a saturation bin holding a third of all notes.** Measured marginal on the
held-out corpus (after the `std::min(d, 5)` clamp at `encoder.cpp:82`):

| Delta | 0 | 1 | 2 | 3 | 4 | **5** |
|---|---|---|---|---|---|---|
| % of notes | 15.7 | 12.7 | 17.3 | 13.1 | 8.3 | **32.9** |

Pre-clamp, the raw magnitudes run 0–12 and there is a second mode up at d=10 (14.0% of notes
on its own). Because the residual is a *truncated* fraction, `d≈10-12` means "a hair **early**
relative to the next grid point" — and it gets clamped down to `Delta=5`, i.e. rendered as
maximally *late*. So the top bin is a mixture of "genuinely late by ≥17 ms" and "slightly
early", pointing opposite directions.

> **Brief §6.2's "does the model use the full 0-5 range or cluster near 0?" is confounded.**
> The reference distribution isn't concentrated near 0 — it's bimodal with a garbage bin at the
> top. Matching that histogram is not evidence of good microtiming.

**A6. The corpus has no drums.** `filter_expressive_tracks.py` sets `TYPE_FILTER = "no-drums"`,
and `build_filtered_parquet.py` prunes every row to its passing tracks only. Brief §6.4's
"drums vs bass vs lead" comparison can't be run — drums are strictly out-of-distribution here.

**A7. `humanize_tiny-20260731-003724` has no weights on disk** (only `wandb/`). The completed
200k-step run the brief describes is `humanize_tiny-20260804-065806`
(`model_final.safetensors`). Both target checkpoints of the live run do exist:
`runs/humanize_tiny-20260804-124407/checkpoints/model-step={122000,188000}.safetensors`.
`metrics.jsonl` confirms the brief's numbers (best val 0.91192 @ 121999; 0.91580 @ 187999).

**A8. The default `bars_per_step=1` makes humanization sequential over the model's own
output.** `SamplingSession.run()` feeds each step's result back in
(`session.py:355`), so bar *N* is humanized conditioned on the model's generated bars
`< N`, not on the real performance. This is a confound in *every* experiment below and is
itself a quality lever — see §C-F.

---

## §B. The measurement that reframes the evaluation: the reconstruction ceiling

The encoding is lossy in a way nobody has quantified. Round-tripping ground truth through
`normalize_input → encode → decode` gives onset error, measured on the same 124k notes:

| | mean | median | p90 |
|---|---|---|---|
| onset error (12-TPQ cells) | 0.134 | 0.033 | 0.458 |
| @ 120 BPM | 5.6 ms | 1.4 ms | **19.1 ms** |

19 ms at the 90th percentile is squarely audible — it's the same order as the entire
expressive range the model is being asked to produce. **A perfect model, one that predicted the
true appendix token-for-token, would still land ~19 ms off on a tenth of its notes.**

Two consequences:

1. **Every metric must be computed against *round-tripped* ground truth, not raw ground truth.**
   Round-tripped GT is the oracle / upper bound. Scoring against raw GT systematically
   understates the model and makes "how much headroom is left" unanswerable. This baseline is
   missing from the brief's list in §6.1 and it's the most important one.
2. **The representation, not the model, may be the binding constraint on microtiming quality.**
   Worth flagging to whoever owns the next training run: changing `resample_delta` to
   round-to-nearest with a *signed* residual (which would activate `DeltaDirection`, halve the
   worst-case error, and un-mix the top bin) is a small change with a large ceiling effect.
   Retrain cost is low — the live run reached 198k steps in ~7.5 h on a `h100 2g.20gb` MIG
   slice. I'd prototype the fixed encoding offline (measure the new reconstruction error, no
   training) in parallel with E0/E1 below, and decide from that number.

Velocity, for reference (same probe): the marginal peaks hard at levels 16-18 (33.6% of notes
in three of 32 bins — the MIDI 64/80/100 defaults), per-piece velocity-level `pstdev` = 2.46
levels, per-piece entropy = 2.50 bits of a possible 5. Those are the numbers generated output
has to match.

---

## §C. Questions I'd add

**C-A. Evaluate by likelihood first, not by sampling.** The brief jumps straight to generating
and comparing distributions. But `SamplingSession.score_from_tokens(token_ids, step_idx)`
(`session.py:820`) already gives teacher-forced log P of an arbitrary token sequence under a
built prompt. That means we can score the *true* appendix of a held-out piece under different
contexts, with no sampling noise, no generation plumbing, and no metric design — a paired test
on identical targets, so statistical power is enormous. It answers brief Q1 (does it beat a
marginal baseline), Q3 (is context being used), and Q6 (checkpoint comparison) in one cheap
job. This should be the first thing built and it is also the exact analogue of the
style-prototype's metric-1 hard gate.

**C-B. Absolute loudness is unidentifiable — don't measure it.** Training applies
`VelocityScale((0.8, 1.2))` unconditionally (`dataset.py:526-527`), and `SkeletonOnly` withholds
velocity entirely, so when *all* bars are humanized the model has no signal at all about the
intended global level. A large fraction of velocity NLL is irreducible by construction. Metrics
must be on **contour** — mean-removed profiles, rank correlation — not absolute MAE.

**C-C. Coverage fraction is the primary experimental axis, not a footnote.** Humanizing 100% of
bars (no expressive context anywhere → pure prior) and humanizing 50% of bars (training's
`humanize_bar_fraction`) are qualitatively different tasks. The brief folds this into Q3; I'd
promote it to a first-class factor: sweep {100%, 50%, 25%} and expect the model to look much
better at 50% than at 100%. If it doesn't, that's the headline finding.

**C-D. Does the model know *why* to accent?** This is the metric I'd care most about and the
brief has no equivalent. A model can nail every marginal distribution while emitting
structureless noise. Test: regress velocity on musical features — metrical position within the
bar, pitch rank within the simultaneous onset group, whether the note is the top voice,
interval from the previous note — and compare **R² for round-tripped GT vs. generated vs. a
shuffled control**. If GT gets R²≈0.4 and generated gets R²≈0.05, the model is producing
plausible histograms and musically empty output, and every distributional metric above it is
misleading. Cheap (a linear regression), high information.

**C-E. Temperature is not a knob, it's the dispersion dial.** Cross-entropy models sampled at
τ=1 are over-dispersed relative to conditional means; at τ<1 they regress to the mode. NLL
(§C-A) is τ-independent; every distributional metric is τ-dependent. So: find τ\* that matches
GT per-piece velocity dispersion, then report all sampling-based results at τ\*. Reporting
distributional match at an arbitrary τ=1.0 is not interpretable.

**C-F. `bars_per_step` needs to be an experimental condition.** `bars_per_step=1` (default)
vs. `bars_per_step=model_dim` (single-shot over the window) is the difference between
"conditioned on own output, drift possible" and "conditioned on real context". Run both at
least once early; if they differ materially, fix one for all later experiments and say which.

**C-G. Statistical design.** With 4,815 held-out pieces there's no excuse for eyeballed
histograms. Everything paired per-piece, bootstrap CIs over pieces, effect sizes normalized
against the shuffled-context control. The unit of analysis is the piece, not the note — notes
within a piece are massively correlated.

**C-H. The "strength" control the brief asks for (Q5) doesn't need a sampling trick.** Because
humanize output is a pure *residual* on a fixed skeleton, and we hold both the mechanical
version and the generated version, a strength knob is a post-hoc α-blend of velocity and delta
between the two. That's trivially implementable, monotone, and continuous — strictly better
than token-level resampling games. Note it's a *different axis* from τ: α controls magnitude of
deviation, τ controls variety. The eventual product wants both.

---

## §D. The ladder

Order is cheapest-and-most-informative first. Each gate is a genuine stop.

### E0 — Reference statistics and the encoding ceiling *(no model; CPU; ~1 h)*
Partially done already (numbers in §A5, §B). Complete it over the full 4,815-piece validation
split.

- **Inputs:** `data/humanize_filtered/validation.parquet`, `models/humanize_encoder.json`.
- **Produces:** GT marginals for `VelocityLevel` / `Delta`; per-piece velocity dispersion and
  entropy; round-trip onset error distribution; the per-beat-position velocity/timing profile
  (reuse the representation the style-prototype doc specifies, so E4 and the conditioning phase
  share it); the §C-D structural R² for GT; a breakdown of the delta histogram by *source* TPQ
  to confirm the d≈10 mode is a resampling artifact rather than performance nuance.
- **Also produces:** the fixed-encoding counterfactual — reconstruction error under signed
  round-to-nearest delta, no training required.
- **Gate:** none, this is prerequisite. But if the fixed-encoding counterfactual cuts p90 error
  by >2×, escalate the retrain decision before spending on E2+.

### E1 — Held-out teacher-forced NLL with context ablations *(GPU, no sampling; ~2 h)*
The hard gate. Build the `InferenceEngine` + `TrackPrompt(humanize=True)` plumbing here; it's
reused by everything below.

- **Method:** for each held-out piece, build the humanize prompt, then score the *true*
  appendix tokens with `score_from_tokens`. Report NLL per token, split by token type
  (velocity vs. delta) — they'll behave very differently and an aggregate hides that.
- **Conditions:**
  (a) true context; (b) context bars flattened to constant velocity / zero delta;
  (c) context bars swapped in from a different piece; (d) uniform baseline;
  (e) corpus-marginal (unigram) baseline; (f) both checkpoints, 122k and 188k.
- **Gates:**
  - **G1a:** model NLL must beat the corpus-marginal baseline decisively on *both* token types.
    If it doesn't beat unigram on velocity, the model isn't doing anything — stop and debug.
  - **G1b:** true context must beat swapped context, paired, with a CI excluding zero. This is
    the "is implicit style transfer happening at all" answer, obtained without generating a
    single note. **If G1b fails, skip E4 entirely** and treat brief Q3 as answered "no".
  - **G1c:** if 122k and 188k are indistinguishable on held-out NLL, that alone doesn't settle
    the checkpoint question — carry both into E2, drop the loser after.

### E2 — Sampling, calibration, and τ\* *(GPU; ~4 h)*
- Generate for ~500 held-out pieces × τ ∈ {0.7, 0.85, 1.0, 1.15} × top_p ∈ {1.0, 0.95},
  at 50% coverage, both `bars_per_step` settings (§C-F).
- **Metrics vs. round-tripped GT:** per-piece velocity marginal distance (Wasserstein — `scipy`
  is available via the module stack); per-piece dispersion and entropy; delta marginal;
  **degeneracy rate** = fraction of pieces where generated velocity `pstdev` < 0.5 levels.
- **Baselines:** flat/mechanical; round-tripped GT (oracle); shuffled (piece A's expression on
  piece B's skeleton).
- **Gate G2:** at some τ, (i) degeneracy rate ≈ 0, and (ii) per-piece dispersion CI overlaps
  GT's. Fix τ\* here. If no τ gets dispersion into range, the model is mode-collapsed on
  velocity and that's the finding — report it and stop before E4/E5.

### E3 — Structural correctness *(CPU, on E2's outputs; ~1 h)*
The §C-D regression, at τ\*: R² and per-feature coefficients for GT / generated / shuffled.
Plus the per-beat-position profile correlation against GT.

- **Gate G3:** generated R² should reach a meaningful fraction of GT's R² (I'd want ≥50%), and
  the sign of the metrical-position coefficient must match GT's. Failing this while passing G2
  means "right histogram, wrong music" — a much more useful diagnosis than either metric alone.

### E4 — Context probes *(GPU; ~3 h)* — **only if G1b passed**
The sampling versions of the brief's Q3(a)/(b), now with a known-good τ\* and a known-real
effect from E1.
- Coverage sweep {100%, 50%, 25%} (§C-C).
- Flat-context OOD probe: does output collapse toward the flat style, or does it go
  *unpredictable*? Measure both a style-match score and a degeneracy/fluency check —
  the brief is right that "less expressive" and "broken" are different failure modes and
  training never contained non-`NOMML==12` context.
- Expressive-context style match: generated bars' per-beat profile vs. surrounding real bars'
  profile vs. corpus-average profile, scored against a shuffled-context control.

### E5 — Memorization / checkpoint decision *(CPU-heavy; ~6 h)* — **only if E1 left it open**
Nearest-neighbour search of generated appendix feature vectors against the training split,
122k vs 188k. Expensive and probably redundant: E1 + E2 will most likely have separated the
checkpoints already. Deliberately last.

### E6 — Per-instrument *(cheap, on E2's outputs)* — **reframed**
No drums (§A6). Group melodic programs by `instrument_merge_groups` from the encoder config and
check whether generated per-group velocity/timing profiles track GT's per-group differences. If
the model homogenizes across programs, that's a direct argument for conditioning.

### E7 — Controls prototype *(cheap)*
Implement the §C-H α-blend strength knob, produce a small α × τ grid of audio-ready MIDI for
listening. Informal, last, and explicitly not a metric.

---

## §E. What this changes for the style-conditioning phase

- **E1's G1b is a direct read on how much lift conditioning can add.** If true-context already
  beats swapped-context by a large margin, the explicit style vector is "make an existing
  tendency controllable", and the conditioning plan's own metric-1 gate should be
  re-baselined against *implicit context* rather than against no-style — otherwise it will
  clear a bar the model already clears for free.
- **Share the per-beat-position profile implementation** between E0/E4 here and metric 2 there.
  Write it once in `scripts/humanize_eval/`, import it from `scripts/style_prototype/eval/`.
- **If §B's encoding ceiling turns out to bind**, fix the representation *before* training
  anything conditioned. Conditioning a model whose microtiming target is a bin that conflates
  early and late is spending GPU time on a corrupted signal.
- **If E3 fails** (histograms right, structure absent), the conditioning design assumption
  changes: a global per-performance `z` won't fix a model that isn't attending to local musical
  structure, and the priority shifts to representation/architecture over conditioning.

## Conventions

New code in `scripts/humanize_eval/`, flat scripts + `slurm_*.sh`, mirroring
`scripts/humanize_data_quality/`: `argparse` with long kebab-case flags, module docstring with
a `Usage:` block, per-piece JSON + a `summary.json`, `sys.path.insert` sibling imports. SLURM
preamble must load `arrow/19.0.1` **before** activating `$SCRATCH/MIDI-GPT/venv-humanize`.
`scipy`, `pandas`, `matplotlib` come free from the module stack; `sklearn` and `seaborn` do not
(the §C-D regression should use `numpy.linalg.lstsq`).
