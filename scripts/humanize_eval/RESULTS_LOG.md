# Humanize — Results Log

Single source of truth for every eval result produced for this project.
Raw summary JSON is archived under `results/<run_dir>/` in this repo (git-tracked,
small — ~520K total) because the canonical copies live under `$SCRATCH/MIDI-GPT/humanize_eval/`,
which is subject to the cluster's scratch purge policy and is **not** durable.

**Convention going forward**: every time an eval script finishes a run whose result
should count as canonical (not a smoketest/debug iteration), copy its `*_summary.json`
into `results/<run_dir>/` here and add a row below in the same commit. Debug/smoketest
runs (e.g. `e1_smoketest`, `e2_smoketest2`, `e0`, `e0_postfix`, `e0_v2` — superseded by
bugfixes, see the entries they were replaced by) are intentionally not archived; only the
canonical run per (eval, checkpoint) is kept as the record.

Training matrix definitions (A-E) and eval suite definitions (E0-E8) are in
`EXPERIMENT_PLAN.md` / `RESEARCH_BRIEF.md` in this directory.

## Checkpoint paths

| Ckpt | Run dir | Notes |
|---|---|---|
| A | `runs/humanize_tiny-20260807-035822/model_final.safetensors` | baseline: per-cell random targeting, always-expressive context |
| B | `runs/humanize_tiny_b-20260807-223944/model_final.safetensors` | A + structured targeting (`structured_target_probability=0.4`) |
| C | `runs/humanize_tiny_c-20260807-223944/model_final.safetensors` | B + mechanical context, naive/unconstrained targeting (`context_mechanical_fraction=1.0`, `mechanical_coherent_targets=false`) |
| D | `runs/humanize_tiny_d-20260807-223944/model_final.safetensors` | C + coherent targeting, 10% residual incoherent (`mechanical_coherent_targets=true`, `mechanical_coherent_residual=0.1`) |
| E | `runs/humanize_tiny_e-20260807-223945/model_final.safetensors` | D but strict, 0% residual (`mechanical_coherent_residual=0.0`) |

All paths relative to `$SCRATCH/MIDI-GPT/`.

## 2026-08-06 — E0: reference-stats / encoding reconstruction ceiling

Full validation set (4,815 pieces, 4.1M notes). Canonical run: `e0_postfix_paircorrected`
(supersedes `e0`, `e0_postfix`, `e0_v2` — those predate the Delta sign/domain-size fix
and/or had a chord-pairing bug in the eval script itself, both fixed 2026-08-05/06).

- Reconstruction error (current pipeline, signed round-to-nearest): p90 = 1.736 ms @ 120bpm,
  mean = 0.865 ms, median = 0.694 ms (833,144 delta events) — matches the theoretical floor
  for the resolution/tick config; not a bug.
- `delta_direction_fire_rate_pct` = 31.0% (DeltaDirection token fires on ~31% of notes).
- Side investigations (not part of the checkpoint matrix, one-off diagnostics):
  - `e0b` — delta-mode investigation (is the 65/35 DeltaDirection skew + ~19% non-organic
    timing a real data property or an artifact?). Conclusion: real data property, not a bug.
  - `e0c` — source-TPQ artifact check. All corpus pieces are `source_tpq=480`; no
    cross-TPQ artifact to control for.

Archived: `results/e0_postfix_paircorrected/e0_summary.json`, `results/e0b/e0b_summary.json`,
`results/e0c/e0c_summary.json`.

## 2026-08-07/08 — E1-E8, checkpoints A-E (the main comparison matrix)

All 30 jobs (6 eval suites × 5 checkpoints; E7 excluded — informal/not a gated metric)
confirmed `COMPLETED` via `sacct` before any number below was trusted. Full report
published as an artifact (see chat history) synthesizing the collapse-vs-precision
trade-off finding.

### E1 — conditional NLL / context-sensitivity

Gates: G1a (model beats uniform/marginal/lag-1 baselines on every token type — all 5
checkpoints pass) and G1b (true context reduces NLL vs. swapped context, paired,
CI must exclude zero).

| Ckpt | G1b mean diff (swapped−true NLL) | 95% CI | Pass |
|---|---|---|---|
| A | 0.970 | [0.946, 0.993] | yes |
| B | 0.890 | [0.867, 0.913] | yes |
| C | 0.715 | [0.694, 0.734] | yes (weakest margin) |
| D | 0.856 | [0.833, 0.880] | yes |
| E | 0.813 | [0.791, 0.836] | yes |

Archived: `results/e1_full[_b|_c|_d|_e]/e1_summary.json`.

### E2 — sampling calibration (16 conditions: batching × τ × top_p)

Gate G2 = excess degeneracy (generated − GT degeneracy rate) within tolerance, per condition.

| Ckpt | Conditions passing |
|---|---|
| A | 7/16 |
| B | 8/16 |
| C | 8/16 |
| D | 5/16 |
| E | 5/16 |

Archived: `results/e2_full[_b|_c|_d|_e]/e2_summary.json`.

### E3 — structural correctness (velocity regressed on metrical/pitch/voice/interval features)

Gate G3 = R²_ratio (generated/GT) ≥ 0.5 AND matching coefficient signs vs. GT.

| Ckpt | R²_ratio | r2_pass | sign_pass | Overall |
|---|---|---|---|---|
| A | 0.931 | yes | yes | pass |
| B | 0.958 | yes | yes | pass |
| C | 0.832 | yes | yes | pass |
| D | 1.146 | yes | yes | pass |
| E | 1.495 (500pc) / 1.392 (2000pc) | yes | **no** (500pc) / **yes** (2000pc) | **fail→pass**, see below |

**Resolved 2026-08-08**: E's original failure (500-piece sample) was noise floor, as
suspected. Added piece-level bootstrap sign-stability to `metrical_pos`'s coefficient
(`bootstrap_r2_ci` in `e3_structural_correctness.py`) and reran E at 4x the sample
(2000 pieces, 28,027 notes, `results/e3_full_e_rerun/e3_summary.json`, job 19364994):
`gt_metrical_pos_sign_stability=0.962` (the sign is well-determined after all, just not
at n=500) and **`sign_match_pass` now True, `overall_pass` now True**. Gate logic also
updated to treat a sign mismatch as inconclusive rather than a hard fail when GT's own
bootstrap sign stability is below 0.9 (i.e. not double-counting a coefficient that isn't
itself reliably estimated) — didn't end up needed for E's case since more data resolved
it outright, but kept for future runs where a smaller sample is used.

Archived: `results/e3_full[_b|_c|_d|_e]/e3_summary.json`.

### E4 — context ablation (true/flat/swap context, single-track probe)

| Ckpt | Degeneracy true/flat/swap | style_match_effect (mean [CI]) |
|---|---|---|
| A | 0.133 / 0.707 / 0.187 | 0.362 [0.250, 0.480] |
| B | 0.157 / 0.560 / 0.163 | 0.274 [0.169, 0.384] |
| C | 0.203 / 0.317 / 0.213 | 0.108 [−0.006, 0.208] — **CI crosses zero** |
| D | 0.170 / 0.250 / 0.213 | 0.141 [0.015, 0.251] |
| E | 0.170 / 0.240 / 0.153 | 0.165 [0.057, 0.281] |

Archived: `results/e4_full[_b|_c|_d|_e]/e4_summary.json`.

### E6 — per-instrument homogenization (excludes drums)

| Ckpt | Pearson (vel, mean) | Cross-group variance ratio (gen/GT) |
|---|---|---|
| A | 0.805 | 0.863 |
| B | 0.907 | 1.339 |
| C | 0.890 | 1.256 |
| D | 0.948 | 1.144 |
| E | 0.770 | 1.056 |

**Update 2026-08-08**: E's raw pearson (0.770, worst of five) was a small-sample artifact —
`pearson_vel_mean` was unweighted across all reported groups down to `MIN_PIECES_PER_GROUP=3`,
and a single n=3 group (GM program 44) with a 46-point GT/generated velocity gap alone
swings a 44-group correlation (leave-one-out: dropping it moves r from 0.770 to 0.925).
Added `pearson_vel_mean_robust` (groups with >=10 pieces only) to `e6_per_instrument.py`
and backfilled it into all 5 existing summaries from already-saved per-group data (no
rerun needed). Robust ranking: D=0.985, B=0.966, A=0.960, E=0.945, C=0.910 — E is no
longer an outlier; C is now the weakest, consistent with it being the weakest checkpoint
elsewhere. See `homogenization.pearson_vel_mean_robust` in each `e6_summary.json`.

Archived: `results/e6_full[_b|_c|_d|_e]/e6_summary.json`.

### E8 — realistic multi-track deployment scenarios

Degeneracy rate / mean velocity Wasserstein, per scenario:

| Ckpt | whole_track | mechanical_ctx | mixed_ctx | vertical_slice | whole_piece |
|---|---|---|---|---|---|
| A | 0.10 / 9.9 | 0.52 / 17.7 | 0.41 / 16.2 | 0.06 / 4.1 | 0.04 / 17.9 |
| B | 0.10 / 10.5 | 0.47 / 14.9 | 0.34 / 14.5 | 0.06 / 5.2 | 0.06 / 16.6 |
| C | 0.09 / 11.9 | 0.17 / 22.0 | 0.12 / 17.0 | 0.07 / 8.4 | 0.10 / 21.2 |
| D | 0.07 / 11.3 | 0.09 / 20.4 | 0.13 / 17.1 | 0.08 / 6.2 | 0.04 / 17.2 |
| E | 0.08 / 11.3 | 0.11 / 19.9 | 0.14 / 16.7 | 0.07 / 6.0 | 0.01 / 17.8 |

`track_by_track` (iterative, one track humanized at a time from a fully-mechanical
starting piece) — mean Wasserstein-to-GT by step (0-5). **Degeneracy rate by step is
not currently tracked, only Wasserstein — known gap, action item below.**

| Ckpt | step0 | step1 | step2 | step3 | step4 | step5 |
|---|---|---|---|---|---|---|
| A | 18.20 | 17.96 | 17.27 | 19.26 | 16.01 | 15.28 |
| B | 18.33 | 18.53 | 16.54 | 15.72 | 14.66 | 15.37 |
| C | 24.02 | 20.55 | 18.87 | 18.11 | 16.39 | 16.56 |
| D | 25.10 | 23.16 | 23.79 | 24.73 | 22.53 | 22.50 |
| E | 24.04 | 21.91 | 18.85 | 18.13 | 17.84 | 19.04 |

Archived: `results/e8_full_A/e8_summary.json`, `results/e8_full_[b|c|d|e]/e8_summary.json`.

### E7 — controls prototype (informal, not gated)

Alpha (mechanical-anchor↔generated blend, strength) × tau (sampling variety) MIDI grid,
see `e7_controls_prototype.py` docstring for the exact mechanism. Checkpoint A run
2026-08-07 ~18:05 (predates B/C/D/E finishing training). **Update 2026-08-08**:
also run for B and E (jobs 19364784/19364785, both `COMPLETED`, ~50s each) — same grid,
4 pieces × 2 tau × 5 alpha = 40 files each. Output: `$SCRATCH/MIDI-GPT/humanize_eval/
e7_controls[_b|_e]/`. Not archived in git (audio-adjacent binary output, not a metric).
Nobody has listened yet — still open.

## Decision — 2026-08-08: keep A / B / E, drop C / D

Reasoning (research-publication framing: keep whichever checkpoints own a distinct
metric niche, drop ones that are dominated):

- **C dropped**: weakest E1 margin, and E4 `style_match_effect` CI crosses zero (no
  statistically distinguishable context-style effect) — the only checkpoint with this
  problem. Worst E8 Wasserstein in 4/5 scenarios. No metric where C is the best choice.
- **D dropped**: dominated by E. D's only edges (mechanical_ctx degeneracy 0.09 vs
  E's 0.11; whole_track 0.07 vs 0.08) are within noise at this sample size. E matches
  or beats D everywhere else, decisively on `track_by_track` (E recovers 24→19 over
  6 steps; D stays flat 22-25 the entire time).
- **A kept**: best precision under trustworthy context (E1=0.970, E4 style-match=0.362,
  best E8 Wasserstein on easy scenarios).
- **B kept**: best generalist — best `track_by_track` recovery of all five, best E8
  Wasserstein on 3/5 scenarios, tied-best E2 calibration.
- **E kept**: best collapse-robustness under bad/mechanical context (best whole_piece
  and near-best mechanical_ctx degeneracy) without D's track_by_track failure.

## 2026-08-08 — Test-set confirmatory run (item 5), results as they land

Full E1/E2/E3/E4/E6/E8 suite for A/B/E against the held-out test split (5,166 pieces,
via `$SCRATCH/MIDI-GPT/data/humanize_filtered_v2_testonly/` symlinking
`validation.parquet` -> `test.parquet`). Raw output under `$SCRATCH/MIDI-GPT/
humanize_eval/{e1,e2,e3,e4,e6,e8}_test_{a,b,e}/`, not yet archived to git (still landing).

- **E1**: replicates cleanly. G1b diff — A: test=1.003 [0.977,1.028] vs val=0.970;
  B: test=0.905 [0.881,0.928] vs val=0.890. Same ordering, same conclusion.
- **E4**: replicates cleanly. A flat-context degeneracy 0.677 (val 0.707), E 0.243
  (val 0.240) — the core "A collapses under bad context, E doesn't" finding holds.
  style_match_effect also consistent (A stronger than E in both splits).
- **E6**: replicates cleanly, robust pearson all >=0.92 for A/B/E, no checkpoint an
  outlier (test: A=0.981, B=0.973, E=0.926 vs val: A=0.960, B=0.966, E=0.945).
- **E3**: did NOT cleanly replicate at the default n=500 — and this is a metric-
  stability finding, not a checkpoint finding. At n=500 on the test split, GT's own
  bootstrap sign-stability for metrical_pos was 0.642 (A) and 0.632 (B), both below the
  0.9 reliability bar (only E's, at 0.897, was close) — i.e. the "true" sign isn't
  determined by 500 pieces for ANY checkpoint here, same instability that hit E on
  validation. r2_ratio swung to 0.264 (A, fail) / 0.760 (B, pass) / 0.224 (E, fail) at
  n=500. GT R^2 itself is 0.0034-0.0040 here vs 0.00033 in E's 2000-piece validation
  rerun -- a >10x swing that's a hallmark of R^2 noise when the true relationship is
  this close to zero. **Reran all three at n=2000 to match E's now-validated
  methodology (jobs 19371097/98/99, in progress)** rather than report the unstable
  n=500 numbers. Takeaway so far regardless of outcome: E3's gate needs >=2000 pieces
  to be trustworthy; the historical A-E matrix comparison (500 pieces each) should be
  read with that caveat.

## 2026-08-09 — MAJOR FINDING: track_by_track was masking severe collapse in A/B

Item 4 (add `degeneracy_rate_by_step` to E8's `track_by_track`, previously
Wasserstein-only) landed, and it overturns part of the earlier read of that scenario.
Validation-set rerun (`results/e8_full_{A,b,c,d,e}_rerun/e8_summary.json`), degeneracy
rate by step (0-5):

| Ckpt | step0 | step1 | step2 | step3 | step4 | step5 |
|---|---|---|---|---|---|---|
| A | 0.80 | 0.79 | 0.80 | 0.90 | 0.86 | 0.78 |
| B | 0.78 | 0.69 | 0.69 | 0.76 | 0.70 | 0.65 |
| C | 0.20 | 0.21 | 0.23 | 0.23 | 0.14 | 0.22 |
| D | 0.18 | 0.24 | 0.29 | 0.32 | 0.32 | 0.22 |
| E | 0.24 | 0.22 | 0.13 | 0.10 | 0.18 | 0.13 |

**This replicates cleanly on the held-out test split** (`results/e8_test_{a,b,e}/
e8_summary.json`): A=0.72-0.88, B=0.73-0.82, E=0.11-0.23. Not noise, not a
validation-set artifact.

A and B's Wasserstein-to-GT in `track_by_track` looked fine (recovering to ~15) — but
that was masking the fact that 65-90% of their generated output at every step is
**degenerate** (near-zero variance / collapsed). A collapsed output that happens to
land near GT's central tendency can still score a low Wasserstein distance; Wasserstein
alone can't tell "precise" apart from "collapsed to something GT-adjacent." C/D/E,
despite worse absolute Wasserstein, are actually producing real non-degenerate
variation 70-90% of the time. In hindsight this is consistent with E4's flat-context
degeneracy numbers (A=0.707, B=0.560) — `track_by_track` starts fully mechanical and
stays mostly non-real context throughout, i.e. it's a sustained version of exactly the
"unreliable context" stress test A/B are already known to fail. We just never measured
degeneracy specifically for this scenario before now.

**This reopens two calls from the 2026-08-08 Decision section**, not yet re-decided:
- C's `track_by_track` degeneracy (0.14-0.23) is competitive with E and clearly better
  than A/B — undercuts "C has no niche where it's the best choice."
- B's `track_by_track` degeneracy (0.65-0.82) is nearly as bad as A's — undercuts "B =
  best generalist" specifically for the iterative/track-by-track deployment pattern,
  even though B still looks strong on every other axis (E1, E2, E4, E6, E8 single-shot
  Wasserstein).

Net effect: track_by_track was probably the single most realistic deployment scenario
in the whole suite (mirrors how a user would actually build up a multi-track piece),
and on the metric that actually matters there (does it collapse), the mechanical-context
checkpoints are the clear winners, not A/B. Needs a discussion, not a unilateral
rewrite of the keep/drop call.

## 2026-08-09 — E3 at n=2000: still not a stable differentiator

Ran E3 at n=2000 on the test split too (`results/e3_test_{a,b,e}_2k/e3_summary.json`),
expecting the val-set fix (n=500->2000 resolved E's failure cleanly) to generalize.
It didn't cleanly: A r2_ratio=0.339 (fail), B r2_ratio=0.141 (fail, badly), E
r2_ratio=0.514 (barely pass). GT R^2 at n=2000 ranges 0.0014-0.0062 across these three
runs -- still bouncing around by ~4x depending on exactly which 2000 pieces get
sampled. Conclusion: this isn't a sample-size problem that a bigger N straightforwardly
fixes in the n=500-2000 range -- the real underlying relationship (velocity ~
metrical_pos + pitch_rank_in_onset + top_voice + interval_from_prev) is just weak
enough in this corpus that R^2-ratio-based gating doesn't reliably discriminate
checkpoints. Recommend treating E3's gate as a weak/unreliable signal for ranking
checkpoints against each other going forward, rather than continuing to chase larger N.
The absolute-scale finding (GT R^2 is near-zero everywhere) is itself a real, useful
result -- just not one that cleanly ranks A vs B vs E.

## Next steps (in progress, started 2026-08-08)

1. E7 controls grid for checkpoints B and E (currently only A has audio/listening output).
2. E6 per-instrument-group breakdown for checkpoint E — root-cause the 0.770 pearson
   (worst of all five, worse than A).
3. Firm up the E3/E gate failure — larger held-out sample or bootstrap CI on the sign
   estimate itself, so "noise floor" is demonstrated, not just asserted.
4. Add per-step degeneracy rate to E8 `track_by_track` (currently Wasserstein-only).
5. Confirmatory run on the untouched test split (`humanize_filtered_v2/test.parquet`,
   5,166 rows) for A, B, E once 1-4 land.

## 2026-08-09 — A/E logit-mixture blend: mechanism, detector, and track_by_track all validated

Motivation: given the final verdict (E recommended for robustness, A precise only under
trustworthy context), both checkpoints are tiny (4.1M params) — cheap enough to run
side by side and blend their predictive distributions per request instead of picking
one. `scripts/humanize_eval/blend_model.py`'s `BlendedModel` wraps both GPT2LMHeadModel
checkpoints, duck-typing the same `forward/make_empty_kv/kv_null_positions/max_context`
surface `InferenceEngine` expects so it drops in with zero production-code changes.
Blends in **log-probability space** (`logsumexp`), i.e. a true mixture
`p = alpha*softmax(logits_A) + (1-alpha)*softmax(logits_E)`, not raw logit averaging
(which would be a product-of-experts, a different and less interpretable operation).
Verified exact at the alpha=0/1 boundaries against a unit test before any eval ran.

### E9 — does blending beat either checkpoint alone? (`e9_blend_ae_probe.py`, n=200)

Same window/target/mechanization pattern scored 3 ways (A_alone/E_alone/Blend), alpha
set from **oracle** ground truth (`1 - mechanized_fraction`, known exactly since the
harness itself mechanizes the tracks) on E8's `mixed_ctx` scenario (mean mechanized
fraction 0.77 across the sample):

| Condition | Degeneracy | Wasserstein |
|---|---|---|
| A_alone | 0.392 | **15.04** |
| E_alone | **0.116** | 16.20 |
| Blend | 0.141 | **15.78** |

Blend nearly matches E's collapse-resistance while beating E on precision — a real,
controllable middle ground, not a broken compromise. Bookend sanity checks (`alpha`
forced to 0/1 on `mechanical_ctx`/`whole_track`) confirm Blend tracks E_alone/A_alone
respectively, as it must by construction. Archived:
`results/e9_blend_ae/e9_summary.json`.

**Important scope note**: E9's alpha is oracle (ground truth), not estimated — it
validates the blending *mechanism*, not whether alpha can be *estimated* from a real,
unlabeled piece. That's a separate question, answered next.

### E9b — does the deployable heuristic detector estimate alpha well? (`e9b_blend_auto_probe.py`, n=200)

`humanize_server.py`'s `mode=auto` can't see oracle mechanization — it estimates
per-track "mechanicalness" from note data (fraction of notes exactly on
`mechanize.py`'s own onset-quantization grid, reusing that function so the detector
can't drift from the definition of "mechanical" the model was trained against). This
run computes both oracle and heuristic-estimated alpha per piece on the same `mixed_ctx`
construction, adding a 4th condition (Auto_Blend) alongside A_alone/E_alone/Oracle_Blend:

- Detector accuracy: `mean_abs_alpha_error = 0.026`, `pearson_r(oracle, estimated) = 0.959`
  (mean oracle alpha 0.206, mean estimated 0.180) — the heuristic is close.
- Auto_Blend nearly matches Oracle_Blend: degeneracy 0.125 vs 0.10, Wasserstein 15.19 vs
  15.07 — both still far ahead of A_alone (0.375 / 15.87) and E_alone (0.14 / 16.29).

**Conclusion: `mode=auto` is validated, not just the blend mechanism.** Flipped
`auto_detector_validated: True` in `humanize_server.py`'s `/info` response. Archived:
`results/e9b_blend_auto/e9b_summary.json`.

### E9c — does blending help on `track_by_track`, the scenario that mattered most? (`e9c_blend_track_by_track_probe.py`, n=60)

Three full independent iterative trajectories per piece (A_alone/E_alone/Blend), same
starting mechanization and track order, diverging as each model generates different
content. Blend's alpha per step `i` = `i / (len(order)-1)` (oracle: step 0 = fully
mechanical context so alpha=0; last step = everything else already real so alpha=1).

| Step | A degeneracy | E degeneracy | Blend degeneracy | A Wass | E Wass | Blend Wass |
|---|---|---|---|---|---|---|
| 0 | 0.82 | 0.13 | 0.13 | 17.7 | 18.5 | **16.7** |
| 1 | 0.70 | 0.27 | 0.22 | 17.8 | 19.2 | 19.4 |
| 2 | 0.91 | 0.23 | 0.18 | 16.0 | 17.8 | 16.7 |
| 3 | 0.93 | **0.14** | 0.36 | 14.8 | 17.4 | 16.1 |
| 4 | 0.89 | **0.05** | 0.11 | 14.1 | 16.9 | 16.9 |
| 5 | 1.00 | **0.00** | **0.00** | 12.0 | 13.7 | **12.1** |

A stays catastrophic throughout (70-100% degenerate, confirming the original
`track_by_track` finding). Blend tracks close to E's low degeneracy at 5 of 6 steps and
is competitive-or-best on Wasserstein at nearly every step (beats both alone at steps 0
and 5) — but spikes to 0.36 at step 3, worse than E's 0.14 there (still far better than
A's 0.93). At n=60 (vs. 200 elsewhere) this could be a per-step sampling-noise blip, not
confirmed either way — worth a bigger rerun before treating it as fully settled, but the
overall pattern (blend ≈ E's robustness + better precision than either alone) replicates
from E9's single-shot result. Archived: `results/e9c_blend_tbt/e9c_summary.json`.

### Net effect

Both the blend mechanism (E9) and the actual deployable auto-detector (E9b) are
validated, and the win holds up on iterative use (E9c) with one open caveat (step-3, small
n). `midigpt-humanize-http` (new dedicated server, see below) ships all four modes:
`robust` (E alone, default), `expressive` (A alone), `blend` (explicit alpha),
`auto` (heuristic alpha, now validated).

## 2026-08-09 — Contrastive style-encoder pretraining: gate passed

Style-conditioning prototype (`scripts/style_prototype/`, separate track from the
Humanize checkpoint matrix — see the plan doc for full design). Variant B (InfoNCE
contrastive pretraining of the style encoder, LM-independent) ran to completion:
2000 steps, loss dropped from 2.92 (already below the `log(64)=4.159` chance level) to
~1.0-1.5 by the end, clearly discriminating same-piece segments from other pieces.
Clears the plan's own cheapest go/no-go gate. Checkpoint saved:
`runs/style_encoder_contrastive/style_encoder_contrastive.pt`. Unblocks Variant A
(joint-conditioning), next up.

(First attempt at this run, job 19371866, TIMED OUT at 4h without finishing — stuck
entirely in a one-time dataset valid-indices cache build that a matching
min_bars/min_tracks/encoder-config should have hit an existing cache for and didn't;
root cause not fully resolved, worth revisiting if it recurs. Retried with a 10h budget,
completed in 4h19m.)

## 2026-08-09/10 — Dedicated Humanize HTTP server

New `midigpt-humanize-http` entry point (`src/python/midigpt/http/humanize_server.py`),
separate from the existing generic `midigpt-http`. Same request/response JSON shape as
the generic server (`score` = `Score.to_dict()`, response = `{"score": ..., "timing":
...}`) — deliberately, so a caller talking to both needs only one representation for the
piece. The actual differences are `targets` (track/bar-level prompt, simpler than
building a full `TrackPrompt` list) and dual-checkpoint `checkpoint_mode`/`alpha`
selection (robust/expressive/blend/auto, see above).

**Final request schema** (settled after several rounds of scoping — deliberately not a
full `TrackPrompt`/`InferenceConfig` mirror; excluded fields and why are documented
inline in `GenParams`'s docstring):

```json
{
  "score": {...},
  "targets": [
    {"track": 0, "bars": [4, 5, 6, 7], "mechanize_before": false},
    {"track": 2, "mechanize_before": true}
  ],
  "checkpoint_mode": "blend", "alpha": 0.7,
  "temperature": 0.9, "top_p": 0.95, "top_k": 0,
  "seed": 42, "max_attempts": 3, "novelty_check": true,
  "bars_per_step": 4, "tracks_per_step": 1, "model_dim": 16
}
```

- `targets[].mechanize_before` — actually transforms that track's notes (constant
  velocity, grid-snapped onsets) before generation runs, real transformation not a hint.
  Applied once up front to the whole track; harmless on any bars that are also targets
  (Humanize always overwrites a target bar's velocity/timing regardless of its starting
  state) — the only real effect is that the track reads as mechanical context wherever
  it appears before its own target bars are reached, which is intentional.
- Explicitly excluded and why: `mask_bars`/`attributes`/`controls`/`autoregressive` (no
  masking or attribute/genre controls in use yet; AR generation is a different task, out
  of scope for a Humanize-specific server); `mask_mode`/`silence_check` (nothing is ever
  masked or can go silent — Humanize never changes note presence); `polyphony_hard_limit`
  /`density_hard_limit` (polyphony/density are fully determined by the fixed skeleton,
  nothing for these constraints to act on); `shuffle` (no ordering ambiguity in a single
  call); piece-level `genre` control (checked directly: `humanize_encoder.json`'s
  `token_domains` has zero Genre-type entries and none of the 5 training configs set
  `genre_probability` — `genre_groups` metadata existing in the JSON is a red herring,
  not something these checkpoints ever learned to use).
- `novelty_check` kept (default `true`, unlike everything else dropped) — a real,
  observed failure mode: confirmed firing on a sparse-bar test case (model regenerated
  input-identical values 3x, correctly raised past `max_attempts` instead of silently
  returning a no-op, which is what happened before this defaulted off).

Handles pieces longer than the model's bar window via sequential chunking with
backward-shifting final windows to keep `model_dim` valid (see `_iter_windows`'s
docstring for the exact bug this fixes — a naive shrink-to-fit last window produced
invalid sizes). End-to-end tested: partial track/bar targeting leaves untargeted content
provably unchanged, `mechanize_before` verified to flatten exactly the intended track
and nothing else, invalid params (`checkpoint_mode`, `top_p`, `max_attempts`, `model_dim`)
rejected with 400/422s, multi-window splicing correct, mechanize-only requests (no
generation at all) work as a fast path.

## 2026-08-20 — E10 sampling-mismatch note (follow-up needed)

E10 was run with `temperature=0.9, top_p=1.0` — outside E2's calibrated passing region
for the degeneracy gate (G2), which requires `tau>=1.0` with `top_p=0.95` (see E2 above).
E10's results are therefore not directly comparable to the E2/E3/etc. gated pass/fail
table as-is.

**Follow-up (not yet done):** rerun E10 with checkpoint A, using sampling params inside
E2's calibrated region, for a fair comparison.

## 2026-08-20 — Style-conditioning prototype, metric 1 (held_out_conditional_loss.py)

First real eval signal for the style-conditioning plan's three variants (A: random-init
StyleEncoder, A+B: StyleEncoder pretrained via Variant B's contrastive run then frozen,
C: explicit hand-designed controls). Checkpoints trained on a 10GB MIG slice with
`batch_size=16` (fp32, no mixed precision — reduced from the default 32 after that OOM'd
the slice; see the SLURM job history for context), only 20000/20000 steps (~7.7 epochs on
the 41,513-row train set) — same tiny architecture as checkpoint E (`n_embd=256,
n_layer=4, n_head=4`) but 1/10th checkpoint E's step budget (200000), so these are
undertrained checkpoints, not final ones. Training loss was still slowly declining at
step 20000, not flat.

Gate: `true` reference/controls must clearly beat both a `mismatched` piece's and `none`
(win-rate mean + bootstrap CI clearly above 0.5). Run: `n_scored=300` held-out pieces
each, `held_out_conditional_loss.py`, job 20194243 (bundled CPU eval, ~10 min total).

| Variant | mean CE true | mean CE mismatched | mean CE none | win vs. mismatched (95% CI) | win vs. none (95% CI) | Gate |
|---|---|---|---|---|---|---|
| A (random-init) | 1.050 | 1.255 | 1.979 | 0.827 [0.783, 0.867] | 0.987 [0.973, 0.997] | **PASS** |
| A+B (pretrained/frozen style enc) | 1.094 | 1.203 | 1.836 | 0.833 [0.790, 0.873] | 0.987 [0.973, 0.997] | **PASS** |
| C (explicit controls) | 1.131 | 1.231 | 1.676 | 0.703 [0.653, 0.753] | 0.990 [0.977, 1.000] | **PASS** |

All three pass even undertrained — the conditioning mechanism itself works (the model
demonstrably prefers the true reference/controls over a mismatched or absent one). C is
weakest on win-vs-mismatched (0.70 vs. ~0.83 for A/A+B), consistent with its signal being
4 quantized scalars vs. a learned encoder's richer representation. A and A+B are close on
both win-rates, so pretraining the style encoder (Variant B) hasn't yet shown a clear
edge over random-init at this step count.

Archived: `humanize_eval/style_metric1_A_random/metric1_summary.json`,
`humanize_eval/style_metric1_A_pretrained/metric1_summary.json`,
`humanize_eval/style_metric1_C_explicit/metric1_summary.json`.

**Open:** a matched-batch-size (20GB slice, `batch_size=32`) rerun is in progress (jobs
20183823/24/25, still queued as of this entry) — pending vs. the results above to see if
the smaller batch size (16 vs. 32) has any material effect independent of step count.

## 2026-08-20 — Style-conditioning prototype, metric 2 (style_transfer_faithfulness.py)

New script this session: `scripts/style_prototype/eval/style_transfer_faithfulness.py` +
shared `scripts/humanize_eval/beat_profile.py` (per-beat-position velocity/microtiming
profile, factored out for reuse with E0/E4 per EXPERIMENT_PLAN.md's "share this" note, not
yet wired into e4_context_probes.py itself to avoid touching already-validated E4 code).
Resamples every VelocityLevel/DeltaDirection/Delta position in a held-out target piece's
true sequence in one teacher-forced forward pass conditioned on a reference piece's z
(real scope simplifications documented in the script's own docstring: no target-bar mask
exists, so this is whole-sequence, not autoregressive -- see docstring for full reasoning),
decodes to a Score, and compares the generated per-beat profile to the reference's own
whole-piece profile via cosine similarity, true-z vs. a shuffled/mismatched-z control.

Run: job 20195624 (bundled CPU, ~2 min total), `n=100` held-out pieces, both Variant A
checkpoints from the metric-1 entry above.

| Variant | sim true (vel) | sim shuffled (vel) | CI of diff (vel) | sim true (timing) | sim shuffled (timing) | CI of diff (timing) |
|---|---|---|---|---|---|---|
| A (random-init) | 0.9941 | 0.9943 | [-0.0009, 0.0005] | 0.7105 | 0.7000 | [-0.0078, 0.0296] |
| A+B (pretrained) | 0.9952 | 0.9951 | [-0.0007, 0.0007] | 0.6902 | 0.6842 | [-0.0185, 0.0335] |

**Gate FAILS for both variants, both profile types** (CI includes zero every time; win-rate
vs. shuffled control ~0.33-0.49, i.e. no better than chance). Conditioning on the true
reference's z does not make the generated output's per-beat velocity/timing shape more
similar to that reference's own performance than a wrong z would.

**Caveat on the velocity result specifically:** `mean_sim` sits at ~0.99 for BOTH true and
shuffled conditions -- cosine similarity between almost any two pieces' 12-bin velocity
profiles is close to 1.0 (most performances share a broadly similar per-beat velocity
shape, e.g. downbeat emphasis), so the metric may have very little discriminative headroom
for velocity specifically, independent of whether the model is doing anything real. The
timing-profile result has much more spread (0.68-0.71) and thus more room to show an
effect, and does show a small positive-but-not-significant trend (true > shuffled on
average for both variants) -- suggestive, not conclusive.

**Read together with metric 1 (passed) and metric 2 (failed):** the model demonstrably
uses the presence of a reference/z signal to improve next-token prediction (metric 1), but
that does not yet show up as the SPECIFIC reference's style being reproduced in aggregate
output statistics more than a wrong reference would (metric 2). This is a real, structural
finding, not a bug in either eval script -- consistent with the plan's own framing that
metric 1 alone is a weak sanity check, not proof of genuine style transfer.

Archived: `humanize_eval/style_metric2_A_random/metric2_summary.json`,
`humanize_eval/style_metric2_A_pretrained/metric2_summary.json`.

## 2026-08-20 — Style-conditioning prototype, metric 3 (probe_z.py)

New script this session: `scripts/style_prototype/eval/probe_z.py`. Held-out R^2 (80/20
split, 20 repeats, `numpy.linalg.lstsq` -- sklearn confirmed unavailable in venv-humanize)
of `z -> nomml` and `z -> velocity_range` (both computed on the reference piece's own
decoded Score, see script docstring for the whole-piece-vs-reference-segment scope
simplification). Job 20195796, `n=300` held-out pieces, both Variant A checkpoints.

| Variant | nomml R^2 | velocity_range R^2 |
|---|---|---|
| A (random-init) | 0.491 (+/-0.107) | 0.057 (+/-0.088) |
| A+B (pretrained) | 0.273 (+/-0.130) | -0.115 (+/-0.223) |

**z is not degenerate/collapsed** -- it carries real, non-trivial held-out-predictable
signal about nomml (median metric depth / quantization looseness), especially for the
random-init encoder (R^2=0.49). Velocity range shows no reliable signal either way (R^2
near/below zero for both).

**Synthesis across metrics 1-3:** the StyleEncoder extracts real signal (metric 3), and the
LM's next-token loss is sensitive to that specific signal beyond just "a reference exists"
(metric 1's true-vs-mismatched CI clearly excludes 0.5) -- but that doesn't yet manifest as
the generated CONTINUATION's aggregate per-beat profile resembling the true reference more
than a wrong one (metric 2 fails). Rules out "collapsed/degenerate encoder" as the
explanation for metric 2's null result; leaves undertraining (20k steps vs. checkpoint E's
200k at the same architecture) and/or metric 2's own scope simplifications (one-shot
non-autoregressive resampling, coarse 12-bin profile, velocity's near-ceiling similarity
floor) as the live candidates.

Archived: `humanize_eval/style_metric3_A_random/metric3_summary.json`,
`humanize_eval/style_metric3_A_pretrained/metric3_summary.json`.

## 2026-08-20 — Style-conditioning prototype, metric 4 (steering_strength_sweep.py)

New script this session: `scripts/style_prototype/eval/steering_strength_sweep.py`, sharing
`resample_expressive_from_logits` (factored out of `style_transfer_faithfulness.py` into
`steered_forward.py`) so metric 2's and metric 4's resampling protocol is identical -- only
the injection mechanism differs (soft-prefix/`ConditionedGPT2` vs. activation steering on
the unconditioned base checkpoint `humanize_tiny_e-20260807-223945`). Job 20201980, CPU,
`n=100` held-out pieces, alpha in {0, 0.25, 0.5, 1, 2, 4, 8}, all 4 transformer blocks
steered, run against the same 20k-step Variant A checkpoints as metrics 1-3.

Velocity faithfulness diff-CI (true z minus shuffled z) at every alpha, both checkpoints:

| alpha | A (random) vel CI | A (random) win-rate | A+B (pretrained) vel CI | A+B (pretrained) win-rate |
|---|---|---|---|---|
| 0.0 | [-0.0008, 0.0188] | 0.44 | [-0.0025, 0.0073] | 0.55 |
| 0.25 | [-0.0134, 0.0033] | 0.49 | [-0.0034, 0.0059] | 0.53 |
| 0.5 | [-0.0048, 0.0031] | 0.47 | [-0.0060, -0.0005] | 0.41 |
| 1.0 | [-0.0012, 0.0065] | 0.48 | [-0.0008, 0.0043] | 0.58 |
| 2.0 | [-0.0019, 0.0025] | 0.52 | [-0.0096, 0.0026] | 0.51 |
| 4.0 | [-0.0002, 0.0064] | 0.53 | [-0.0003, 0.0060] | 0.56 |
| 8.0 | [-0.0016, 0.0013] | 0.56 | [-0.0000, 0.0037] | 0.57 |

Timing faithfulness CIs are similarly noisy and straddle zero at nearly every alpha for
both checkpoints (one exception: A/random at alpha=0.5 shows CI [0.012, 0.070], i.e.
excludes zero -- but this is 1 of 14 (checkpoint x alpha) cells and not stable at
neighboring alphas, so read as sampling noise rather than a real effect until it replicates).

**Gate FAILS at every alpha, both checkpoints** -- steering shows no better faithfulness to
the reference's own profile than a shuffled/wrong reference, matching metric 2's soft-prefix
null result. Degeneracy proxy (`velocity_std`) drops monotonically with alpha for both
checkpoints (random: 0.170->0.115; pretrained: 0.166->0.122 from alpha=0 to alpha=8) with
no corresponding faithfulness gain -- i.e. large steering strengths start flattening the
output distribution (the failure mode the plan calls out) while buying nothing.

**Read together with metrics 1-3:** two independent conditioning mechanisms (soft-prefix and
activation steering) both fail to reproduce reference-specific style in the generated
output's aggregate profile, despite the encoder carrying real signal (metric 3) that the LM
demonstrably uses to reduce loss (metric 1). This strengthens the undertraining hypothesis
(same null result across mechanisms is more consistent with "the signal path from z to
expressive token choice hasn't been trained hard enough to sharpen" than with a bug specific
to one mechanism) but does not rule out metric 2/4's shared scope simplifications
(one-shot non-autoregressive resampling, coarse 12-bin profile) as a contributing measurement
ceiling. These are the **20k-step baseline / lower-bound** results -- 60k-step retraining
(jobs 20196080/20196081, queued, not yet started as of this run) is expected to raise the
LM's overall utilization of z (per checkpoint E's ~200k-step precedent) and should be
compared against this baseline once it lands.

Archived: `humanize_eval/style_metric4_A_random_20k/metric4_summary.json`,
`humanize_eval/style_metric4_A_pretrained_20k/metric4_summary.json`.

## 2026-08-21 — CORRECTION: metrics 1/2/4 were scoring the wrong tokens; all
## re-run at full scale with the fix. Supersedes the 2026-08-20 entries above.

**Root cause.** `humanize_probability=1.0` training/eval data encodes each piece as
in-place context bars (real or mechanized VelocityLevel/DeltaDirection/Delta tokens the
model is trained to reproduce as a FIXED value, never z-conditioned) followed by a
separate `HumanizeStart...HumanizeEnd` appendix (the target notes' velocity/timing --
the ONLY region z was ever trained to influence). Confirmed via the C++ encoder
(`encoder.cpp`) and `TokenType` enum (`HumanizeStart=89`/`HumanizeEnd=90`, checkable via
`vocab.get_type`).

All three metrics were scanning for VelocityLevel/DeltaDirection/Delta tokens across the
WHOLE sequence instead of just the appendix:
- **Metrics 2/4** (`resample_expressive_from_logits`) resampled and profiled in-place
  context-bar tokens too, diluting any real signal in the small appendix region with
  noise from positions z was never trained to vary. Per-beat profile comparisons also
  used the WHOLE decoded piece instead of the target/reference bars specifically.
- **Metric 1** (`expr_ce`) was worse: it reused `dataset.py`'s `_expressive_mask`, which
  *categorically excludes* everything at/after `TrackEnd` (i.e. all appendix content) by
  design, for its own different purpose. So metric 1 could never score the actual
  humanize target at all -- it was scoring in-place context-bar CE the whole time.

**Fix.** New shared `appendix_expressive_mask()` (`steered_forward.py`) walks the token
stream tracking a `HumanizeStart`/`HumanizeEnd` boolean, used by both the resampler and
metric 1's `expr_ce`. `MidiGPTDataset` now also returns `humanize_bars`/`reference_bars`
(additive -- confirmed the training collators pick keys by name, so this doesn't touch
the training path), used to scope metric 2/4's profile comparisons to the real bars.

**Full-scale re-run** (jobs 20218378/20218385, `n=300` metric 1/3, `n=100` metric 2/4),
against the original 10GB/20k checkpoints AND the 20GB/20k (2x-examples) checkpoints:

Metric 1 (win-rate of true beating mismatched/none on CE, appendix-only):

| checkpoint | true vs mismatched | true vs none | gate |
|---|---|---|---|
| 10GB random | 0.547 [0.493,0.60] | 0.610 [0.557,0.667] | FAIL (mismatched CI touches 0.5) |
| 10GB pretrained | 0.563 [0.507,0.62] | 0.567 [0.51,0.623] | **PASS** (barely) |
| 20GB random | 0.570 [0.517,0.627] | 0.497 [0.44,0.553] | FAIL (none CI includes/below 0.5) |
| 20GB pretrained | 0.577 [0.523,0.633] | 0.460 [0.403,0.517] | FAIL (worse than chance vs. none) |

True-vs-mismatched is weakly but consistently above chance across all 4 checkpoints
(0.547-0.577) -- some real reference-specific signal survives appendix-only scoring.
True-vs-none is inconsistent, sometimes below chance -- having the correct reference is
not reliably better than having no reference at all, on the tokens that actually matter.
Only 1 of 4 checkpoints clears the original hard-gate bar. **This overturns the
2026-08-20 metric 1 entry's clean PASS for both variants** -- that result was almost
entirely driven by in-place context-bar CE, not appendix CE.

Metric 2 (profile faithfulness, target/reference-bar-scoped):

| checkpoint | velocity gate | timing gate |
|---|---|---|
| 10GB random | FAIL (ci=[-0.0059,0.0019]) | FAIL (ci=[-0.0007,0.0556]) |
| 10GB pretrained | FAIL (ci=[-0.0044,0.0008]) | FAIL (ci=[-0.0430,0.0628]) |
| 20GB random | FAIL (ci=[-0.0025,0.0008]) | FAIL (ci=[-0.0303,0.0523]) |
| 20GB pretrained | FAIL (ci=[-0.0039,0.0016]) | **PASS** (ci=[0.0088,0.0761]) |

Still mostly null even correctly scoped -- 1 of 8 (checkpoint x profile-type) cells
passes. A small-sample (n=20) smoke test during development showed a misleadingly
strong pass; does not replicate at full n=100. Treat that smoke result as noise, not a
finding.

Metric 3 (probe_z): unaffected by this bug (was already correctly scoped to
`reference_bars`), results replicate closely: nomml R^2 0.475/0.270 (10GB
random/pretrained), 0.417/0.210 (20GB) -- z remains non-degenerate.

Metric 4 (steering): still fails at every alpha, both checkpoint pairs, both
before and after the fix -- consistent null, not an artifact of the scoping bug.

**Revised synthesis:** with coherent scoping, the evidence for genuine reference-specific
style transfer is much weaker than originally reported. The encoder extracts real,
non-degenerate signal (metric 3, unaffected). The LM shows a weak, fairly consistent
"true beats mismatched" edge (metric 1, ~0.55-0.58 across all 4 checkpoints) but no
reliable "true beats no-reference" edge, and neither soft-prefix nor steering produces
output whose per-beat profile is detectably closer to the true reference than a wrong one
(metrics 2/4, mostly null even at 2x examples). The 60k-step retrain (jobs
20196080/20196081, running as of this entry, ~1h20m into a 6h budget) is the next real
test of the undertraining hypothesis -- but given metric 1's weak-not-zero true-vs-
mismatched signal survives 2x more examples with no visible improvement (10GB->20GB
win-rates: 0.547->0.570, 0.563->0.577 -- flat, not trending up), undertraining alone may
not be the whole explanation; worth treating the 60k result as a real test rather than an
assumed win.

Archived: `humanize_eval/style_metric{1,2,3,4}_A_{random,pretrained}_{10gb,20gb}_fixed/*.json`.

## 2026-08-21 — Non-controllable follow-up items closed out (E6, E10)

**E6 item resolved (was: "root-cause checkpoint E's worst-of-five 0.770 pearson").**
The data already existed in `results/e6_full_e/e6_summary.json` -- `e6_per_instrument.py`
already computes a size-filtered "robust" pearson (groups with >=10 pieces only,
alongside the raw all-groups figure), added at some point but never written up here.
Checkpoint E, n=800 pieces, 44 groups reported: raw pearson=0.770, **robust
pearson=0.945** (21 groups with >=10 pieces). Root cause confirmed: E's apparent
homogenization weakness was almost entirely a handful of noisy small-instrument-group
means (3-9 pieces) dragging down an unweighted correlation across 44 groups, not a real
per-instrument collapse. Cross-group variance ratio (gen/GT) = 1.056 -- E preserves real
cross-instrument dynamic differences essentially exactly. No further action needed; E6
item closed.

**E10 rerun with calibrated sampling params for checkpoint A (job 20220705, running).**
Checkpoint A's own E2 gate passes at `bps=single|tau=1.0|top_p=1.0` (not E's
`tau=1.0|top_p=0.95` -- the two checkpoints calibrate to slightly different points), so
reran E10 at `--temperature 1.0 --top-p 1.0` (previous run used the script's default
temperature=0.9, outside A's passing region) via `humanize_eval/e10_listening_a_calibrated/`.
Results to follow once the job completes.

**Also answered inline (not a new run, existing E2 data):** "does the generated
distribution match ground truth for velocity/micro-timing" is exactly what E2 measures.
Checkpoint E at its calibrated point (tau=1.0, top_p=0.95, n=150): velocity Wasserstein
5.42 (vs. 18.87 flat-mechanical baseline, 25.04 shuffled-donor baseline), delta
(micro-timing) Wasserstein 0.77, degeneracy rate 6.7% (GT's own baseline: 6.0%,
excess only 0.7pp), generated dispersion 5.94 [4.94,7.01] overlaps GT's 7.13
[6.88,7.39]. Gate G2 PASSES. Checkpoint A at its own calibrated point (tau=1.0,
top_p=1.0, n=150): velocity Wasserstein 5.61 (vs. 18.00/23.37 baselines), delta
Wasserstein 0.80, degeneracy 14.7% (GT baseline 10.7%, excess 4.0pp -- higher than E's
but still gated pass), dispersion 6.17 [5.08,7.43] overlaps GT's. Gate G2 PASSES.

**E10 (job 20220705, calibrated tau=1.0/top_p=1.0) results**: 10 pieces, real MIDI
before/after/reference triples written to `humanize_eval/e10_listening_a_calibrated/`.
Informal (not gated), but the per-piece expressive-token stats are worth a look before
calling this done: at this calibrated point, checkpoint A's generated velocity_std and
onset-deviation are consistently much smaller than the real reference performance's own
(e.g. piece00: gen velocity_std=1.58 vs ref=6.19, gen onset_dev=25.6 ticks vs ref=132.3;
piece01: gen std=1.55 vs ref=10.27, gen onset_dev=19.8 vs ref=110.4) -- qualitatively,
individual pieces look much less dynamically varied than a real human performance, even
though E2's aggregate Wasserstein/dispersion-overlap gate passes. Not a contradiction --
E2 compares distributions pooled across many pieces, this is per-piece -- but worth a
human actually listening to `humanize_eval/e10_listening_a_calibrated/*.mid` before
calling checkpoint A's expressiveness fully validated.

## 2026-08-21 — Variant C (explicit controls) extended to metrics 2/4

New `--variant {A,C}` flag on `eval/style_transfer_faithfulness.py` and
`eval/steering_strength_sweep.py`, plus `load_steering_source_c`/
`compute_steer_vector_c`/`_collect_usable_pieces_c` in `steered_forward.py` --
Variant C (ExplicitControlsGPT2) had only ever been evaluated via metric 1; metrics 2/4
were hard-wired to Variant A's ConditionedGPT2/z_proj. Both funnel through the same
appendix-scoped `resample_expressive_from_logits` and bar-scoped `piece_profiles` as
Variant A, so numbers are directly comparable. Full-scale runs (jobs 20220912/20220913,
n=100, both 10GB and 20GB explicit-controls checkpoints):

| checkpoint | metric 2 velocity gate | metric 2 timing gate | metric 4 (best alpha) |
|---|---|---|---|
| 10GB | FAIL (ci=[-0.0042,0.0001]) | FAIL (ci=[-0.0384,0.0660]) | FAIL at every alpha |
| 20GB | FAIL (ci=[-0.0032,-0.0002]) | FAIL (ci=[-0.0052,0.0304]) | FAIL at every alpha |

Same null pattern as Variant A -- Variant C's hand-designed, interpretable control
prefix doesn't fare any better than the learned StyleEncoder at metrics 2/4, despite
different mechanisms. Some metric 4 win-rates for Variant C are notably below chance
(20GB, alpha=0.5/1.0 timing: win_rate=0.04) -- worth a second look if Variant C is
pursued further, though CIs still cross zero at n=100 so not yet a confirmed reversal.

Archived: `humanize_eval/style_metric{2,4}_C_{10gb,20gb}/*.json`.

## 2026-08-21 — Real autoregressive steering eval (bypasses the one-shot resampler)

New `scripts/style_prototype/eval/steering_autoregressive.py`: routes activation
steering through REAL `InferenceEngine`/`SamplingSession` generation (KV-cached,
grammar-constrained, genuine note-to-note compounding) instead of the one-shot static
resampler metrics 2/4 use -- addresses the "does resampling's lack of compounding
matter" open question from earlier this session. Mechanism: `InferenceEngine.__init__`
accepts an already-built model object directly; built `SteeredGPT2LMHeadModel`
wrapping a plain loaded `GPT2LMHeadModel` whose `forward()` delegates to
`steered_forward.py`'s existing `steered_forward()` (same per-block `alpha*steer_vec`
injection) -- `_KVRunner.forward()` calls the model via pure duck typing with zero
`isinstance` checks, so the wrapper is a legal drop-in with zero changes to production
`session.py`/`engine.py`/`gpt2.py`. Two independent piece pools feed the comparison:
`MidiGPTDataset`-based pieces (reference-side style_ids, matching metric 2/4) and raw
parquet + `pick_window_and_targets` (target-side, needs a real `Score` for
`InferenceEngine.session()`, which `MidiGPTDataset` doesn't expose) -- same
real-generation pattern as `e6_per_instrument.py`. Currently Variant-A-only (reuses
z_proj/StyleEncoder steering-source loading; no Variant C path yet).

Smoke test (n=15, alphas 0/1, CPU): alpha=0 gives exactly 0.0 diff across all pairs
(correct sanity check -- true/shuf share a seed and alpha=0 makes steering inert, so
outputs are bit-identical). alpha=1.0: mean=-0.0025, CI=[-0.0068,0.0015] (crosses
zero), consistent with the one-shot resampler's null, though n=7 pairs is far too small
to be evidence either way. Real full-scale run (job 20221196, n=50, alphas
0/1/2/4/8, 10GB random-init checkpoint) submitted -- results to follow. Real generation
is materially slower than the one-shot resampler (multiple seconds/generation), so this
run uses n=50/5 alphas rather than metric 2/4's usual n=100/7 alphas.

**Full-scale real-generation results (jobs 20221196/20221424, n=50, both 10GB
checkpoints):** velocity faithfulness gate FAILS at every alpha for both checkpoints
(all CIs cross zero, win-rates 0.36-0.59, no trend with alpha). Timing is noisy across
14 (checkpoint x alpha) cells; one cell (pretrained, alpha=1.0) shows CI [-0.301,-0.061]
excluding zero on the NEGATIVE side (win_rate=0.17, well below chance) -- read as an
outlier among many noisy cells, not a stable effect, until it replicates at a
neighboring alpha.

**This is an important confirmation, not just a repeat**: steering's null result from
the one-shot resampler (metric 4, `steering_strength_sweep.py`) now REPLICATES under
genuine autoregressive generation (real KV-cache, grammar-constrained, real
note-to-note compounding dependencies). Rules out "the one-shot resampler's lack of
compounding was masking a real effect" as an explanation for metric 4's null -- the
null is real, not a measurement artifact of that specific simplification.

## 2026-08-21 — Flexible Humanize HTTP server (autoregressive + humanize, per-bar
## expressive/quantized control, per-request checkpoint incl. conditioning variants,
## parquet MIDI retrieval)

Built per user request + approved plan (`/home/triana24/.claude/plans/understand-how-the-expressive-declarative-cloud.md`).
New standalone script (not a pyproject.toml entry point, prototype scope):
`scripts/style_prototype/humanize_style_server.py`, plus supporting modules
`engine_cache.py`, `parquet_retrieval.py`, `prefix_conditioned_gpt2.py`, and a
refactor of `steered_forward.py` (moved `SteeredGPT2LMHeadModel`/`build_steered_engine`
in from `eval/steering_autoregressive.py` so both share one definition).

**Core capability**: per track, choose `context`/`autoregressive`/`humanize`; per
bar (not just per track, extending the existing `humanize_server.py`'s track-only
`mechanize_before`), force expressive-or-quantized context via `mechanize_bars`/
`expressive_bars`/`mechanize_before` (reuses `mechanize.py`'s already-bar-scoped
`mechanize_bar` directly -- no core code changes needed there). Fixed one real bug
found by reading `humanize_server.py`'s `_humanize_window`: its `bar.notes` gate
(line 290) is correct for humanize (needs an existing skeleton) but wrong for AR (an
empty bar is the normal AR target) -- the new server's window-target resolver gates on
`bar.notes` only for `mode="humanize"`.

**Checkpoint selection is per-request** (not fixed at server startup like both existing
servers), via `EngineCache` -- an LRU keyed at the base-checkpoint level, nesting
steered sub-entries so evicting a base evicts everything built on it. Three engine
shapes: plain, steered (activation steering on a base checkpoint), soft-prefix (real
KV-cache-capable conditioned model, see below).

**New architecture work (the plan's flagged high-risk item): real, incremental,
KV-cache-capable soft-prefix generation.** `ConditionedGPT2`/`ExplicitControlsGPT2`
were single-shot, full-sequence, no-KV-cache models ("not meant to be dropped into
InferenceEngine as-is" per their own docstrings) -- `prefix_conditioned_gpt2.py`'s
`PrefixConditionedGPT2` wraps either one to implement the exact `ModelBase` protocol
(confirmed by reading `inference/base.py` + `GPT2LMHeadModel`'s concrete shape this
session): injects the computed prefix as an extra KV position 0 on the first
(prefill) call only, with every subsequent call behaving like ordinary incremental
GPT2 (the prefix is already baked into `past_kv`'s length by then). `max_context()`
returns `n_positions - 1` (position 0 permanently reserved); `kv_null_positions`
shifts spans by +1 for the same reason.

**Verification (this was NOT just smoke-tested, it was numerically confirmed
correct):** direct logit/KV inspection before trusting any generation output --
confirmed the prefix vector genuinely differs between conditioning states (diff
norm 2.76, cosine similarity ~0 between "real reference" and "no reference"
prefixes), confirmed this reaches the output logits (diff norm 27-36 on a 755-dim
vector, top-5 candidates completely reshuffle), confirmed the KV cache grows to
exactly `T_real + 1` positions after prefill. Both variants A and C checked this way
before any full-server testing.

**Full integration smoke test (15/15 passed, real checkpoints, via FastAPI
TestClient)**: plain humanize, plain full-track AR, humanize+per-bar-mechanize mix
(confirmed mechanized bar's velocity is exactly constant 80), both error cases
(mechanize/expressive_bars overlap -> 422, bad checkpoint path -> 400, context-mode-
with-bars -> 422), steering conditioning, soft-prefix conditioning (both variant A
and C, including C via both `control_bins` and derived `reference`), parquet
retrieval (both `score` and `midi` formats), path-traversal rejection, out-of-range
index rejection. **Plus the plan's explicitly flagged highest-uncertainty item**:
multi-window AR on a real 78-bar piece (5 windows at 16 bars each) -- passed, bar
count preserved, no crash. Also confirmed the real deployment path (not just
TestClient): booted via `python3 humanize_style_server.py --data-root ... --port
...` and hit with real `curl` requests (`/health`, `/parquet/info`), both correct.

12 fast unit tests added (`tests/test_humanize_style_server.py`, no checkpoint
needed): `_iter_windows`/window-target bar.notes gating (humanize vs AR), mechanize-
set precedence, `_TargetSpec` pydantic validation, `ParquetIndex` row-group math
across multiple row groups, path-traversal rejection. All pass.

**Status: all three of the plan's explicitly flagged risks are now resolved, not
just carried forward** -- soft-prefix KV-cache correctness (numerically verified),
multi-window AR (smoke-tested on a real 78-bar piece), and steering-vs-training-path
steer-vector equivalence (not yet cross-checked numerically against the eval
scripts' own derivation -- still open, lower stakes than the other two since
steering's server-side wiring reuses the exact same `steered_forward.py` functions
the eval scripts already use, just with a differently-derived reference token
sequence).
