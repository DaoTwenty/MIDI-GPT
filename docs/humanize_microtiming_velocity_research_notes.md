# Humanize Microtiming & Velocity Encoding: Research Notes

**Status:** working notes, not published documentation. Written as raw material for a paper — captures motivation, theory, method, results, and open questions from an investigation that ran informally (no lab notebook) across several sessions. Numbers below are pulled from actual instrumented runs on the GigaMIDI-derived corpus, not estimates.

**Scope:** the Humanize feature's microtiming (Delta/DeltaDirection tokens) and velocity encoding — a bug in the encoding that made it lossy, an investigation into whether the corpus's timing data reflects genuine human performance, and a redesigned data-quality filter for selecting training data.

---

## 1. Background and motivation

### 1.1 What Humanize does

Humanize is a generation mode in this symbolic-music model: given a piece with fixed pitch/rhythm/duration content, it regenerates only the *expressive* attributes — velocity (dynamics) and microtiming (small deviations from the metrical grid) — for chosen bars, via a soft skeleton-in-place + appendix mechanism. The intent is to let a model take a mechanically-quantized or skeleton performance and make it "sound played" by a human, or to transfer a performance style onto new material.

This only works if the *data* the model trains on actually contains genuine expressive signal. If the training corpus's "microtiming" is mostly quantization noise, or an artifact of how the MIDI was produced (rather than a human playing against a tempo), the model will learn to reproduce that artifact instead of real expressiveness. That concern — is the signal we're modeling real? — is the throughline of this document.

### 1.2 The tokenization scheme

The tokenizer (`src/cpp/tokenizer/`, exposed via pybind11 as `midigpt._core`) represents note timing at two nested resolutions:

- **Pos tokens**: `resolution` (config field, `12` for Humanize) discrete positions per quarter note — the coarse metrical grid a note's onset snaps to.
- **Delta / DeltaDirection tokens**: a *microtiming residual* layered on top of the Pos grid. Each Pos-cell is further subdivided into `resolution` (again `12`) sub-steps, so the finest grid the encoding can represent is `resolution × resolution = 144` steps per quarter note (≈ 3.47 ms at a 120 BPM reference tempo). `DeltaDirection` is a binary sign token; `Delta` is the unsigned magnitude.
- **VelocityLevel tokens**: a quantized (or, after this work, unquantized) representation of MIDI velocity (0–127).

Two resamplings happen around this representation: raw MIDI ticks (native resolution, typically 480 ticks/quarter as read by `MidiReader`) are resampled *down* to the 12/144 grid for encoding (`Tokenizer.normalize_input`), and after decoding, tokens are resampled *up* to `decode_resolution` (1920 ticks/quarter for Humanize) for output (`Tokenizer.normalize_output`). Both directions go through the same function, `resample_delta()` (`src/python/midigpt/tokenizer/tokenizer.py`).

### 1.3 Corpus

Training data is derived from GigaMIDI. Each row carries a `NOMML` (Number Of Music Meters/something similar — a GigaMIDI-provided per-track metadata field) value; `NOMML == 12` is GigaMIDI's own signal for "freely timed" (not already quantized to a grid) and is the starting eligibility gate for Humanize training data. `music_style_scraped` carries scraped genre labels (frequently `"unknown"`).

---

## 2. Problem 1: the Delta encoding was lossy (a real bug, not a modeling limitation)

### 2.1 Symptom

`DeltaDirection` fired on **0% of notes** in the corpus — i.e. the sign token was structurally dead code. The Delta magnitude histogram was **bimodal with 35.7% of notes saturated at the maximum bin** (a clamping wall), and the measured onset reconstruction error (round-tripping real pieces through encode→decode) was **19.1 ms at p90** — far worse than the grid resolution should allow.

### 2.2 Root cause

`resample_delta()`'s old implementation truncated the continuous target position (`int(true_pos)`) rather than rounding to nearest. Truncation always produces a **non-negative** residual, so:

- `DeltaDirection` (which only fires when the residual is negative) could never fire.
- The residual magnitude was systematically inflated (up to a full grid cell, instead of at most half a cell for round-to-nearest), which is what produced the saturation/clamping.

### 2.3 Fix

This algorithm was specified directly by the project owner (not designed independently), and matches standard lossless-quantization practice: quantize to the *finest* representable grid first (the only place information should be discarded), then re-derive the coarser (Pos-token) position and the exact leftover as an *exact integer* residual — no further rounding loss beyond the one initial quantization step.

```python
def resample_delta(score, source_res, target_res, use_delta):
    ...
    scale = target_res / source_res
    micro_per_cell = target_res  # Delta's own unit: 1/target_res of one target_res cell
    for note in ...:
        true_pos = (note.onset_ticks + note.delta / source_res) * scale
        micro = round(true_pos * micro_per_cell)          # round to the finest grid
        new_onset = max(0, round(micro / micro_per_cell))  # nearest Pos-cell
        note.onset_ticks = new_onset
        note.delta = micro - new_onset * micro_per_cell    # exact integer signed residual
```

A companion bug was found and fixed while implementing this: the `Delta` token's vocabulary domain size was `resolution / 2` (magnitude range 0–5), but a signed round-to-nearest residual can reach *exactly* half a Pos-cell at the tie point — magnitude 6, not 5. The domain was one short of its own worst case. Fixed to `resolution / 2 + 1` (0–6) in `src/cpp/tokenizer/encoder_config.cpp`. This raised the total vocabulary size for `humanize_encoder.json` from 720 to 732 tokens — meaning **any checkpoint trained before this fix is incompatible** and must be retrained.

At the same time, velocity encoding for Humanize specifically was changed from a 32-level quantization to the full 128-level raw MIDI velocity range (`models/humanize_encoder.json`: `"velocity_levels": 32 → 128`), following an existing precedent already used by `expressive_encoder.json` elsewhere in the codebase. Rationale: Humanize's whole purpose is reproducing fine-grained dynamics; coarse quantization throws away exactly the signal being modeled.

### 2.4 Verification methodology

`scripts/humanize_eval/e0_reference_stats.py` round-trips every piece in a held-out validation split through the *real* production pipeline (`Tokenizer.normalize_input → Encoder.encode → Decoder.decode → Tokenizer.normalize_output`) and measures onset reconstruction error against the original, unquantized position, plus token marginal distributions (Delta histogram, DeltaDirection fire rate, VelocityLevel marginal). It also computes an *analytic* counterfactual — the theoretical best case for a correct signed round-to-nearest scheme — without needing a rebuild, as an independent ceiling to compare the real pipeline against.

### 2.5 A second bug, found while verifying the fix (worth including as method, not just result)

The first post-fix measurement showed p90 = 4.86 ms — better than 19.1 ms, but not matching the 1.74 ms analytic ceiling, leaving an unexplained ~3 ms gap. Rather than accept "close enough," this was tracked down: it turned out to be a bug in the *measurement script*, not the encoding. `encoder.cpp` re-sorts chord notes (multiple notes sharing one onset) into ascending-pitch order before emitting tokens, but the eval script compared original vs. decoded notes via `zip(original_order, decoded_order)` using the *original* (unsorted) note list — so for any chord, it silently paired note A's original position against note B's decoded position and reported their onset difference as "error." ~28% of note-pairs in the validation set were affected this way (worst individual case: 0.87 grid cells ≈ 36 ms, between two completely unrelated notes).

Fix: `canonical_note_order()` — group notes by onset tick, sort each group by pitch, apply to *both* the original and decoded note lists before pairing — added to `e0_reference_stats.py`.

### 2.6 Results (full 4,815-piece validation split, 833,144 real note-pairs, post both fixes)

| Metric | Pre-fix | Post-fix (corrected measurement) |
|---|---|---|
| p90 onset reconstruction error | 19.1 ms | **1.736 ms** |
| mean onset reconstruction error | 5.95 ms | **0.865 ms** |
| `DeltaDirection` fire rate | 0.0% | **31.0%** |
| Delta post-clamp histogram | bimodal, 35.7% saturated at max bin | clean monotonic decay, 0.99% at new max bin |
| Delta domain clamp hits | — | **0 / 220,249** nonzero deltas |

The corrected real-pipeline figure (1.7361111 ms) matches the analytic ceiling to floating-point precision. This is the expected, unavoidable floor: with a 144-step-per-quarter grid, worst-case round-to-nearest error is half a step = 1/288 quarter = 500/288 ≈ 1.736 ms at the 120 BPM reference tempo used for the ms conversion. **The encoding is now exactly at its theoretical limit** — no further gain is possible without a finer grid (a real vocab-size tradeoff, not a bug fix), and no bug remains to chase.

All 16 Python `resample_delta`/tokenizer unit tests and all 8 C++ ctest suites pass unchanged under the fix (existing fixtures' specific numeric examples happened to produce identical results under both the old and new algorithms, since their residuals were small enough not to distinguish truncation from rounding).

---

## 3. Problem 2: is the corpus's microtiming *real*?

Fixing the encoding only guarantees the model can faithfully learn whatever timing signal is *in* the data. It says nothing about whether that signal reflects genuine human performance (played against a tempo/metronome, producing natural, continuously-varying deviations from the grid) versus an artifact of how a given MIDI file was produced (e.g. quantized-then-shifted by a transcription or arrangement tool, which can *look* like microtiming without containing any real expressive information).

### 3.1 Is `DeltaDirection`'s sign imbalance a bug?

Measured directly on 220,249 nonzero deltas: **65.3% positive / 34.7% negative** — not the ~50/50 a purely symmetric, zero-mean jitter process would produce. Two implementation explanations were tested and ruled out before concluding this is a property of the data:

1. **Domain clamping** — 0/220,249 deltas exceed the ±6 domain; nothing is silently clipped.
2. **A sign-flip bug in the C++ round-trip** — read through `encoder.cpp`'s `DeltaDirection`/`Delta` emission and `decoder.cpp`'s `delta_direction`/`delta_total` accumulation; both reset correctly per-note and round-trip correctly.

**The decisive test was a random-MIDI null-model control**, run because code-reading rules out specific bugs but doesn't positively prove the *system* is unbiased. 2,000,000 notes were generated with onsets uniformly random over a bar at a source resolution 100× finer than the Delta grid (i.e. no organic structure of any kind — pure uniform noise), and fed through the *exact same* `resample_delta()`. Result:

```
signed delta histogram:  -6: 4.15%   -5..+5: ~8.33% each (flat)   +6: 4.18%
sign split among nonzero: 49.99% positive / 50.01% negative
```

This exactly matches the closed-form prediction for round-to-nearest quantization of uniform noise onto a 12-residue-class grid (flat across residues, with the round-half-to-even tie class splitting evenly across the ±6 boundary bins). **The tokenizer is provably unbiased**: fed symmetric input, it returns symmetric output. The 65/35 skew in the real corpus is therefore a genuine data property, not a code defect.

### 3.2 Characterizing the real corpus's Delta distribution

The real signed pre-clamp histogram (full corpus, 4.1M notes) is neither uniform nor normal:

```
delta:  -6    -5    -4    -3    -2     -1     0     1     2     3     4     5     6
pct:   0.89  2.56  3.30  3.00 13.89   7.36  16.24 14.47 16.76  9.59  8.33  2.63  0.99
```

A purely organic (symmetric-jitter) distribution would be smooth, unimodal, and monotonically decaying away from 0. Instead there are non-monotonic local maxima at **−2** (13.89%, above both neighbors) and **+2** (16.76%, the global maximum, above both neighbors) — the signature of a mixture: an organic near-zero component, plus a second population sitting at a roughly fixed offset from the grid (~2/144 quarter ≈ 6.9 ms at 120 BPM). The +2 bump outweighing −2 is what drags the aggregate sign split to 65/35.

### 3.3 First hypothesis: is the "mode 2" population swing, or a resampling artifact?

Two benign explanations were tested and ruled out:

- **Swing** (systematically pushing the "and" of the beat late) predicts a strong *phase-dependent* asymmetry — upbeat-eighth notes should show a very different mean delta than downbeat-eighth notes. Measured across the 934 "mode2-heavy" pieces (≥50% of a piece's notes falling in the anomalous band): mean phase gap ≈ 0. **Ruled out.**
- **Pure resampling arithmetic** (the `480 → 144` downsampling itself creating an artifact independent of any real timing) was tested by recovering each file's *true* native tick resolution directly from the raw MIDI header bytes (`MThd` chunk, bypassing `MidiReader`'s canonicalization to 480 — which is a fixed target, not evidence about source diversity) and building a deterministic "zero real jitter" null model for comparison. All 500 sampled files were natively 480 PPQ, and the null model does **not** match the real histogram's concentration (L1 distance 59.23 — real data is far more concentrated than pure arithmetic alone predicts). **Ruled out.**

This left the initial working hypothesis: the mode-2 population (19.4% of the validation set) reflects a *systematic, non-organic* bias — e.g. audio-to-MIDI transcription latency — based on tight per-piece standard deviation (~0.7 grid-units) within the anomalous band, in isolation.

### 3.4 Revised conclusion: it's probably real "laid-back"/"pushing" performance, not an artifact

That "tight stdev" conclusion was reached without a baseline for comparison, and turned out to be wrong. Re-tested by comparing the mode-2-heavy population against the *rest of the corpus* using the production filter's own validated timing-variance metric (`std_ratio` — see §4), plus, per a live suggestion, velocity organicness:

| | mode2-heavy (n=934) | rest (n=3,881) |
|---|---|---|
| `std_ratio` (real note-to-note timing variance) | 0.274 | 0.296 |
| `conc_score` (metrical grid alignment) | 0.724 | 0.742 |
| velocity stdev | **12.45** | 9.65 |
| velocity flat-rate | **17.5%** | 35.4% |

The two groups are statistically indistinguishable on real timing variance (0% of either group fails the production std-ratio threshold), and — strikingly — the mode2-heavy group shows **more** organic velocity than the rest of the corpus, not less. An artifact wouldn't also happen to carry richer dynamics. This is much more consistent with **genuine performers who consistently play a bit ahead of or behind the beat** — a well-documented real phenomenon in music performance ("in the pocket," "laid-back," "pushing") — than with a mechanical transcription bias.

**Methodological note for the paper**: this reversal is itself a useful result. The earlier conclusion (non-organic) was reached from a single statistic evaluated in isolation, without a corpus baseline; only cross-checking against an established metric and an independent signal (velocity) exposed the error. The filter design in §4 was built directly to avoid repeating this mistake — by requiring comparison against the corpus's own natural distribution, and using manual example inspection to sanity-check any conclusion drawn from a summary statistic.

---

## 4. Designing a principled data-quality filter

### 4.1 The existing production filter

`scripts/humanize_data_quality/filter_expressive_tracks.py` already existed (predates this investigation) and gates Humanize training eligibility per track on:

1. `NOMML == 12` (GigaMIDI's own "freely timed" signal)
2. **Concentration score** `= 1 − H(onset-phase bins)/log(12) ≥ 0.6` — rejects tracks whose coarse onset-phase histogram looks closer to uniform noise than to a real 12-slot metrical grid (drift/misalignment detector).
3. (Original) **`std_ratio ≥ 0.15`** — within the dominant coarse phase bin, the *fine* sub-grid residual's standard deviation (normalized by half the coarse cell width) must show real note-to-note variance, not be pinned to a constant — intended to reject "quantized + constant tick offset" artifacts.

This filter is what produced the `humanize_filtered` dataset used throughout §2–3. Its threshold (0.15) had not been empirically validated.

### 4.2 Why a single hand-picked threshold is the wrong approach

Rather than adjust the 0.15 number by guesswork, the corpus-wide *shape* of several candidate timing-organicness metrics was examined directly, on the already-filtered 4,815-piece corpus, looking for a natural bimodal split (a data-driven threshold, rather than an arbitrary one):

- `std_ratio` (existing metric)
- **fine-residual entropy ratio**: Shannon entropy of the fine within-dominant-bin residual values, normalized by the max possible entropy for that many discrete tick slots (`log2(coarse_unit)`) — designed to catch a failure mode `std_ratio` can miss: a mechanical process alternating between a *small number of discrete* offsets (nonzero std, but still zero real continuous variability).
- **mode-concentration**: the fraction of a track's dominant-bin notes sharing the single most common *exact* fine residual tick value.

**All three were smooth, unimodal, continuously-decaying distributions across the whole corpus — no valley, no second cluster, anywhere.** This is itself an important, honest negative finding: there is no clean two-population ("real" vs. "fake") structure to threshold at, for timing. Organicness varies on a genuine continuum across this heterogeneous, many-sourced corpus.

That said, the *extreme tail* was directly, manually inspected (not just statistically inferred) and found to contain unambiguous artifacts:

```
row=596: residual ticks = {4: 159, -16: 4}    -- 159/163 notes at the exact same tick
row=305: residual ticks = {0: 153, -20: 4}    -- 153/157 notes at the exact same tick
row=290: residual ticks = {-16: 275, 4: 14}   -- 275/289 notes at the exact same tick
```

versus a typical mid-corpus piece:

```
row=1129: {-5: 719, 5: 439, 0: 305, -10: 182, 10: 142, -15: 42, ...}
```

Hundreds of notes landing on the *literal same* one-of-~40-possible tick value is not something continuous human timing variability produces at any realistic sample size — it is the fingerprint of quantize-then-shift. So: reject only this extreme, individually-verified tail (`mode_frac ≥ 0.90`), and treat the score as a continuous quality signal everywhere else, rather than pretending a clean binary split exists.

### 4.3 Velocity is different — a real threshold *is* justified

The same shape analysis applied to **velocity mode-concentration** (max single velocity value / note count) showed something qualitatively different from timing: a genuine gap.

```
p90: 0.70   p95: 0.95   p99: 1.00
mode_frac >= 0.999 (perfectly flat velocity):   4.26% of corpus (205 pieces)
mode_frac in [0.95, 0.999)  (near-empty gap):   0.73% of corpus (35 pieces)
```

Pieces are either spread out like the bulk of the corpus, or sitting in a distinct cluster pinned almost exactly at 1.0 (every note the literal same velocity — dynamics never authored at all). This makes sense: whether a MIDI author bothered to program dynamics at all is a much more binary choice than continuous timing precision ever is. **Reject `velocity_mode_frac ≥ 0.95`** — targeting a confirmed, separable population, not an arbitrary cut through a continuum.

### 4.4 Final filter design

| # | Check | Threshold | Basis |
|---|---|---|---|
| 1 | `NOMML == 12` | — | GigaMIDI-provided |
| 2 | `conc_score` (metrical alignment) | `≥ 0.6` | unchanged from original filter |
| 3 | `timing_mode_frac` (dominant-bin fine-residual concentration) | `< 0.90` | replaces `std_ratio ≥ 0.15`; targets the individually-verified degenerate tail only |
| 4 | `velocity_mode_frac` (whole-track velocity concentration) | `< 0.95` | new; targets the confirmed flat-dynamics population |

Implemented in an updated `filter_expressive_tracks.py`, computing both the *old* (`std_ratio`) and *new* criteria in the same single pass over the corpus, so the two filters' pass/fail decisions could be directly compared without re-scanning.

### 4.5 Full-corpus before/after comparison

Run via SLURM (`MIDI-GPT-infra/slurm/filter_humanize_data.sh`, 12 CPUs, 12.3 min wall time) over the complete `v2.0.0` corpus: 1,922,594 rows total, 610,878 no-drums rows, 811,457 `NOMML==12` candidate tracks.

| | OLD filter | NEW filter |
|---|---|---|
| rows eligible | 43,943 | **47,033** (+7.0%) |
| tracks eligible | 101,564 | **114,840** (+13.1%) |

Track-level churn, out of 657,320 scored tracks:

| Group | Count | % | Interpretation |
|---|---|---|---|
| pass both | 95,100 | 14.5% | stable core |
| pass NEW only | 19,740 | 3.0% | 19,725 of these have `std_ratio < 0.15` — recovered false negatives from the old, unjustified threshold |
| pass OLD only | 6,464 | 1.0% | 5,611 rejected purely for flat velocity (a check that didn't exist before); 775 rejected purely for timing degeneracy `std_ratio` missed but `mode_frac` caught (discrete-multi-state artifacts); 78 fail both |
| pass neither | 536,016 | 81.6% | fail `conc_score` (metrical alignment) regardless — untouched by this change |

The net effect is not merely "more permissive" — the new filter does real, different, better-justified work in *both* directions: it recovers data wrongly excluded by an arbitrary threshold, and it catches a real gap (flat-velocity tracks) the old filter never checked for at all, plus a small set of timing artifacts the variance-only test structurally couldn't see.

### 4.6 Rebuilt corpus and final verification

`humanize_filtered_v2/` (41,859 train / 5,174 validation rows, vs. the prior 39,128 / 4,815) was materialized from the new pass list and re-verified with `e0_reference_stats.py`:

| Metric | v1 (old filter) | v2 (new filter) | Expected direction |
|---|---|---|---|
| p90 onset reconstruction error | 1.736 ms | 1.736 ms | **unchanged** — filter doesn't touch encoding, correctly |
| `DeltaDirection` fire rate | 31.0% | 31.2% | **~unchanged** — filter deliberately does not touch the mode-2 population (§3.4 showed it's real, not artifact) |
| per-piece velocity stdev (mean) | 10.2 | **11.1** | **improved** — flat-velocity tracks removed |
| per-piece velocity entropy (mean, bits, max 7.0) | 3.52 | **3.74** | **improved** |

All four move in exactly the direction the design predicts, which is itself a form of validation: the filter changed what it was supposed to change (velocity organicness) and left alone what it was supposed to leave alone (the reconstruction ceiling, and the legitimate "plays behind/ahead the beat" population).

---

## 5. Training

With both fixes and the new filtered corpus in place, a tiny Humanize checkpoint was launched (`MIDI-GPT-infra/slurm/train_humanize_tiny.sh`, job `19210474`):

- Model: `n_embd=256, n_layer=4, n_head=4` (tiny), `humanize_encoder.json` (732-token vocab, picked up automatically — `vocab_size` is computed from the tokenizer at train time, not hardcoded in the model config).
- Data: `humanize_filtered_v2` (§4.6).
- Hardware: single H100 MIG `2g.20gb` slice, 12h walltime.
- A previous checkpoint (`humanize_tiny-20260804-124407`) had trained against the pre-fix, 720-token encoding and stale filter — confirmed no longer running/queued (no conflict), effectively superseded/abandoned by this work.

---

## 6. Summary of contributions (for framing the paper)

1. **A lossless microtiming encoding fix.** Diagnosed and fixed a truncation bug that made a symmetric-residual quantization scheme silently one-sided, killing an entire token type (`DeltaDirection`) and inflating reconstruction error by ~11× (19.1 ms → 1.736 ms p90, exactly matching the theoretical floor of a 144-steps-per-quarter grid). Included finding and fixing a companion off-by-one vocabulary domain bug, and a chord-note-ordering bug in the *evaluation methodology itself* that had been masking the true post-fix result.
2. **A null-model methodology for auditing tokenizer bias.** Rather than trust code review alone, validated that the fixed encoding is provably unbiased by feeding it synthetic uniform-random input and confirming the output matches closed-form predictions exactly (flat histogram, 49.99/50.01 sign split) — establishing that any asymmetry observed in real data is a data property, not an artifact of the encoding.
3. **A performance-authenticity investigation with an explicit self-correction.** Investigated whether the corpus's timing signal reflects real human performance, ruled out two artifact hypotheses (swing, resampling arithmetic) via targeted statistical tests, initially concluded a subpopulation was non-organic based on an unbaselined statistic, then caught and reversed that conclusion by comparing against the corpus's own distribution and an independent signal (velocity) — landing on a more defensible interpretation (genuine "laid-back"/"pushing" performers).
4. **A data-quality filter grounded in corpus-wide distributional shape rather than guessed thresholds.** Showed that timing organicness is a *continuum* in this corpus (no natural bimodal split across three different candidate metrics), while velocity authorship is closer to *binary* (a real, separable cluster of flat-dynamics tracks) — and designed differently-shaped filtering rules for each, validated against manually-inspected examples and a full 1.9M-row corpus scan with an explicit before/after comparison (+13.1% eligible tracks, with the churn attributed to specific, explainable mechanisms in both directions).

---

## 7. Open questions / possible future work

- The revised "laid-back performer" interpretation of the mode-2 population (§3.4) is well-evidenced but not proven via ground-truth provenance (no such metadata exists in this corpus). A stronger test would correlate mode-2-heavy status with instrument role (e.g. bass/drums, where "pocket" playing is a known convention) or with tempo (a genuine felt offset should scale with beat duration in a *proportional*, not fixed-millisecond, way — unlike a mechanical latency artifact).
- The continuous timing-organicness score (`std_ratio` / `fine_entropy_ratio` / `mode_frac` — they track together) is currently unused beyond the extreme-tail reject. It could be used as a soft per-piece training weight or curriculum signal rather than a binary filter, now that its shape is known.
- `filter_expressive_tracks.py` and `build_filtered_parquet.py` changes are implemented and verified but not yet committed to version control.
- Whether to also re-run this filter against the general (non-Humanize) corpus, or extend the velocity/timing organicness checks to other model variants, hasn't been explored.

---

## 8. Reference: scripts and files

**Production code (in `worktree-humanize`):**
- `src/python/midigpt/tokenizer/tokenizer.py` — `resample_delta()` fix
- `src/cpp/tokenizer/encoder_config.cpp` — Delta domain size fix
- `src/cpp/tokenizer/encoder.cpp` / `decoder.cpp` — DeltaDirection/Delta encode/decode (read, unmodified, verified correct)
- `src/cpp/io/midi_reader.h` / `.cpp` — native-tick canonicalization to 480 (read, unmodified)
- `models/humanize_encoder.json` — `velocity_levels: 32 → 128`
- `scripts/humanize_data_quality/filter_expressive_tracks.py` — data-quality filter, updated for §4
- `scripts/humanize_data_quality/build_filtered_parquet.py` — materializes filtered parquet, updated for new filter's output schema

**Evaluation / analysis scripts (`scripts/humanize_eval/`):**
- `e0_reference_stats.py` — reconstruction ceiling + token marginals (checked in, includes the `canonical_note_order()` fix)
- `e0b_delta_mode_investigation.py` — swing-vs-artifact phase test (checked in)
- `e0c_source_tpq_artifact_check.py` — source-tpq null-model check (checked in)
- `e0f`–`e0k` (random-MIDI control, joint timing/velocity validation, entropy metric, tail examples, mode-concentration shape, velocity shape) — ad hoc, not yet checked into the repo; recreate from the descriptions in §3–4 if needed for the paper's reproducibility appendix.

**Infra:**
- `MIDI-GPT-infra/slurm/filter_humanize_data.sh` — full-corpus filter scan
- `MIDI-GPT-infra/slurm/train_humanize_tiny.sh` — training launch (now supports `DATA_DIR` override)

**Data artifacts:**
- `$SCRATCH/MIDI-GPT/data/humanize_filtered_v2/{train,validation}.parquet` — current filtered corpus
- `$SCRATCH/MIDI-GPT/humanize_filter_results/` — per-shard + summary JSON from the full-corpus filter scan
- `$SCRATCH/MIDI-GPT/humanize_eval/e0_v2/e0_summary.json` — post-fix, post-refilter token statistics
