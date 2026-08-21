"""E1 -- held-out teacher-forced NLL with context ablations and "smart
guessing" baselines (EXPERIMENT_PLAN.md section E1, the hard gate).

Answers, for the trained Humanize checkpoint, on held-out pieces:

  1. Token-level NLL (via SamplingSession.score_from_tokens, no sampling):
     does the model beat a "uniform over whatever the grammar legally
     allows at that position" baseline, under (a) the real surrounding
     context, (b) context with velocity/delta flattened to a constant
     (no expressive information), (c) context with velocity/delta swapped
     in from a donor piece (same skeleton, different "performance")?
     Reported per token type (VelocityLevel / DeltaDirection / Delta),
     since they behave very differently and an aggregate hides that.

  2. Value-level "smart guessing" baselines (not just uniform-random):
     per target note, reconstruct the true velocity level (0-127) and
     signed delta (-6..6, 0 = none emitted) and compare top-1 accuracy of
       - marginal-mode  : always guess the corpus-wide most common value
       - lag-1 transition: guess argmax P(value | previous note's value),
         a table fit on the *training* split (not held-out, so it can't
         peek) -- "smart guessing" in the sense the user asked for: does
         the model beat something that just looks similar to the last
         token, not just a strawman uniform-random guess.
       - model (own argmax at the corresponding token position(s))
     Reported for round-tripped ground truth against these same baselines
     too (not just the model) -- if GT itself barely beats them, that's
     a data-signal finding, not a model failure (see conversation).

Gates (from EXPERIMENT_PLAN.md):
  G1a: model NLL (true context) must beat the corpus-marginal baseline
       decisively on both token types.
  G1b: true context must beat swapped context, paired per piece, CI
       excluding zero -- "is the model using context at all".

Simplifications versus the full plan (documented, not hidden): single
track, single humanized bar per piece (the last non-empty bar in a 4-bar
window, so it has maximal preceding context); one checkpoint. Delta
accuracy is scored only among notes where a DeltaDirection/Delta token was
actually emitted (delta != 0) -- whether the model correctly predicts "no
delta at all" is a separate, un-scored question here (see report caveats).

Usage:
    python3 e1_conditional_nll.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny-20260807-035822/model_final.safetensors \\
        --train-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/train.parquet \\
        --val-parquet   $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir    $SCRATCH/MIDI-GPT/humanize_eval/e1 \\
        [--limit N] [--train-limit N] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

VEL_LEVELS = 128
DELTA_LO, DELTA_HI = -6, 6
DELTA_DOMAIN = DELTA_HI - DELTA_LO + 1  # 13
LAPLACE_ALPHA = 1.0


def velocity_encode(v: int, num_levels: int = VEL_LEVELS) -> int:
    if v <= 0:
        return 0
    if v >= 127:
        return num_levels - 1
    return min(1 + v * (num_levels - 1) // 128, num_levels - 1)


def canonical_note_order(notes):
    by_onset: dict[int, list] = defaultdict(list)
    for n in notes:
        by_onset[n.onset_ticks].append(n)
    out = []
    for onset in sorted(by_onset):
        out.extend(sorted(by_onset[onset], key=lambda n: n.pitch))
    return out


# --------------------------------------------------------------------- #
#  Marginal + lag-1 transition tables, fit on the TRAIN split only
# --------------------------------------------------------------------- #

def fit_baseline_tables(train_parquet: str, limit: int | None):
    from midigpt._types import Score
    from midigpt.tokenizer.tokenizer import resample_delta

    vel_marginal: Counter = Counter()
    delta_marginal: Counter = Counter()
    vel_transition: dict[int, Counter] = defaultdict(Counter)
    delta_transition: dict[int, Counter] = defaultdict(Counter)

    pf = pq.ParquetFile(train_parquet)
    row_i = 0
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            row_i += 1
            if limit is not None and row_i > limit:
                break
            raw = bytes(music.as_py())
            try:
                score = Score.from_bytes(raw)
            except Exception:
                continue
            if not score.tracks:
                continue
            # Use the real resample_delta (not a hand-rolled approximation --
            # the true model-resolution delta depends on the coarse Pos-cell
            # rounding remainder too, not just the fine sub-tick residual;
            # an earlier version of this function got that wrong).
            resample_delta(score, score.resolution, 12, use_delta=True)
            for track in score.tracks:
                prev_vel = None
                prev_delta = None
                for bar in track.bars:
                    for note in canonical_note_order(bar.notes):
                        vel = velocity_encode(note.velocity)
                        d = max(DELTA_LO, min(DELTA_HI, note.delta))
                        vel_marginal[vel] += 1
                        delta_marginal[d] += 1
                        if prev_vel is not None:
                            vel_transition[prev_vel][vel] += 1
                        if prev_delta is not None:
                            delta_transition[prev_delta][d] += 1
                        prev_vel, prev_delta = vel, d
        if limit is not None and row_i > limit:
            break

    return {
        "vel_marginal": vel_marginal,
        "delta_marginal": delta_marginal,
        "vel_transition": vel_transition,
        "delta_transition": delta_transition,
    }


def marginal_logprob_and_argmax(marginal: Counter, domain: int, offset: int = 0):
    total = sum(marginal.values())
    argmax_val = max(marginal.items(), key=lambda kv: kv[1])[0] if marginal else offset

    def logprob(value: int) -> float:
        c = marginal.get(value, 0)
        return math.log((c + LAPLACE_ALPHA) / (total + LAPLACE_ALPHA * domain))

    return logprob, argmax_val


def transition_logprob_and_argmax(transition: dict, marginal_argmax: int, domain: int, lo: int = 0):
    def logprob(prev_value: int, value: int) -> float:
        row = transition.get(prev_value)
        if not row:
            return math.log(1.0 / domain)
        total = sum(row.values())
        c = row.get(value, 0)
        return math.log((c + LAPLACE_ALPHA) / (total + LAPLACE_ALPHA * domain))

    def argmax(prev_value: int) -> int:
        row = transition.get(prev_value)
        if not row:
            return marginal_argmax
        return max(row.items(), key=lambda kv: kv[1])[0]

    return logprob, argmax


# --------------------------------------------------------------------- #
#  Per-piece: build window, target bar, context perturbations
# --------------------------------------------------------------------- #

def pick_window_and_target(score, n_bars: int, rng: random.Random):
    from midigpt.augmentation.score_window import select_window

    window = select_window(score, n_bars, 1, min_fill_ratio=0.0)
    if window is None or not window.tracks:
        return None
    track = window.tracks[0]
    eligible = [b for b, bar in enumerate(track.bars) if bar.notes]
    if not eligible:
        return None
    target_bar = eligible[-1]
    if len(eligible) < 2 and target_bar == 0:
        return None  # no preceding context at all
    return window, target_bar


def flatten_context(window, target_bar: int, const_velocity: int = 80):
    from midigpt.augmentation.mechanize import mechanize_bar

    out = copy.deepcopy(window)
    for b, bar in enumerate(out.tracks[0].bars):
        if b == target_bar:
            continue
        mechanize_bar(bar, out.resolution, const_velocity)
    return out


def swap_context(window, target_bar: int, donor_notes: list, rng: random.Random):
    out = copy.deepcopy(window)
    if not donor_notes:
        return flatten_context(window, target_bar)
    i = rng.randrange(len(donor_notes))
    for b, bar in enumerate(out.tracks[0].bars):
        if b == target_bar:
            continue
        for note in canonical_note_order(bar.notes):
            donor = donor_notes[i % len(donor_notes)]
            note.velocity = donor.velocity
            note.delta = donor.delta
            i += 1
    return out


def flat_note_values(window):
    """[(bar_idx, vel_level, signed_delta)] across the whole window, canonical
    order. Resamples a deep copy to model resolution (12) via the real
    resample_delta -- signed delta is only correct after that resampling
    (it depends on the coarse-cell rounding remainder, not just the raw
    native sub-tick residual)."""
    from midigpt.tokenizer.tokenizer import resample_delta

    resampled = copy.deepcopy(window)
    resample_delta(resampled, resampled.resolution, 12, use_delta=True)
    out = []
    for b, bar in enumerate(resampled.tracks[0].bars):
        for note in canonical_note_order(bar.notes):
            vel = velocity_encode(note.velocity)
            d = max(DELTA_LO, min(DELTA_HI, note.delta))
            out.append((b, vel, d))
    return out


# --------------------------------------------------------------------- #
#  Score one condition (true / flattened / swapped) via score_from_tokens
# --------------------------------------------------------------------- #

def score_condition(engine, ctx_score, target_bar: int, true_appendix: list[int]):
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[target_bar], humanize=True)],
        config=InferenceConfig(bars_per_step=1, tracks_per_step=1, model_dim=4, mask_mode="remove"),
    )
    session = engine.session(ctx_score, req)
    return session.score_from_tokens(true_appendix, step_idx=0)


def extract_true_appendix(tok, window, target_bar: int):
    import midigpt._core as _core

    opts = _core.EncodeOptions()
    opts.multi_humanize = {(0, target_bar)}
    full = tok.encode(copy.deepcopy(window), opts=opts, compute_attributes=False)
    vocab = tok._vocab
    hstart = vocab.encode_val(_core.TokenType.HumanizeStart, 0)
    hend = vocab.encode_val(_core.TokenType.HumanizeEnd, 0)
    if hstart not in full or hend not in full:
        return None
    i0 = full.index(hstart)
    i1 = full.index(hend, i0 + 1)
    return full[i0 + 1 : i1 + 1]


def reconstruct_note_values_from_appendix(appendix: list[int], vocab):
    """[(vel_level, signed_delta)] per note, from a true (or scored) appendix
    token stream, using VelocityLevel as the per-note group-start marker."""
    import midigpt._core as _core

    notes: list[dict] = []
    cur = None
    for tid in appendix:
        try:
            tt, val = vocab.decode(int(tid))
            tt = tt.name
        except Exception:
            continue
        if tt == "VelocityLevel":
            if cur is not None:
                notes.append(cur)
            cur = {"vel": val, "sign": 1, "mag": 0, "has_delta": False}
        elif tt == "DeltaDirection" and cur is not None:
            cur["sign"] = -1
        elif tt == "Delta" and cur is not None:
            cur["mag"] = val
            cur["has_delta"] = True
    if cur is not None:
        notes.append(cur)
    return [(n["vel"], n["sign"] * n["mag"] if n["has_delta"] else 0) for n in notes]


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.engine import InferenceEngine

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting marginal + lag-1 transition baselines on the train split...", flush=True)
    tables = fit_baseline_tables(args.train_parquet, args.train_limit)
    vel_logp, vel_argmax = marginal_logprob_and_argmax(tables["vel_marginal"], VEL_LEVELS)
    delta_logp, delta_argmax = marginal_logprob_and_argmax(tables["delta_marginal"], DELTA_DOMAIN)
    vel_trans_logp, vel_trans_argmax = transition_logprob_and_argmax(tables["vel_transition"], vel_argmax, VEL_LEVELS)
    delta_trans_logp, delta_trans_argmax = transition_logprob_and_argmax(
        tables["delta_transition"], delta_argmax, DELTA_DOMAIN
    )
    print(f"  vel marginal argmax={vel_argmax}  delta marginal argmax={delta_argmax}", flush=True)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    tok = engine._tokenizer
    vocab = tok._vocab

    # Preload a pool of pieces for donor sampling (swapped-context condition).
    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
            if len(raw_pieces) >= max(args.limit * 2, 50):
                break
        if len(raw_pieces) >= max(args.limit * 2, 50):
            break
    rng.shuffle(raw_pieces)

    per_piece_records = []
    n_attempted = 0
    n_used = 0

    for raw in raw_pieces:
        if n_used >= args.limit:
            break
        n_attempted += 1
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue

        try:
            picked = pick_window_and_target(score, args.n_bars, rng)
        except Exception as exc:
            print(f"  [skip] window selection failed: {exc}", flush=True)
            continue
        if picked is None:
            continue
        window, target_bar = picked

        try:
            true_appendix = extract_true_appendix(tok, window, target_bar)
        except Exception as exc:
            print(f"  [skip] appendix extraction failed: {exc}", flush=True)
            continue
        if not true_appendix:
            continue

        # donor for swap: a different random piece from the pool
        donor_raw = raw_pieces[rng.randrange(len(raw_pieces))]
        try:
            donor_score = Score.from_bytes(donor_raw)
            donor_notes = [n for t in donor_score.tracks for b in t.bars for n in b.notes]
        except Exception:
            donor_notes = []

        flat_ctx = flatten_context(window, target_bar)
        swap_ctx = swap_context(window, target_bar, donor_notes, rng)

        try:
            res_true = score_condition(engine, window, target_bar, true_appendix)
            res_flat = score_condition(engine, flat_ctx, target_bar, true_appendix)
            res_swap = score_condition(engine, swap_ctx, target_bar, true_appendix)
        except Exception as exc:
            print(f"  [skip] scoring failed: {exc}", flush=True)
            continue
        if res_true["n_tokens"] == 0:
            continue

        # ---- value-level reconstruction for the target bar's notes ----
        window_values = flat_note_values(window)  # includes context + target
        target_note_values = [(v, d) for (b, v, d) in window_values if b == target_bar]
        # find the index in the flat window list where the target bar starts,
        # so we know each target note's "previous" note (context or earlier
        # target note) for the transition baseline.
        flat_all = window_values
        target_start = next(i for i, (b, _, _) in enumerate(flat_all) if b == target_bar)

        note_records = []
        for j, (b, vel, d) in enumerate(flat_all):
            if b != target_bar:
                continue
            global_i = j
            if global_i == 0:
                continue  # no previous note at all; skip (rare, window starts populated)
            prev_vel = flat_all[global_i - 1][1]
            prev_delta = flat_all[global_i - 1][2]
            note_records.append({
                "vel": vel, "delta": d,
                "marginal_vel_correct": vel_argmax == vel,
                "marginal_delta_correct": delta_argmax == d,
                "transition_vel_correct": vel_trans_argmax(prev_vel) == vel,
                "transition_delta_correct": delta_trans_argmax(prev_delta) == d,
                "marginal_vel_logp": vel_logp(vel),
                "marginal_delta_logp": delta_logp(d),
                "transition_vel_logp": vel_trans_logp(prev_vel, vel),
                "transition_delta_logp": delta_trans_logp(prev_delta, d),
            })

        # Uniform-baseline NLL per position is log(n_legal) (NLL of a uniform
        # guess over whatever the grammar mask actually allowed there) --
        # bucket by token type so it's comparable like-for-like against
        # nll_table, not mixed across types with very different domain sizes.
        uniform_nll_by_type: dict[str, list] = defaultdict(list)
        for tt, n in zip(res_true["token_types"], res_true["n_legal"]):
            if n > 0:
                uniform_nll_by_type[tt].append(math.log(n))

        rec = {
            "n_target_notes": len(target_note_values),
            "n_scored_notes": len(note_records),
            "true_context": {
                "total_logp": res_true["total_logp"], "n_tokens": res_true["n_tokens"],
                "per_type": res_true["per_type"],
                "uniform_nll_by_type": {k: sum(v) / len(v) for k, v in uniform_nll_by_type.items()},
                "mean_argmax_acc": (
                    sum(res_true["argmax_correct"]) / len(res_true["argmax_correct"])
                    if res_true["argmax_correct"] else None
                ),
                "argmax_acc_by_type": {
                    tt: sum(1 for t, c in zip(res_true["token_types"], res_true["argmax_correct"]) if t == tt and c)
                    / max(1, sum(1 for t in res_true["token_types"] if t == tt))
                    for tt in set(res_true["token_types"])
                },
            },
            "flattened_context": {
                "total_logp": res_flat["total_logp"], "n_tokens": res_flat["n_tokens"],
                "per_type": res_flat["per_type"],
            },
            "swapped_context": {
                "total_logp": res_swap["total_logp"], "n_tokens": res_swap["n_tokens"],
                "per_type": res_swap["per_type"],
            },
            "notes": note_records,
        }
        per_piece_records.append(rec)
        n_used += 1
        if n_used % 25 == 0:
            print(f"  scored {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done scoring: {n_used} pieces used out of {n_attempted} attempted.", flush=True)

    # ------------------------------------------------------------- #
    #  Aggregate
    # ------------------------------------------------------------- #
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    def per_type_mean_nll(records, cond_key, type_name):
        vals = []
        for r in records:
            pt = r[cond_key]["per_type"].get(type_name)
            if pt:
                vals.append(-pt["mean"])  # NLL = -logp
        return mean(vals)

    def bootstrap_ci(diffs, n_boot=2000, seed=0):
        if not diffs:
            return None
        rb = random.Random(seed)
        n = len(diffs)
        means = []
        for _ in range(n_boot):
            sample = [diffs[rb.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        lo = means[int(0.025 * n_boot)]
        hi = means[int(0.975 * n_boot)]
        return {"mean": sum(diffs) / n, "ci95_lo": lo, "ci95_hi": hi, "n_pieces": n}

    token_types_seen = set()
    for r in per_piece_records:
        token_types_seen.update(r["true_context"]["per_type"].keys())

    nll_table = {}
    for tt in sorted(token_types_seen):
        nll_table[tt] = {
            "model_true_context": per_type_mean_nll(per_piece_records, "true_context", tt),
            "model_flattened_context": per_type_mean_nll(per_piece_records, "flattened_context", tt),
            "model_swapped_context": per_type_mean_nll(per_piece_records, "swapped_context", tt),
        }

    def uniform_nll_for_type(tt):
        vals = [
            r["true_context"]["uniform_nll_by_type"][tt]
            for r in per_piece_records
            if tt in r["true_context"]["uniform_nll_by_type"]
        ]
        return mean(vals)

    # value-level accuracy ladder
    all_notes = [n for r in per_piece_records for n in r["notes"]]
    delta_notes = [n for n in all_notes if n["delta"] != 0]  # scope: delta accuracy only when a delta token existed

    def acc(key, notes=all_notes):
        vals = [n[key] for n in notes]
        return sum(vals) / len(vals) if vals else None

    # model argmax accuracy by type, aggregated across pieces (mean of per-piece rates)
    model_vel_acc = mean([r["true_context"]["argmax_acc_by_type"].get("VelocityLevel") for r in per_piece_records])
    model_delta_acc = mean([r["true_context"]["argmax_acc_by_type"].get("Delta") for r in per_piece_records])
    model_deltadir_acc = mean(
        [r["true_context"]["argmax_acc_by_type"].get("DeltaDirection") for r in per_piece_records]
    )

    accuracy_ladder = {
        "velocity": {
            "marginal_mode": acc("marginal_vel_correct"),
            "lag1_transition": acc("transition_vel_correct"),
            "model_argmax_token_level": model_vel_acc,
        },
        "delta_magnitude": {
            "marginal_mode": acc("marginal_delta_correct", delta_notes),
            "lag1_transition": acc("transition_delta_correct", delta_notes),
            "model_argmax_token_level": model_delta_acc,
            "model_deltadirection_argmax_token_level": model_deltadir_acc,
            "note": "scored only among notes where a Delta token was actually "
                    "emitted (true signed delta != 0); does not cover whether "
                    "the model correctly predicts 'no delta at all'.",
        },
    }

    def neg_mean(xs):
        m = mean(xs)
        return -m if m is not None else None

    value_level_nll = {
        "velocity": {
            "marginal": neg_mean([n["marginal_vel_logp"] for n in all_notes]),
            "lag1_transition": neg_mean([n["transition_vel_logp"] for n in all_notes]),
        },
        "delta": {
            "marginal": neg_mean([n["marginal_delta_logp"] for n in all_notes]),
            "lag1_transition": neg_mean([n["transition_delta_logp"] for n in all_notes]),
        },
    }

    # G1a: model (true context) beats a uniform-over-legal-tokens baseline,
    # per token type (types have very different domain sizes, e.g. 128 for
    # VelocityLevel vs 2 for DeltaDirection, so this must not be pooled).
    g1a = {
        tt: {
            "model_nll": nll_table[tt]["model_true_context"],
            "uniform_nll": uniform_nll_for_type(tt),
            "beats_uniform": (
                nll_table[tt]["model_true_context"] is not None
                and uniform_nll_for_type(tt) is not None
                and nll_table[tt]["model_true_context"] < uniform_nll_for_type(tt)
            ),
        }
        for tt in nll_table
    }
    # Also fold in the marginal/lag-1-transition value-level baselines for
    # VelocityLevel and Delta specifically, since those are the ones the
    # "smart guessing" question is really about.
    if "VelocityLevel" in g1a:
        g1a["VelocityLevel"]["marginal_nll"] = value_level_nll["velocity"]["marginal"]
        g1a["VelocityLevel"]["lag1_transition_nll"] = value_level_nll["velocity"]["lag1_transition"]
    if "Delta" in g1a:
        g1a["Delta"]["marginal_nll"] = value_level_nll["delta"]["marginal"]
        g1a["Delta"]["lag1_transition_nll"] = value_level_nll["delta"]["lag1_transition"]

    # G1b: true context vs swapped context, paired per piece, on total NLL/token
    paired_diffs = []
    for r in per_piece_records:
        t_true = r["true_context"]["n_tokens"]
        t_swap = r["swapped_context"]["n_tokens"]
        if t_true and t_swap:
            nll_true = -r["true_context"]["total_logp"] / t_true
            nll_swap = -r["swapped_context"]["total_logp"] / t_swap
            paired_diffs.append(nll_swap - nll_true)  # positive = swapped is worse (higher NLL) = context matters
    g1b = bootstrap_ci(paired_diffs)
    g1b_pass = g1b is not None and g1b["ci95_lo"] > 0

    summary = {
        "checkpoint": args.checkpoint,
        "n_pieces_scored": n_used,
        "n_pieces_attempted": n_attempted,
        "nll_by_token_type": nll_table,
        "value_level_nll": value_level_nll,
        "accuracy_ladder": accuracy_ladder,
        "gate_G1a_beats_baselines": g1a,
        "gate_G1b_true_vs_swapped_context": {
            "paired_nll_diff_swapped_minus_true": g1b,
            "passes": g1b_pass,
            "interpretation": (
                "positive mean with CI excluding zero => true context reduces "
                "NLL relative to swapped context => model is using context"
            ),
        },
    }

    with open(out_dir / "e1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / "e1_per_piece.json", "w") as f:
        json.dump(per_piece_records, f, indent=2, default=str)
    with open(out_dir / "e1_report.md", "w") as f:
        f.write(render_markdown_report(summary, args))

    print(json.dumps(summary, indent=2, default=str))


def render_markdown_report(summary: dict, args) -> str:
    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)

    lines = []
    lines.append("# E1 -- Held-out conditional NLL with context ablations and guessing baselines\n")
    lines.append(f"- Checkpoint: `{summary['checkpoint']}`")
    lines.append(f"- Pieces scored: {summary['n_pieces_scored']} / {summary['n_pieces_attempted']} attempted")
    lines.append(f"- Train pieces used to fit baselines: up to {args.train_limit}")
    lines.append(f"- Window: {args.n_bars} bars, 1 track, single humanized bar (last non-empty bar in the window)\n")

    lines.append("## Token-level NLL by type, and G1a (beats-baselines gate)\n")
    lines.append("| Token type | model (true ctx) | model (flattened ctx) | model (swapped ctx) | "
                  "uniform-over-legal | marginal | lag-1 transition | beats uniform |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tt, row in summary["gate_G1a_beats_baselines"].items():
        nll = summary["nll_by_token_type"][tt]
        lines.append(
            f"| {tt} | {fmt(nll['model_true_context'])} | {fmt(nll['model_flattened_context'])} | "
            f"{fmt(nll['model_swapped_context'])} | {fmt(row.get('uniform_nll'))} | "
            f"{fmt(row.get('marginal_nll', '-'))} | {fmt(row.get('lag1_transition_nll', '-'))} | "
            f"{'YES' if row['beats_uniform'] else 'NO'} |"
        )
    lines.append("\nLower NLL is better (in nats). `uniform-over-legal` is -log(n) where n is the actual "
                  "grammar-legal token count at that position (not the raw vocab size).\n")

    lines.append("## Value-level accuracy ladder (\"does it beat smart guessing\")\n")
    lines.append("| | marginal-mode guess | lag-1 transition guess | model (argmax) |")
    lines.append("|---|---|---|---|")
    vel = summary["accuracy_ladder"]["velocity"]
    dl = summary["accuracy_ladder"]["delta_magnitude"]
    lines.append(f"| VelocityLevel accuracy | {fmt(vel['marginal_mode'])} | {fmt(vel['lag1_transition'])} | "
                  f"{fmt(vel['model_argmax_token_level'])} |")
    lines.append(f"| Delta magnitude accuracy* | {fmt(dl['marginal_mode'])} | {fmt(dl['lag1_transition'])} | "
                  f"{fmt(dl['model_argmax_token_level'])} |")
    lines.append(f"\n*{dl['note']}\n")
    lines.append(f"DeltaDirection argmax accuracy (model, true context): "
                 f"{fmt(dl.get('model_deltadirection_argmax_token_level'))}\n")

    lines.append("## Gate G1b -- is the model using context at all?\n")
    g1b = summary["gate_G1b_true_vs_swapped_context"]
    d = g1b["paired_nll_diff_swapped_minus_true"]
    if d:
        lines.append(f"Paired per-piece NLL difference (swapped context − true context), mean/token: "
                     f"**{fmt(d['mean'])}** (95% CI [{fmt(d['ci95_lo'])}, {fmt(d['ci95_hi'])}], n={d['n_pieces']} pieces)")
        lines.append(f"\n**Gate {'PASSES' if g1b['passes'] else 'FAILS'}**: {g1b['interpretation']}\n")
    else:
        lines.append("Not enough paired data to compute.\n")

    lines.append("## Caveats / scope\n")
    lines.append("- Single track, single humanized bar per piece (not the full multi-bar/multi-track plan).")
    lines.append("- One checkpoint scored (final, `humanize_tiny-20260807-035822`).")
    lines.append("- Delta accuracy is scored only among notes where a Delta token was actually emitted "
                  "(true signed delta != 0) -- correctly predicting *no* delta is not covered here.")
    lines.append("- Marginal/transition baselines are fit on the training split (not held-out), so they "
                  "cannot cheat by peeking at the evaluation pieces.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
