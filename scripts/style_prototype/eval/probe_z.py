"""Eval metric 3: does z correlate with known expressive attributes without
collapsing onto them entirely? Near-perfect probe accuracy combined with low
metric-2 faithfulness would flag a trivial/degenerate z (e.g. one that just
memorizes a single scalar rather than capturing a richer style signal).

Fits linear probes (ordinary least squares, held-out R^2 -- NOT sklearn:
confirmed unavailable in venv-humanize, and EXPERIMENT_PLAN.md's own
convention for this codebase's eval scripts is numpy.linalg.lstsq instead)
from z (the StyleEncoder's raw 32-dim bottleneck output, NOT the z_proj-
projected soft-prefix vector) to two ordinal attributes computed directly
via their existing `.compute()` methods:
  - nomml (midigpt.attributes.nomml.Nomml, track-level, 0-12 ordinal):
    NOTE -- computed on the tokenizer's DECODED (quantized-grid) Score, not
    a freshly-read high-resolution MIDI; Nomml's own docstring says
    quantization "destroys the fine onset alignment this metric depends
    on", so these absolute nomml values are not directly comparable to
    GigaMIDI's published ones -- only the z-correlation is being tested
    here, not nomml's absolute accuracy.
  - velocity_range (midigpt.attributes.velocity.VelocityRange, bar-level):
    averaged over all bars of the piece (see below).

Real scope simplification (consistent with metric 2's precedent, documented
not assumed): labels are computed over the reference piece's WHOLE decoded
Score (track 0, all bars), not literally restricted to the reference-
segment bars z was extracted from -- no (track,bar) mask for the reference
segment is exposed by MidiGPTDataset (only a flat token-position
style_ref_mask), and re-deriving one would require re-walking Track/Bar
boundary tokens as dataset.py's internal `_expressive_mask` does but
doesn't expose. Whole-piece nomml/velocity_range is a reasonable proxy for
"this performance's overall expressive character", just not literally the
exact same bars z was pooled from.

Held-out (not in-sample) R^2: an in-sample OLS fit with z_dim=32 features
against a modest N would trivially look good even for a random/degenerate
z, which would make this metric's degenerate-z diagnostic useless -- an
80/20 train/test split is used instead.

Usage:
    python3 eval/probe_z.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random_10gb/conditioned_gpt2_random.pt \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/style_metric3_A_random \\
        [--limit 300] [--test-fraction 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import midigpt._core as _core
from held_out_conditional_loss import _load_variant_a  # noqa: E402
from midigpt.attributes.base import AttributeAnalyzer  # noqa: E402
from midigpt.attributes.nomml import Nomml  # noqa: E402
from midigpt.attributes.velocity import VelocityRange  # noqa: E402
from midigpt.tokenizer.tokenizer import Tokenizer  # noqa: E402
from midigpt.training.dataset import MidiGPTDataset  # noqa: E402
from steered_forward import _collect_usable_pieces  # noqa: E402
from style_vocab import StyleVocab  # noqa: E402


def piece_labels(tokenizer: Tokenizer, token_ids: list[int]) -> tuple[float, float] | None:
    score = tokenizer.decode(token_ids)
    if not score.tracks or not score.tracks[0].bars:
        return None
    nomml = Nomml().compute(score, 0)
    vr = VelocityRange()
    bars = score.tracks[0].bars
    vr_vals = [vr.compute(score, 0, b) for b in range(len(bars))]
    vr_vals = [v for v in vr_vals if v]  # VelocityRange.compute returns 0 for <2-note bars
    mean_vr = statistics.mean(vr_vals) if vr_vals else 0.0
    return float(nomml), float(mean_vr)


def held_out_r2(X: np.ndarray, y: np.ndarray, rng: random.Random, test_fraction: float) -> float | None:
    n = X.shape[0]
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, int(n * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    if len(train_idx) < X.shape[1] + 2:
        return None
    Xtr = np.hstack([X[train_idx], np.ones((len(train_idx), 1))])
    Xte = np.hstack([X[test_idx], np.ones((len(test_idx), 1))])
    ytr, yte = y[train_idx], y[test_idx]
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pred = Xte @ beta
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    return 1.0 - ss_res / ss_tot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=20, help="Repeated random train/test splits, averaged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    vocab = tokenizer._vocab
    style_vocab = StyleVocab(vocab)

    print("Loading checkpoint...", flush=True)
    model, _, style_cfg = _load_variant_a(args.checkpoint, device)
    if style_cfg.vocab_size != style_vocab.size:
        raise ValueError(
            f"Checkpoint's StyleVocab size={style_cfg.vocab_size} does not match "
            f"this encoder config's StyleVocab size={style_vocab.size}."
        )

    print("Building held-out dataset...", flush=True)
    dataset = MidiGPTDataset(
        args.val_parquet,
        tokenizer,
        infill_probability=0.0,
        humanize_probability=1.0,
        humanize_bar_fraction=0.5,
        mask_bar_config=None,
        max_seq_len=args.max_seq_len,
        context_mechanical_fraction=1.0,
        structured_target_probability=0.4,
        mechanical_coherent_targets=True,
    )
    pieces = _collect_usable_pieces(dataset, style_vocab, args.limit)
    n = len(pieces)
    print(f"Held-out pieces with a usable reference: {n}", flush=True)
    if n < 20:
        raise RuntimeError(f"Too few usable samples ({n}) -- check dataset/checkpoint compatibility.")

    print("Computing z and labels for each piece...", flush=True)
    zs, nommls, vranges = [], [], []
    for p in pieces:
        labels = piece_labels(tokenizer, p["input_ids"][: args.max_seq_len])
        if labels is None:
            continue
        ids = torch.tensor([p["style_ids"]], dtype=torch.long, device=device)
        mask = torch.ones(1, len(p["style_ids"]), dtype=torch.bool, device=device)
        with torch.no_grad():
            z = model.style_encoder(ids, mask)  # (1, z_dim), unit-norm
        zs.append(z.squeeze(0).cpu().numpy())
        nommls.append(labels[0])
        vranges.append(labels[1])

    X = np.stack(zs)
    y_nomml = np.array(nommls)
    y_vrange = np.array(vranges)
    print(f"Probing with {X.shape[0]} samples, z_dim={X.shape[1]}.", flush=True)

    def repeated_r2(y: np.ndarray) -> dict:
        scores = [
            r for _ in range(args.n_splits)
            if (r := held_out_r2(X, y, rng, args.test_fraction)) is not None
        ]
        if not scores:
            return {"mean_r2": None, "n_splits_used": 0}
        return {
            "mean_r2": statistics.mean(scores),
            "stdev_r2": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "n_splits_used": len(scores),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "n_scored": X.shape[0],
        "z_dim": X.shape[1],
        "nomml_probe": repeated_r2(y_nomml),
        "velocity_range_probe": repeated_r2(y_vrange),
    }
    (out_dir / "metric3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'metric3_summary.json'}")


if __name__ == "__main__":
    main()
