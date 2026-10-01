"""Verify YourMT3PlusExtractor against the real pretrained checkpoint.

Loads the vendored plain-T5 YourMT3+ encoder architecture
(midigpt/audio/yourmt3_real_extractor.py), points it at a real Lightning
checkpoint, and reports:

  1. Key-match statistics between the checkpoint's state_dict and the
     vendored architecture (matched / missing / unexpected / shape-mismatched),
     restricted to the spectrogram.*/encoder.* keys this extractor actually
     implements -- the check this whole rewrite exists for: the OLD fake
     YourMT3Extractor's load_pretrained() silently dropped almost all real
     weights via a loose "encoder" in k substring filter and strict=False;
     this script makes that failure mode impossible to miss.
  2. A tiny end-to-end forward pass on synthetic audio (single chunk and a
     multi-chunk, non-aligned-length clip), to catch structural/runtime
     issues (e.g. transformers-version API drift -- T5StackYMT3 depends on
     HF's ModuleUtilsMixin methods, which its own pyproject.toml constraint
     already pins around, but this is the actual end-to-end check).

This can't run in the sandbox that wrote it (no torch/the real 542MB
checkpoint there); it's meant to run in this project's nibi venv. See the
module docstring in yourmt3_real_extractor.py / that file's NOTICE.md for
architecture provenance.

Usage (locally, if you have the checkpoint and deps):
    python3 scripts/verify_yourmt3_extractor.py --checkpoint /path/to/model.ckpt

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[yourmt3]" &&
        python3 scripts/verify_yourmt3_extractor.py --checkpoint \
            /scratch/triana24/MIDI-GPT/hf_cache/yourmt3/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt
    '
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify YourMT3PlusExtractor against a real checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="Path to model.ckpt (Lightning checkpoint).")
    ap.add_argument("--device", default="cpu", help="Device for the forward-pass smoke test (default: cpu).")
    ap.add_argument(
        "--max-print", type=int, default=20, help="Max number of missing/unexpected keys to print (default: 20)."
    )
    args = ap.parse_args()

    import torch

    from midigpt.audio.yourmt3_real_extractor import (
        YourMT3PlusExtractor,
        _load_checkpoint_state_dict,
        key_match_report,
    )

    print(f"Loading checkpoint state_dict from {args.checkpoint} ...")
    state_dict = _load_checkpoint_state_dict(args.checkpoint)
    print(f"  {len(state_dict)} total keys in checkpoint state_dict.")

    print("Building YourMT3PlusExtractor (random init, no checkpoint) ...")
    extractor = YourMT3PlusExtractor()
    n_own = sum(1 for _ in extractor.state_dict())
    print(f"  {n_own} keys in the vendored spectrogram+encoder architecture.")

    report = key_match_report(extractor, state_dict)
    n_matched = len(report["matched"])
    n_missing = len(report["missing"])
    n_unexpected = len(report["unexpected"])
    n_shape_mismatched = len(report["shape_mismatched"])
    n_relevant = report["n_relevant_checkpoint_keys"]

    print()
    print("=== Key-match report (spectrogram.*/encoder.* keys only) ===")
    print(f"  checkpoint keys matching those prefixes : {n_relevant}")
    print(f"  extractor's own keys                    : {report['n_own_keys']}")
    print(f"  matched (same key, same shape)          : {n_matched}")
    print(f"  missing (in extractor, not in ckpt)     : {n_missing}")
    print(f"  unexpected (in ckpt, not in extractor)  : {n_unexpected}")
    print(f"  shape-mismatched (same key, diff shape) : {n_shape_mismatched}")
    match_pct = 100.0 * n_matched / report["n_own_keys"] if report["n_own_keys"] else 0.0
    print(f"  match rate                              : {match_pct:.1f}% of extractor's own keys")

    if n_missing:
        print(f"\n  missing keys (showing up to {args.max_print}):")
        for k in report["missing"][: args.max_print]:
            print(f"    {k}")
    if n_unexpected:
        print(f"\n  unexpected keys (showing up to {args.max_print}):")
        for k in report["unexpected"][: args.max_print]:
            print(f"    {k}")
    if n_shape_mismatched:
        print(f"\n  shape-mismatched keys (showing up to {args.max_print}):")
        for k, own_shape, ckpt_shape in report["shape_mismatched"][: args.max_print]:
            print(f"    {k}: extractor={own_shape} checkpoint={ckpt_shape}")

    ok = n_matched > 0 and match_pct > 95.0 and n_shape_mismatched == 0
    print()
    if not ok:
        print("FAIL: key-match rate is not near-100%, or shape mismatches were found.")
        print("      This is exactly the failure mode this rewrite exists to catch --")
        print("      do not proceed to use this checkpoint until resolved.")
        sys.exit(1)
    print("PASS: near-100% key match, no shape mismatches.")

    print("\nLoading the matched weights for real and running a forward-pass smoke test ...")
    extractor = YourMT3PlusExtractor(checkpoint_path=args.checkpoint)
    extractor = extractor.to(args.device).eval()

    with torch.no_grad():
        single_chunk = torch.randn(2, 32767, device=args.device)
        out1 = extractor(single_chunk)
        print(f"  single-chunk (2, 32767) waveform -> {tuple(out1.shape)} hidden states")
        assert out1.shape == (2, out1.shape[1], extractor.feature_dim)

        multi_chunk = torch.randn(1, int(16000 * 5.3), device=args.device)  # non-chunk-aligned length
        out2 = extractor(multi_chunk)
        print(f"  multi-chunk (1, {multi_chunk.shape[1]}) waveform -> {tuple(out2.shape)} hidden states")
        assert out2.shape[0] == 1 and out2.shape[2] == extractor.feature_dim
        assert not torch.isnan(out2).any(), "NaN in encoder output"

    print("\nPASS: forward pass ran end-to-end with real weights loaded, no NaNs.")


if __name__ == "__main__":
    main()
