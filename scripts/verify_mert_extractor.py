"""Verify MERTExtractor against the real pretrained MERT-v2 checkpoint(s).

Loads the real ``MERT2Model`` (via ``transformers.AutoModel.from_pretrained``,
``trust_remote_code=True``) through midigpt/audio/mert_extractor.py and reports:

  1. Load status. Unlike the YourMT3+ vendored-architecture path (which needs
     an explicit key-match report because it re-declares the module shapes by
     hand), this is a clean ``trust_remote_code`` load of the checkpoint's own
     ``modeling_mert2.py`` -- there is no separate architecture to match keys
     against. "did AutoModel.from_pretrained succeed" already IS the check;
     this also prints param/layer/dim counts so a human can sanity-check them
     against the checkpoint's advertised size (632M params, 24 layers,
     hidden_size=1024, 25Hz).
  2. A forward-pass smoke test on synthetic audio for each requested variant,
     checking output shape (T ~= L / hop_length, feature_dim == 1024), no
     NaNs, and that changing `layer` actually changes the output (proving the
     layer-selection code path is live, not a no-op).

This needs the real MERT-v2 checkpoints (not gated, but 632M params each) and
the 'transformers' package (pip install midigpt[mert]); meant to run on nibi
against the already-downloaded local checkpoint directories, not this sandbox.

Usage (local, if you have checkpoints + network/HF cache already set up):
    python3 scripts/verify_mert_extractor.py --variants fullsong 30s

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[mert]" &&
        python3 scripts/verify_mert_extractor.py \
            --weights-dir /scratch/triana24/MIDI-GPT/hf_cache \
            --variants fullsong 30s
    '
"""

from __future__ import annotations

import argparse
import sys

_WEIGHTS_DIRNAMES = {"fullsong": "mert-v2-fullsong", "30s": "mert-v2-30s"}


def _verify_one(variant: str, weights_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.mert_extractor import MERTExtractor

    print(f"\n=== {variant} ===")
    try:
        extractor = MERTExtractor(variant=variant, weights_path=weights_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load '{variant}' (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.model.parameters())
    n_tensors = len(list(extractor.model.state_dict()))
    print(
        f"  loaded OK -- AutoModel.from_pretrained(trust_remote_code=True) succeeded: "
        f"{n_tensors} tensors, {n_params / 1e6:.1f}M params"
    )
    print(
        f"  feature_dim={extractor.feature_dim} num_layers={extractor.num_layers} "
        f"layer={extractor.layer} sample_rate={extractor.sample_rate} hop_length={extractor.hop_length}"
    )

    extractor = extractor.eval()
    with torch.no_grad():
        audio = torch.randn(2, extractor.sample_rate * 3, device=device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T~{expected_T} expected)")
        assert out.shape[0] == 2 and out.shape[2] == extractor.feature_dim
        assert not torch.isnan(out).any(), "NaN in extractor output"

        mid_extractor = MERTExtractor(
            variant=variant, weights_path=weights_path, device=device, layer=max(1, extractor.num_layers // 2)
        ).eval()
        out_mid = mid_extractor(audio)
        assert not torch.allclose(out, out_mid), "layer= had no effect on output -- layer selection is dead code"
        print(f"  layer={mid_extractor.layer}/{extractor.num_layers} output differs from full stack: OK")

    print(f"PASS: {variant} loaded and ran end-to-end with real weights, no NaNs.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify MERTExtractor against real checkpoint(s).")
    ap.add_argument("--variants", nargs="+", default=["fullsong", "30s"], choices=["fullsong", "30s"])
    ap.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing mert-v2-fullsong/ and mert-v2-30s/ subdirs "
        "(e.g. /scratch/triana24/MIDI-GPT/hf_cache). If unset, resolves via the "
        "variant keyword -> HuggingFace Hub (requires network access).",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    results = {}
    for variant in args.variants:
        weights_path = None
        if args.weights_dir:
            weights_path = f"{args.weights_dir}/{_WEIGHTS_DIRNAMES[variant]}"
        results[variant] = _verify_one(variant, weights_path, args.device)

    print("\n=== Summary ===")
    for variant, ok in results.items():
        print(f"  {variant}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
