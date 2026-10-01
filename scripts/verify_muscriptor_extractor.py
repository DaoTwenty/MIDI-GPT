"""Verify MuScriptorExtractor against the real pretrained checkpoint(s).

Loads the real `muscriptor` package's LMModel through
midigpt/audio/muscriptor_extractor.py and reports:

  1. Key-match / load status. Unlike the fake GenericProxyAudioExtractor (which
     used a loose "encoder" in k substring filter + strict=False, silently
     dropping almost all real weights), the real muscriptor package's own
     TranscriptionModel.load_model() calls model.load_state_dict(state_dict)
     with the default strict=True -- ANY key or shape mismatch raises
     immediately instead of loading near-random weights. So "did this script
     get past the load line" already IS the key-match check; this just also
     prints the tensor/parameter counts so a human can sanity-check them
     against the checkpoint's advertised size (small ~103M/14 layers/768dim,
     medium ~307M/24 layers/1024dim, large ~1.3-1.4B/48 layers/1536dim,
     24 heads).
  2. A forward-pass smoke test on synthetic audio for each requested size,
     checking output shape (T ~= L / hop_length, feature_dim matches the
     checkpoint's own config.json), no NaNs, and that changing `layer`
     actually changes the output (proving the layer-selection code path is
     live, not a no-op).

This needs the real, gated MuScriptor checkpoints and the `muscriptor`
package (pip install midigpt[muscriptor]); it cannot run without network/HF
auth to the gated repo, which is why it's meant for nibi, not this sandbox.

Usage (local, if you have checkpoints + HF auth already set up):
    python3 scripts/verify_muscriptor_extractor.py --sizes small medium large

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[muscriptor]" &&
        python3 scripts/verify_muscriptor_extractor.py \
            --weights-dir /scratch/triana24/MIDI-GPT/hf_cache \
            --sizes small medium large
    '
"""

from __future__ import annotations

import argparse
import sys


def _verify_one(size: str, weights_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.muscriptor_extractor import MuScriptorExtractor

    print(f"\n=== {size} ===")
    try:
        extractor = MuScriptorExtractor(model_size=size, weights_path=weights_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load '{size}' (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.lm.parameters())
    n_tensors = len(list(extractor.lm.state_dict()))
    print(
        f"  loaded OK -- strict load_state_dict succeeded (this IS the key-match check): "
        f"{n_tensors} tensors, {n_params / 1e6:.1f}M params"
    )
    print(
        f"  feature_dim={extractor.feature_dim} num_layers={extractor.num_layers} "
        f"layer={extractor.layer} sample_rate={extractor.sample_rate} hop_length={extractor.hop_length}"
    )

    extractor = extractor.eval()
    with torch.no_grad():
        audio = torch.randn(2, 16000 * 3, device=device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length + 1
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T~{expected_T} expected)")
        assert out.shape[0] == 2 and out.shape[2] == extractor.feature_dim
        assert not torch.isnan(out).any(), "NaN in extractor output"

        mid_extractor = MuScriptorExtractor(
            model_size=size, weights_path=weights_path, device=device, layer=max(1, extractor.num_layers // 2)
        ).eval()
        out_mid = mid_extractor(audio)
        assert not torch.allclose(out, out_mid), "layer= had no effect on output -- layer selection is dead code"
        print(f"  layer={mid_extractor.layer}/{extractor.num_layers} output differs from full stack: OK")

    print(f"PASS: {size} loaded and ran end-to-end with real weights, no NaNs.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify MuScriptorExtractor against real checkpoint(s).")
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"], choices=["small", "medium", "large"])
    ap.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing muscriptor-<size>/model.safetensors subdirs "
        "(e.g. /scratch/triana24/MIDI-GPT/hf_cache). If unset, resolves via "
        "the size keyword -> HuggingFace Hub (requires HF auth for the gated repo).",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    results = {}
    for size in args.sizes:
        weights_path = None
        if args.weights_dir:
            weights_path = f"{args.weights_dir}/muscriptor-{size}/model.safetensors"
        results[size] = _verify_one(size, weights_path, args.device)

    print("\n=== Summary ===")
    for size, ok in results.items():
        print(f"  {size}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
