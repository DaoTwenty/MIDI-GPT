"""Verify MuQExtractor against the real pretrained MuQ checkpoint.

Loads the real ``muq.MuQ`` model (via ``MuQ.from_pretrained``, the official
``muq`` PyPI package's own ``PyTorchModelHubMixin``-based loader -- not
``transformers.AutoModel``) through midigpt/audio/muq_extractor.py and reports:

  1. Load status: "did `MuQ.from_pretrained` succeed" already IS the check
     here (same rationale as verify_mert_extractor.py) -- also prints
     param/layer/dim counts so a human can sanity-check them against the
     checkpoint's advertised size (~310M params, 12 Conformer layers,
     encoder_dim=1024, 25Hz).
  2. A forward-pass smoke test on synthetic audio, checking output shape
     (T ~= L / hop_length, feature_dim == 1024), no NaNs, that changing
     `layer` actually changes the output (proving the layer-selection code
     path is live, not a no-op), and that `len(output.hidden_states) ==
     num_layers + 1` (13) -- confirming the encoder_depth+1 indexing
     convention this extractor's forward() assumes (see muq_extractor.py's
     module docstring) actually holds against the real checkpoint, not just
     against transformers' source as read during development.

This needs the real MuQ-large-msd-iter checkpoint (not gated, but 310M
params) and the 'muq' package pinned to a transformers version in
[4.53, 4.54) (pip install midigpt[muq]) -- meant to run on nibi against the
already-downloaded local checkpoint directory, not this sandbox (which has
neither the disk headroom to install the torch/transformers/muq stack nor
network access to the checkpoint; see the coordinator report for what was
verified by static source reading instead).

Usage (local, if you have the checkpoint + network/HF cache already set up):
    python3 scripts/verify_muq_extractor.py

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[muq]" &&
        python3 scripts/verify_muq_extractor.py \
            --weights-path /scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter
    '
"""

from __future__ import annotations

import argparse
import sys


def _verify(weights_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.muq_extractor import MuQExtractor

    print("\n=== muq-large-msd-iter ===")
    try:
        extractor = MuQExtractor(weights_path=weights_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.model.parameters())
    n_tensors = len(list(extractor.model.state_dict()))
    print(
        f"  loaded OK -- MuQ.from_pretrained succeeded: "
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

        raw_output = extractor.model(audio, output_hidden_states=True)
        expected_n_hidden = extractor.num_layers + 1
        assert len(raw_output.hidden_states) == expected_n_hidden, (
            f"expected {expected_n_hidden} hidden_states entries (encoder_depth + 1 for the "
            f"pre-Conformer embedding), got {len(raw_output.hidden_states)} -- the layer= "
            "indexing convention this extractor assumes does not hold for this checkpoint"
        )
        print(f"  len(hidden_states)={len(raw_output.hidden_states)} == num_layers+1: OK")

        mid_extractor = MuQExtractor(
            weights_path=weights_path, device=device, layer=max(1, extractor.num_layers // 2)
        ).eval()
        out_mid = mid_extractor(audio)
        assert not torch.allclose(out, out_mid), "layer= had no effect on output -- layer selection is dead code"
        print(f"  layer={mid_extractor.layer}/{extractor.num_layers} output differs from full stack: OK")

    print("PASS: muq-large-msd-iter loaded and ran end-to-end with real weights, no NaNs.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify MuQExtractor against the real checkpoint.")
    ap.add_argument(
        "--weights-path",
        default=None,
        help="Local checkpoint directory (e.g. /scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter). "
        "If unset, resolves to 'OpenMuQ/MuQ-large-msd-iter' on the HuggingFace Hub (requires network access).",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ok = _verify(args.weights_path, args.device)

    print("\n=== Summary ===")
    print(f"  muq-large-msd-iter: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
