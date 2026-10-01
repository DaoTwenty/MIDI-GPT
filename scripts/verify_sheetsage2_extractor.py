"""Verify SheetSage2Extractor against the real pretrained SheetSage2 checkpoint.

Loads the real ``SheetSage2Model`` (via ``transformers.AutoModel.from_pretrained``,
``trust_remote_code=True``) through midigpt/audio/sheetsage2_extractor.py and reports:

  1. Load status. Like the MERT path (and unlike YourMT3+'s vendored-architecture
     path), this is a clean ``trust_remote_code`` load of the checkpoint's own
     ``modeling_sheetsage2.py`` -- there is no separate architecture to match keys
     against. For a ``weights_format="adapter"`` checkpoint (the real distributed
     format), loading also fetches/locates the frozen MERT-v2-FullSong parent
     backbone (sha256-verified against the adapter checkpoint's pinned revision)
     and merges the LoRA adapter into it in-place -- "did AutoModel.from_pretrained
     succeed" already covers all of that. Also prints param/dim counts so a human
     can sanity-check them against the checkpoint's advertised size (632M-param
     backbone + a small BART decoder, hidden_size=512 cross-attended encoder
     output, 25Hz).
  2. A forward-pass smoke test on synthetic audio, checking output shape
     (T ~= L / hop_length, feature_dim == 512), no NaNs, and that the extractor
     rejects audio longer than one ``input_audio_length`` window (300s).

This needs the real SheetSage2 checkpoint (632M-param backbone + adapter/decoder
weights) and the 'transformers'/'mir_eval' packages (pip install midigpt[sheetsage2]);
meant to run on nibi against the already-downloaded local checkpoint directory, not
this sandbox -- see scripts/verify_mert_extractor.py for the MERT-side counterpart
this mirrors.

Note: this extractor's shape/logic (get_audio_features -> encoder_last_hidden_state,
window-padding, frame-count slicing) was locally validated against a synthetic tiny
checkpoint built from the real vendored modeling code with weights_format="merged"
(bypassing the adapter-merge-at-load / parent-hash-verification codepath, which
needs the real 632M-param backbone and can only be exercised here, on nibi).

Usage (local, if you have the checkpoint + network/HF cache already set up):
    python3 scripts/verify_sheetsage2_extractor.py --weights-path m-a-p/SheetSage2

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[sheetsage2]" &&
        python3 scripts/verify_sheetsage2_extractor.py \
            --weights-path /scratch/triana24/MIDI-GPT/hf_cache/sheetsage2 \
            --base-model-path /scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong
    '
"""

from __future__ import annotations

import argparse
import sys


def _verify(weights_path: str | None, base_model_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.sheetsage2_extractor import SheetSage2Extractor

    print(f"\n=== sheetsage2 (weights_path={weights_path!r}, base_model_path={base_model_path!r}) ===")
    try:
        extractor = SheetSage2Extractor(weights_path=weights_path, base_model_path=base_model_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load SheetSage2 (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.model.parameters())
    n_tensors = len(list(extractor.model.state_dict()))
    print(
        f"  loaded OK -- AutoModel.from_pretrained(trust_remote_code=True) succeeded: "
        f"{n_tensors} tensors, {n_params / 1e6:.1f}M params"
    )
    print(
        f"  feature_dim={extractor.feature_dim} sample_rate={extractor.sample_rate} "
        f"hop_length={extractor.hop_length} minimum_input_samples={extractor.minimum_input_samples} "
        f"window_samples={extractor.window_samples}"
    )
    if extractor.feature_dim != 512:
        print(f"  WARNING: expected feature_dim=512 (config.hidden_size); got {extractor.feature_dim}")

    extractor = extractor.eval()
    with torch.no_grad():
        audio = torch.randn(2, extractor.sample_rate * 3, device=device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T~{expected_T} expected)")
        assert out.shape[0] == 2 and out.shape[2] == extractor.feature_dim
        assert not torch.isnan(out).any(), "NaN in extractor output"

        try:
            extractor(torch.randn(1, extractor.window_samples + 1, device=device))
        except ValueError:
            print("  correctly rejected audio longer than one input_audio_length window: OK")
        else:
            raise AssertionError("expected ValueError for audio longer than one window")

    print("PASS: sheetsage2 loaded and ran end-to-end with real weights, no NaNs.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify SheetSage2Extractor against the real checkpoint.")
    ap.add_argument(
        "--weights-path",
        default=None,
        help="Local checkpoint directory (e.g. /scratch/triana24/MIDI-GPT/hf_cache/sheetsage2) or "
        "HuggingFace Hub repo ID. If unset, resolves to 'm-a-p/SheetSage2' (requires network access).",
    )
    ap.add_argument(
        "--base-model-path",
        default=None,
        help="Local directory for the frozen MERT-v2-FullSong parent backbone (e.g. "
        "/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong, the same directory MERTExtractor "
        "uses), only needed for weights_format='adapter' checkpoints (the real distributed format). "
        "Without it, resolves to 'm-a-p/MERT-v2-FullSong' (requires network access).",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ok = _verify(args.weights_path, args.base_model_path, args.device)

    print("\n=== Summary ===")
    print(f"  sheetsage2: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
