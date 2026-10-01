"""Verify SAMEExtractor against the real pretrained SAME-L / SAME-S checkpoint(s).

Loads the real ``AudioAutoencoder`` (via the ``stable_audio_3`` package's own
``load_autoencoder`` / ``AutoencoderModel.from_pretrained``) through
midigpt/audio/same_extractor.py and reports:

  1. Load status. Like the MERT/DAC paths, "did the real loading call succeed"
     already covers architecture correctness -- ``stable_audio_3``'s
     ``SAMEEncoder``/``SAMEDecoder`` are the checkpoint's own reference classes,
     not a hand-written module tree we'd need to key-match. Prints
     param/tensor counts so a human can sanity-check them against the
     checkpoint's advertised size (852M params for same-l, 108M for same-s)
     and the resolved feature_dim/sample_rate/hop_length (should be 256 /
     44100 / ~4096 for both, per the checkpoints' config.json).
  2. A forward-pass smoke test on synthetic mono audio for each requested
     variant, checking output shape (T ~= L / hop_length, feature_dim == 256),
     no NaNs, and that different input audio produces different output
     (proving encode_audio() is live, not a no-op) -- SAME has no `layer`
     selection knob to test against (unlike MERT/MuScriptor), since the whole
     point of this extractor is the bottleneck's single output, not a choice
     of intermediate transformer block.
  3. A short-audio smoke test (well below one hop_length / downsampling_ratio)
     to check the zero-pad-to-multiple-of-downsampling_ratio path (handled
     internally by AudioAutoencoder.preprocess_audio_list_for_encoder) doesn't
     crash and still produces >=1 frame.

This needs the real, gated 'stabilityai/SAME-L'/'stabilityai/SAME-S' checkpoints
(license must be accepted, HF_TOKEN configured) and the 'stable_audio_3' package
(pip install midigpt[same] -- a git+https dependency, not a plain PyPI package,
see pyproject.toml's 'same' extra comment); meant to run on nibi against the
already-downloaded local checkpoint directories, not this sandbox -- see
scripts/verify_mert_extractor.py / verify_dac_extractor.py for the
counterparts this mirrors.

NOT verified locally: this sandbox has neither torch/transformers/stable_audio_3
installed nor the disk headroom to install them (>=1GB just for torch; sandbox
had ~250MB free at the time this was written), so nothing here has actually been
run -- only read against the real stable_audio_3 source
(models/autoencoders.py, loading_utils.py, model.py, model_configs.py,
inference/audio_utils.py, models/bottleneck.py, models/pretransforms.py, all
fetched directly from github.com/Stability-AI/stable-audio-3 at HEAD). The
coordinator should run this on nibi (or any machine with network + disk) before
trusting the forward-pass shape/NaN claims above.

Usage (local, if you have checkpoints + network/HF cache already set up):
    python3 scripts/verify_same_extractor.py --variants same-l same-s

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[same]" &&
        python3 scripts/verify_same_extractor.py \
            --weights-dir /scratch/triana24/MIDI-GPT/hf_cache \
            --variants same-l same-s
    '
"""

from __future__ import annotations

import argparse
import sys

_WEIGHTS_DIRNAMES = {"same-l": "same-l", "same-s": "same-s"}


def _verify_one(variant: str, weights_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.same_extractor import SAMEExtractor

    print(f"\n=== {variant} ===")
    try:
        extractor = SAMEExtractor(variant=variant, weights_path=weights_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load '{variant}' (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.autoencoder.parameters())
    n_tensors = len(list(extractor.autoencoder.state_dict()))
    print(
        f"  loaded OK -- real AudioAutoencoder load succeeded: "
        f"{n_tensors} tensors, {n_params / 1e6:.1f}M params"
    )
    print(
        f"  feature_dim={extractor.feature_dim} sample_rate={extractor.sample_rate} "
        f"hop_length={extractor.hop_length}"
    )

    extractor = extractor.eval()
    ok = True
    with torch.no_grad():
        audio = torch.randn(2, extractor.sample_rate * 3, device=device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T~{expected_T} expected)")
        if not (out.shape[0] == 2 and out.shape[2] == extractor.feature_dim):
            print(f"FAIL: unexpected output shape {tuple(out.shape)}")
            ok = False
        if torch.isnan(out).any():
            print("FAIL: NaN in extractor output")
            ok = False

        audio2 = torch.randn(2, extractor.sample_rate * 3, device=device)
        out2 = extractor(audio2)
        if torch.allclose(out, out2):
            print("FAIL: different input audio produced identical output -- encode_audio() looks like a no-op")
            ok = False
        else:
            print("  different input audio -> different output: OK")

        short_audio = torch.randn(1, max(1, extractor.hop_length // 4), device=device)
        short_out = extractor(short_audio)
        print(f"  sub-hop_length forward ({tuple(short_audio.shape)}) -> {tuple(short_out.shape)} (zero-pad path OK)")
        if not (short_out.shape[1] >= 1 and not torch.isnan(short_out).any()):
            print("FAIL: short-audio path produced empty output or NaNs")
            ok = False

    print(f"{'PASS' if ok else 'FAIL'}: {variant} loaded and ran end-to-end with real weights.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify SAMEExtractor against real checkpoint(s).")
    ap.add_argument("--variants", nargs="+", default=["same-l", "same-s"], choices=["same-l", "same-s"])
    ap.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing same-l/ and same-s/ subdirs, each with "
        "model_config.json + model.safetensors (e.g. /scratch/triana24/MIDI-GPT/hf_cache). "
        "If unset, resolves via the variant keyword -> HuggingFace Hub (requires network "
        "access and an HF_TOKEN with the gated Stability AI Community License accepted).",
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
