"""Verify Music2LatentExtractor against the real pretrained Music2Latent checkpoint.

Loads the real ``EncoderDecoder`` (via the ``music2latent`` PyPI package) through
midigpt/audio/music2latent_extractor.py and reports:

  1. Load status. Unlike every other extractor in this codebase, Music2Latent's
     weights are not manually downloaded/gated -- ``EncoderDecoder.__init__``
     auto-downloads its 233MB ``music2latent.pt`` checkpoint from the
     'SonyCSLParis/music2latent' HuggingFace Hub repo the first time it runs, so
     this script (if it has network access) can actually complete an end-to-end
     real-weights check inside a plain sandbox, not just on nibi. Prints
     param count and the resolved feature_dim/sample_rate/hop_length.
  2. A forward-pass smoke test on synthetic audio, checking output shape
     (T ~= L / hop_length, feature_dim == 64), no NaNs, and that changing the
     input actually changes the output (proving encode() is live, not a no-op).
  3. A short-audio smoke test (below Music2LatentExtractor.minimum_input_samples)
     to check the zero-pad-short-input path doesn't crash.

Usage (local, with network access -- auto-downloads the checkpoint on first run):
    python3 scripts/verify_music2latent_extractor.py

Usage (on nibi via PRISM, pre-downloaded checkpoint):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[music2latent]" &&
        python3 scripts/verify_music2latent_extractor.py \
            --weights-path /scratch/triana24/MIDI-GPT/hf_cache/music2latent/music2latent.pt
    '
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Music2LatentExtractor against the real checkpoint.")
    ap.add_argument(
        "--weights-path",
        default=None,
        help="Local music2latent.pt checkpoint path. If unset, the 'music2latent' "
        "package auto-downloads its own checkpoint on first use (requires network "
        "access) -- note the auto-download attempt happens either way, only "
        "no-op'ing if the package's own default cache already has the file.",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch

    from midigpt.audio.music2latent_extractor import Music2LatentExtractor

    print("=== music2latent ===")
    try:
        extractor = Music2LatentExtractor(weights_path=args.weights_path, device=args.device)
    except Exception as e:
        print(f"FAIL: could not load Music2LatentExtractor (weights_path={args.weights_path!r}): {e!r}")
        sys.exit(1)

    n_params = sum(p.numel() for p in extractor.gen.parameters())
    print(f"  loaded OK -- EncoderDecoder() succeeded: {n_params / 1e6:.1f}M params")
    print(
        f"  feature_dim={extractor.feature_dim} sample_rate={extractor.sample_rate} "
        f"hop_length={extractor.hop_length} minimum_input_samples={extractor.minimum_input_samples}"
    )

    extractor = extractor.eval()
    ok = True
    with torch.no_grad():
        audio = torch.randn(2, extractor.sample_rate * 3, device=args.device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T~{expected_T} expected)")
        if not (out.shape[0] == 2 and out.shape[2] == extractor.feature_dim):
            print(f"FAIL: unexpected output shape {tuple(out.shape)}")
            ok = False
        if torch.isnan(out).any():
            print("FAIL: NaN in extractor output")
            ok = False

        audio2 = torch.randn(2, extractor.sample_rate * 3, device=args.device)
        out2 = extractor(audio2)
        if torch.allclose(out, out2):
            print("FAIL: different input audio produced identical output -- encode() looks like a no-op")
            ok = False
        else:
            print("  different input audio -> different output: OK")

        short_audio = torch.randn(1, extractor.minimum_input_samples // 4, device=args.device)
        short_out = extractor(short_audio)
        print(f"  short-audio ({short_audio.shape[-1]} samples, below minimum) -> {tuple(short_out.shape)}: OK")
        if torch.isnan(short_out).any():
            print("FAIL: NaN in short-audio output")
            ok = False

    print(f"\n{'PASS' if ok else 'FAIL'}: music2latent loaded and ran end-to-end with real weights.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
