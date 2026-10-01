"""Verify DACExtractor against the real pretrained DAC checkpoint(s).

Loads the real ``DacModel`` (via ``transformers.DacModel.from_pretrained`` --
a first-party library-integrated model, no ``trust_remote_code``, no vendored
modeling code) through midigpt/audio/dac_extractor.py and reports:

  1. Load status. Like the MERT/SheetSage2 paths, "did DacModel.from_pretrained
     succeed" already covers architecture correctness -- there is no separate
     hand-written module tree to match keys against. Also prints param/dim
     counts so a human can sanity-check them against the checkpoint's
     advertised size (76.6M params for 'dac_44khz'; feature_dim = n_codebooks
     * codebook_dim = 9 * 8 = 72, not config.hidden_size, which is an
     unrelated post-quantizer-projection width -- see dac_extractor.py's
     module docstring).
  2. A forward-pass smoke test on synthetic audio for each requested variant,
     checking output shape (T == L // hop_length, feature_dim == 72 for
     'dac_44khz'), no NaNs, and that the sub-hop_length zero-padding path
     (audio shorter than one hop_length) doesn't crash.

This needs the real DAC checkpoint(s) and the 'transformers' package
(pip install midigpt[dac]); meant to run on nibi against the already-
downloaded local checkpoint directory, not this sandbox -- see
scripts/verify_mert_extractor.py for the MERT-side counterpart this mirrors.

Note: this extractor's shape/logic (encode().projected_latents ->
(B, T, D) transpose, minimum_input_samples zero-padding, T == L //
hop_length for any L >= hop_length) was locally validated against a synthetic
tiny DacModel built from the real ``transformers.DacConfig``/``DacModel``
classes with the real checkpoint's downsampling_ratios=[2,4,8,8] (the only
config field that fixes the encoder's conv kernel/stride/padding shapes,
independent of encoder_hidden_size/n_codebooks/codebook_dim) -- not against
the real 76.6M-param 'descript/dac_44khz' weights, which need network access
this sandbox doesn't have.

Usage (local, if you have the checkpoint + network/HF cache already set up):
    python3 scripts/verify_dac_extractor.py --variants 44khz

Usage (on nibi via PRISM -- see .prism/config.yaml for the venv/module set):
    prism exec nibi -- '
        module load python/3.11.5 gcc/12.3 cmake/3.31.0 arrow/19.0.1 &&
        source /scratch/triana24/MIDI-GPT/venvs/venv-audio-conditioning/bin/activate &&
        cd /scratch/triana24/MIDI-GPT/worktree-audio-conditioning &&
        pip install -e ".[dac]" &&
        python3 scripts/verify_dac_extractor.py \
            --weights-dir /scratch/triana24/MIDI-GPT/hf_cache \
            --variants 44khz
    '
"""

from __future__ import annotations

import argparse
import sys

_WEIGHTS_DIRNAMES = {"44khz": "dac_44khz", "24khz": "dac_24khz", "16khz": "dac_16khz"}


def _verify_one(variant: str, weights_path: str | None, device: str) -> bool:
    import torch

    from midigpt.audio.dac_extractor import DACExtractor

    print(f"\n=== {variant} ===")
    try:
        extractor = DACExtractor(variant=variant, weights_path=weights_path, device=device)
    except Exception as e:
        print(f"FAIL: could not load '{variant}' (weights_path={weights_path!r}): {e!r}")
        return False

    n_params = sum(p.numel() for p in extractor.model.parameters())
    n_tensors = len(list(extractor.model.state_dict()))
    print(
        f"  loaded OK -- DacModel.from_pretrained() succeeded: "
        f"{n_tensors} tensors, {n_params / 1e6:.1f}M params"
    )
    print(
        f"  feature_dim={extractor.feature_dim} sample_rate={extractor.sample_rate} "
        f"hop_length={extractor.hop_length} minimum_input_samples={extractor.minimum_input_samples}"
    )

    extractor = extractor.eval()
    with torch.no_grad():
        audio = torch.randn(2, extractor.sample_rate * 3, device=device)  # 3s, batch of 2
        out = extractor(audio)
        expected_T = audio.shape[-1] // extractor.hop_length
        print(f"  forward (2, {audio.shape[-1]}) waveform -> {tuple(out.shape)} (T=={expected_T} expected)")
        assert out.shape == (2, expected_T, extractor.feature_dim), f"unexpected shape {tuple(out.shape)}"
        assert not torch.isnan(out).any(), "NaN in extractor output"

        short_audio = torch.randn(1, extractor.hop_length // 4, device=device)
        short_out = extractor(short_audio)
        print(f"  sub-hop_length forward ({tuple(short_audio.shape)}) -> {tuple(short_out.shape)} (zero-pad path OK)")
        assert short_out.shape[1] >= 1 and not torch.isnan(short_out).any()

    print(f"PASS: {variant} loaded and ran end-to-end with real weights, no NaNs.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify DACExtractor against real checkpoint(s).")
    ap.add_argument("--variants", nargs="+", default=["44khz"], choices=["44khz", "24khz", "16khz"])
    ap.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing dac_44khz/ (and optionally dac_24khz/, dac_16khz/) subdirs "
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
