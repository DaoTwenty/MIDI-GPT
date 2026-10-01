# Vendored from mimbres/YourMT3

Source: https://github.com/mimbres/YourMT3 (paper: arXiv:2407.04822,
"YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer
Architectures and Cross-dataset Stem Augmentation").

The actual `amt/src/model/*.py` source (the GitHub repo's own history does
not carry these files directly under a plain `git clone`) was pulled from
the author-maintained mirror at https://huggingface.co/spaces/mimbres/YourMT3,
commit `5e66c1ea173a8186e0d20432b841d3180cc015b5` (2025-01-31), under
`amt/src/model/` and `amt/src/config/`. The top-level GitHub repo was cloned
at commit `8a4fbcabdf660f9f06bcb6f12bbf00d9b3139b98` (2024-11-29) to confirm
the paper/README pointers.

## License note

The top-level GitHub repo ships a repo-wide `LICENSE` file under
GPL-3.0. However, every file actually vendored here
(`t5mod.py`, `spectrogram.py`, `positional_encoding.py`, `ops.py`) carries
its own per-file header: "Copyright 2024 The YourMT3 Authors. Licensed
under the Apache License, Version 2.0 ... Please see the details in the
LICENSE file." `t5mod.py` additionally carries a second header block for
the portions derived from HuggingFace's T5 modeling code (Copyright 2018
Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team, also
Apache-2.0). Those per-file headers are kept verbatim in each vendored
file, as Apache-2.0 requires. `LICENSE` in this directory is the
Apache-2.0 text those headers point to.

This is a real discrepancy between the repo-wide GPL-3.0 `LICENSE` and the
per-file Apache-2.0 headers on the specific files we depend on (the
model-definition code, as opposed to the training loop / data pipeline /
eval scripts, which is most of what's actually GPL-3.0-only in that repo).
We're relying on the per-file headers as the license grant for those
specific files, since that's what their authors wrote on them, but this
hasn't been independently confirmed with the upstream authors -- flag for
legal review before any redistribution beyond this codebase.

## What's vendored and why

Only the minimal set of files/classes needed to run the plain-T5 encoder
variant of YourMT3+ (`encoder_type: 't5'`) as a frame-level feature
extractor -- i.e. `spectrogram -> encoder(inputs_embeds=...) ->
last_hidden_state`, no decoder, no task-token conditioning, no
PerceiverTF/conformer/MoE encoder variants (this repo ships five checkpoint
presets; only one, `notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72`
("YMT3+" in the HF Space's `app.py`), uses the plain-T5 encoder -- the
other four use PerceiverTF, optionally with MoE feed-forward layers and/or
a multi-channel T5 decoder, none of which are vendored here).

- `t5mod.py`: `T5BlockYMT3`, `T5StackYMT3`, `T5EncoderYMT3` only, from
  `amt/src/model/t5mod.py`. Dropped: `T5DecoderYMT3`, `MultiChannelT5Decoder`
  (decoder-side, unused by an encoder-only extractor), and `T5BlockYMT3`'s
  `ff_layer_type == 'moe'` branch (our target checkpoint always uses
  `'t5_gmlp'`, which maps directly onto HuggingFace's own `T5LayerFF` --
  the upstream repo's separate MoE implementation in `model/ff_layer.py`
  was never vendored as a result).
- `spectrogram.py`: `Melspectrogram`, `Spectrogram`,
  `get_spectrogram_layer_from_audio_cfg`, unmodified from
  `amt/src/model/spectrogram.py` other than dropping the `config.config`
  import (callers pass `audio_cfg` explicitly instead of relying on the
  upstream module-level default).
- `positional_encoding.py`: `FixedSinusoidalPositionalEmbedding` only,
  from `amt/src/model/positional_encoding.py`. Dropped:
  `AlibiPositionalBias`, `Wav2Vec2ConformerRotaryPositionalEmbedding`,
  `Wav2Vec2ConformerRelPositionalEmbedding`, `get_rotary_emb` and the
  `model.RoPE.RoPE` dependency they pull in -- all unused by the plain-T5
  (sinusoidal PE) encoder.
- `ops.py`: `minmax_normalize`, `optional_compiler_disable` only, from
  `amt/src/model/ops.py`. Dropped: `get_layer_norm`, `count_parameters`,
  `adjust_b_to_gcd`, `optional_compiler_dynamic` -- unused once the MoE/
  RMSNorm/gMLP paths above are gone.

Config values (audio frontend params, T5 encoder dims, the
`google/t5-v1_1-small` base T5Config overrides) were read out of
`amt/src/config/config.py` (`audio_cfg`, `model_cfg["encoder"]["t5"]`,
`T5_BASE_CFG`) and `amt/src/utils/task_manager.py` /
`amt/src/config/task.py` (`num_max_positions` depends on
`TaskManager("mt3_full_plus").max_task_token_length`, which is `0` --
that task entry doesn't set it, so the `.get(..., 0)` default applies).
Rather than vendor `config.py` itself (it pulls in `DEEPSPEED_CFG`,
`shared_cfg`, W&B/training config unrelated to feature extraction), the
handful of values actually needed are inlined as constants in
`midigpt/audio/yourmt3_real_extractor.py`, sourced from that same commit.

Full derivation (checkpoint -> exact config used):
`amt/src/../huggingface_spaces/app.py` hardcodes
`checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"`
for `model_name == "YMT3+"`, with **no** encoder/task overrides passed --
i.e. every `model_cfg`/`audio_cfg` default in `config.py` applies verbatim
for this specific checkpoint.
