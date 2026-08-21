"""Activation steering (Section 6 of the style-conditioning plan): inject a
style direction directly into an *already-trained, unconditioned* Humanize
checkpoint's residual stream, instead of retraining with a soft-prefix.

Reuses Variant A's already-trained `z_proj` (z_dim -> n_embd, from
conditioned_gpt2.py's ConditionedGPT2) to project a reference performance's
`z` into the base model's embedding space -- projecting into "the space
GPT2 already learned to consume z from" is far more likely to produce a
coherent steering direction than an arbitrary/untrained projection, and
means this mechanism mechanically depends on a soft-prefix run (Variant A)
having already passed metric 1, which is why it's built after, not instead
of, `train_joint_conditioning.py`.

`forward_with_hooks()` (inference/model/gpt2.py) is read-only (fires
callbacks, doesn't feed mutated values back), so `steered_forward()` below
duplicates GPT2LMHeadModel.forward()'s block loop locally rather than
extending it -- mixing read-only introspection and mutation in one API
would muddy its contract.

`SteeredGPT2LMHeadModel`/`build_steered_engine` (below) DO drive real
SamplingSession/InferenceEngine generation -- `InferenceEngine.__init__`
accepts an already-built model object directly, and `_KVRunner.forward()`
calls it via pure duck typing, so this wrapper is a fully legal drop-in
with zero production code changes. Used by eval/steering_autoregressive.py
(real constrained autoregressive eval, KV-cached, grammar-constrained) and
by humanize_style_server.py (live steering-conditioned generation).

Usage:
    python3 steered_forward.py \\
        --base-checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny_e-20260807-223945/model_final.safetensors \\
        --conditioned-checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random_10gb/conditioned_gpt2_random.pt \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/steering_smoke_test \\
        --alphas 0,0.25,0.5,1,2,4,8 \\
        --layers all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

import midigpt._core as _core
from explicit_controls import CONTROL_NAMES, N_BINS, ControlTypeRanges, compute_controls, quantize_controls
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.dataset import MidiGPTDataset
from style_encoder import StyleEncoder, StyleEncoderConfig
from style_vocab import StyleVocab


def steered_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    steer_vec: torch.Tensor | dict[int, torch.Tensor],
    layers: set[int],
    alpha: float,
    past_kv: tuple | None = None,
    key_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple]:
    """Identical to GPT2LMHeadModel.forward()'s body, except `alpha *
    steer_vec` is added to the residual stream right after every block
    whose index is in `layers`.

    steer_vec: (n_embd,) broadcast to every layer in `layers` (the
    recommended, reused-z_proj case), or dict[int, Tensor(n_embd,)] for a
    distinct vector per layer.
    """
    B, T = input_ids.shape
    past_len = (
        past_kv[0][0].shape[2] if past_kv is not None and past_kv[0][0].shape[2] > 0 else 0
    )
    if position_ids is None:
        pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
    else:
        pos = position_ids.to(input_ids.device)

    x = model.transformer.wte(input_ids) + model.transformer.wpe(pos)
    presents = []
    for i, block in enumerate(model.transformer.h):
        pkv = past_kv[i] if past_kv is not None else None
        x, present, _ = block(x, past_kv=pkv, return_attn_weights=False, key_mask=key_mask)
        if i in layers:
            vec = steer_vec[i] if isinstance(steer_vec, dict) else steer_vec
            x = x + alpha * vec
        presents.append(present)
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, tuple(presents)


class SteeredGPT2LMHeadModel(nn.Module):
    """Wraps a plain GPT2LMHeadModel so InferenceEngine/_KVRunner can drive
    it exactly like the real model -- forward() delegates to
    steered_forward() above, the rest of the small ModelBase surface
    _KVRunner/SamplingSession actually touch (make_empty_kv/
    kv_null_positions/max_context, per session.py:252-283 and :292 and
    :1184) delegates straight through to the base model. Moved here from
    eval/steering_autoregressive.py (this session) so the eval script and
    humanize_style_server.py share one definition instead of two."""

    def __init__(self, base_model: GPT2LMHeadModel, steer_vec: torch.Tensor, layers: set[int], alpha: float):
        super().__init__()
        self.base_model = base_model
        self.steer_vec = steer_vec
        self.layers = layers
        self.alpha = alpha

    def forward(self, input_ids, past_kv=None, key_mask=None, position_ids=None):
        return steered_forward(
            self.base_model, input_ids, self.steer_vec, self.layers, self.alpha, past_kv, key_mask, position_ids
        )

    def make_empty_kv(self):
        return self.base_model.make_empty_kv()

    def kv_null_positions(self, past_kv, spans):
        return self.base_model.kv_null_positions(past_kv, spans)

    def max_context(self):
        return self.base_model.max_context()


def build_steered_engine(base_model, enc_cfg, steer_vec: torch.Tensor, layers: set[int], alpha: float, device: torch.device):
    """Mirrors InferenceEngine.from_checkpoint's body (engine.py:113-130),
    substituting a SteeredGPT2LMHeadModel for the plain loaded model.
    `base_model`/`enc_cfg` are loaded ONCE by the caller (checkpoint I/O is
    the expensive part) and reused across every call -- only the
    lightweight wrapper + engine.warmup() (a single-token forward pass) are
    rebuilt per call, since warmup's cached empty-kv and the wrapper's
    steer_vec/alpha are call-specific."""
    from midigpt.inference.engine import InferenceEngine

    wrapped = SteeredGPT2LMHeadModel(base_model, steer_vec, layers, alpha).to(device)
    wrapped.eval()
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    engine = InferenceEngine(wrapped, tokenizer, analyzer)
    engine.warmup()
    return engine


def load_steering_source(
    conditioned_checkpoint: str, device: torch.device
) -> tuple[StyleEncoder, nn.Linear, StyleEncoderConfig]:
    """Loads the trained StyleEncoder + z_proj out of a Variant A
    joint-conditioning checkpoint (conditioned_gpt2.py's ConditionedGPT2) --
    works for either the random-init or the pretrained/frozen-style-encoder
    ("A+B combo") checkpoint, since both are saved in the same format and
    the plan explicitly says to try encoders from both sources (Section 6)."""
    ckpt = torch.load(conditioned_checkpoint, map_location="cpu", weights_only=False)
    style_cfg: StyleEncoderConfig = ckpt["style_encoder_config"]
    gpt2_cfg: GPT2Config = ckpt["gpt2_config"]
    sd = ckpt["model_state_dict"]

    encoder = StyleEncoder(style_cfg).to(device)
    encoder.load_state_dict(
        {k[len("style_encoder.") :]: v for k, v in sd.items() if k.startswith("style_encoder.")}
    )
    z_proj = nn.Linear(style_cfg.z_dim, gpt2_cfg.n_embd).to(device)
    z_proj.load_state_dict(
        {k[len("z_proj.") :]: v for k, v in sd.items() if k.startswith("z_proj.")}
    )
    encoder.eval()
    z_proj.eval()
    return encoder, z_proj, style_cfg


def compute_steer_vector(
    encoder: StyleEncoder, z_proj: nn.Linear, style_ids: list[int], device: torch.device
) -> torch.Tensor:
    """style_ids: a reference segment's expressive tokens, already
    re-indexed into StyleVocab's local id space (see style_vocab.py)."""
    ids = torch.tensor([style_ids], dtype=torch.long, device=device)
    mask = torch.ones(1, len(style_ids), dtype=torch.bool, device=device)
    with torch.no_grad():
        z = encoder(ids, mask)
        vec = z_proj(z)
    return vec.squeeze(0)


def generate_steered(
    model: GPT2LMHeadModel,
    prompt_ids: torch.Tensor,
    steer_vec: torch.Tensor,
    layers: set[int],
    alpha: float,
    gen_len: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[int]:
    """Simple temperature sampling with a KV cache, no grammar-constrained
    masking -- we're not shipping this output, only measuring stats on it
    (see module docstring). prompt_ids: (1, T) raw LM token ids."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits, past_kv = steered_forward(model, prompt_ids, steer_vec, layers, alpha)
    next_logits = logits[:, -1, :]
    generated: list[int] = []
    for _ in range(gen_len):
        probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
        next_id = torch.multinomial(probs.cpu(), num_samples=1, generator=g).to(device)
        generated.append(next_id.item())
        logits, past_kv = steered_forward(
            model, next_id, steer_vec, layers, alpha, past_kv=past_kv
        )
        next_logits = logits[:, -1, :]
    return generated


def appendix_expressive_mask(input_ids: list[int], vocab: "_core.Vocabulary") -> list[bool]:
    """Boolean mask over `input_ids`, True at VelocityLevel/DeltaDirection/
    Delta positions strictly inside a HumanizeStart...HumanizeEnd appendix --
    the only region the model was ever trained to vary based on z (see
    resample_expressive_from_logits's docstring). Shared by that function
    and by eval/held_out_conditional_loss.py (metric 1's expr_ce), which
    previously used dataset.py's `_expressive_mask` -- that function
    explicitly excludes everything at/after TrackEnd (i.e. all appendix
    content, by design, for its own different purpose of flagging in-place
    reference-segment tokens for the style encoder), so it could never
    actually select the humanize target tokens metric 1 means to score."""
    mask = [False] * len(input_ids)
    in_appendix = False
    for i, tid in enumerate(input_ids):
        ttype = vocab.get_type(tid)
        if ttype == _core.TokenType.HumanizeStart:
            in_appendix = True
            continue
        if ttype == _core.TokenType.HumanizeEnd:
            in_appendix = False
            continue
        if in_appendix and ttype in (
            _core.TokenType.VelocityLevel,
            _core.TokenType.DeltaDirection,
            _core.TokenType.Delta,
        ):
            mask[i] = True
    return mask


def resample_expressive_from_logits(
    logits: torch.Tensor,
    input_ids: list[int],
    vocab: "_core.Vocabulary",
    ranges: ControlTypeRanges,
    temperature: float,
    generator: torch.Generator,
) -> list[int]:
    """Given per-position logits over a piece's TRUE input_ids -- however
    they were produced (soft-prefix ConditionedGPT2 forward, or
    steered_forward on the base checkpoint) -- resamples a new value at
    every VelocityLevel/DeltaDirection/Delta position INSIDE a
    HumanizeStart...HumanizeEnd appendix, restricted to that slot's own
    known token type's raw-id range (we already know the correct type from
    the true sequence -- not general grammar-constrained decoding). All
    other positions (structural tokens, and VelocityLevel/DeltaDirection/
    Delta tokens belonging to non-target bars emitted in place -- real or
    mechanized context the model was trained to reproduce as a fixed value,
    never as something z-conditioned) keep their true value. This is the
    only region the model was ever trained to vary based on style
    conditioning (see dataset.py's `humanize_bars`/`_encode_one` --
    NoteEncodeMode::Full for in-place/context bars vs. the appendix's
    ExpressiveOnly emission for target bars); resampling in-place expressive
    tokens too would dilute the profile-comparison metrics with noise from
    positions z was never trained to influence. Shared between metric 2
    (eval/style_transfer_faithfulness.py, soft-prefix) and metric 4
    (eval/steering_strength_sweep.py, activation steering) -- same
    evaluation protocol, different injection mechanism producing the
    logits."""
    type_ranges = {
        _core.TokenType.VelocityLevel: ranges.velocity,
        _core.TokenType.DeltaDirection: ranges.delta_direction,
        _core.TokenType.Delta: ranges.delta,
    }
    mask = appendix_expressive_mask(input_ids, vocab)
    new_ids = list(input_ids)
    for i, tid in enumerate(input_ids):
        if not mask[i]:
            continue
        r = type_ranges[vocab.get_type(tid)]
        slot_logits = logits[0, i, r.start : r.end] / max(temperature, 1e-6)
        probs = torch.softmax(slot_logits, dim=-1)
        choice = torch.multinomial(probs.cpu(), 1, generator=generator).item()
        new_ids[i] = r.start + choice
    return new_ids


def load_steering_source_c(
    conditioned_checkpoint: str, device: torch.device
) -> tuple[nn.ModuleList, nn.Linear]:
    """Variant C counterpart of load_steering_source: loads the trained
    per-control embedding tables + combine projection out of an
    ExplicitControlsGPT2 checkpoint (explicit_controls_gpt2.py). Same
    rationale as Variant A's z_proj reuse -- `combine` is the projection the
    model actually learned to consume a control vector through, so steering
    through it is far more likely to land in a coherent direction than an
    arbitrary one."""
    ckpt = torch.load(conditioned_checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    control_dim = ckpt["control_dim"]
    gpt2_cfg: GPT2Config = ckpt["gpt2_config"]

    control_embeds = nn.ModuleList([nn.Embedding(N_BINS, control_dim) for _ in CONTROL_NAMES]).to(device)
    for i, emb in enumerate(control_embeds):
        emb.load_state_dict(
            {k[len(f"control_embeds.{i}.") :]: v for k, v in sd.items() if k.startswith(f"control_embeds.{i}.")}
        )
    combine = nn.Linear(len(CONTROL_NAMES) * control_dim, gpt2_cfg.n_embd).to(device)
    combine.load_state_dict({k[len("combine.") :]: v for k, v in sd.items() if k.startswith("combine.")})
    control_embeds.eval()
    combine.eval()
    return control_embeds, combine


def compute_steer_vector_c(
    control_embeds: nn.ModuleList, combine: nn.Linear, control_bins: list[int], device: torch.device
) -> torch.Tensor:
    """control_bins: this reference's quantized control values (explicit_controls.
    quantize_controls output), one bin index per control, same order as
    CONTROL_NAMES."""
    with torch.no_grad():
        parts = [
            emb(torch.tensor([control_bins[i]], dtype=torch.long, device=device))
            for i, emb in enumerate(control_embeds)
        ]
        vec = combine(torch.cat(parts, dim=-1))
    return vec.squeeze(0)


def _collect_usable_pieces_c(dataset: MidiGPTDataset, ranges: ControlTypeRanges, limit: int) -> list[dict]:
    """Variant C counterpart of _collect_usable_pieces: same held-out
    filtering and humanize_bars/reference_bars threading, but the
    reference representation is `control_bins` (explicit_controls.
    compute_controls + quantize_controls on the reference segment) instead
    of StyleVocab-local style_ids."""
    pieces = []
    for i in range(len(dataset)):
        if len(pieces) >= limit:
            break
        item = dataset[i]
        ids, mask = item["input_ids"], item["style_ref_mask"]
        ref_raw = [t for t, m in zip(ids, mask, strict=True) if m]
        if not ref_raw:
            continue
        values = compute_controls(ref_raw, ranges)
        if values is None:
            continue
        humanize_bars_t0 = {b for (t, b) in item.get("humanize_bars", set()) if t == 0}
        reference_bars_t0 = {b for (t, b) in item.get("reference_bars", set()) if t == 0}
        pieces.append({
            "input_ids": ids,
            "ref_raw": ref_raw,
            "control_bins": quantize_controls(values),
            "humanize_bars": humanize_bars_t0,
            "reference_bars": reference_bars_t0,
        })
    return pieces


def _collect_usable_pieces(dataset: MidiGPTDataset, style_vocab: StyleVocab, limit: int) -> list[dict]:
    """Held-out pieces with a non-empty reference segment, same filter as
    eval/held_out_conditional_loss.py. Also carries through `humanize_bars`
    (this piece's own target bars -- the only region resample_expressive_
    from_logits is allowed to touch) and `reference_bars` (the bars whose
    expressive content produced `style_ids`/z), track 0 only, for scoping
    per-beat-profile comparisons (beat_profile.profile_points) to the bars
    that are actually meaningful instead of the whole piece."""
    pieces = []
    for i in range(len(dataset)):
        if len(pieces) >= limit:
            break
        item = dataset[i]
        ids, mask = item["input_ids"], item["style_ref_mask"]
        ref_raw = [t for t, m in zip(ids, mask, strict=True) if m]
        if not ref_raw:
            continue
        style_ids = style_vocab.to_local_seq(ref_raw)
        if not style_ids:
            continue
        humanize_bars_t0 = {b for (t, b) in item.get("humanize_bars", set()) if t == 0}
        reference_bars_t0 = {b for (t, b) in item.get("reference_bars", set()) if t == 0}
        pieces.append({
            "input_ids": ids,
            "ref_raw": ref_raw,
            "style_ids": style_ids,
            "humanize_bars": humanize_bars_t0,
            "reference_bars": reference_bars_t0,
        })
    return pieces


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-checkpoint", required=True, help="Unconditioned Humanize .safetensors")
    parser.add_argument("--conditioned-checkpoint", required=True, help="Variant A .pt (random or pretrained)")
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-index", type=int, default=0, help="Which held-out piece supplies z")
    parser.add_argument("--target-index", type=int, default=1, help="Which held-out piece's prefix is the generation prompt")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--layers", default="all", help="Comma-separated block indices, or 'all'")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--search-limit", type=int, default=50, help="How many held-out pieces to scan for usable reference/target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alphas.split(",")]

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    vocab = tokenizer._vocab
    style_vocab = StyleVocab(vocab)
    ranges = ControlTypeRanges(vocab)

    print(f"Loading base checkpoint: {args.base_checkpoint}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.base_checkpoint, device=device)
    model.eval()

    layers = set(range(model.cfg.n_layer)) if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    if not layers.issubset(set(range(model.cfg.n_layer))):
        raise ValueError(f"--layers {sorted(layers)} outside base model's {model.cfg.n_layer} blocks")

    print(f"Loading steering source: {args.conditioned_checkpoint}", flush=True)
    encoder, z_proj, style_cfg = load_steering_source(args.conditioned_checkpoint, device)
    if style_cfg.vocab_size != style_vocab.size:
        raise ValueError(
            f"Conditioned checkpoint's StyleVocab size={style_cfg.vocab_size} does not match "
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
        max_seq_len=2048,
        context_mechanical_fraction=1.0,
        structured_target_probability=0.4,
        mechanical_coherent_targets=True,
    )
    pieces = _collect_usable_pieces(dataset, style_vocab, args.search_limit)
    n_needed = max(args.reference_index, args.target_index) + 1
    if len(pieces) < n_needed:
        raise RuntimeError(
            f"Only found {len(pieces)} usable pieces scanning the first {args.search_limit} "
            f"rows; need at least {n_needed}. Raise --search-limit."
        )

    ref_piece = pieces[args.reference_index]
    target_piece = pieces[args.target_index]
    print(
        f"Reference piece #{args.reference_index}: {len(ref_piece['style_ids'])} style tokens. "
        f"Target piece #{args.target_index}: {len(target_piece['input_ids'])} tokens, "
        f"prompt_len={args.prompt_len}.",
        flush=True,
    )

    steer_vec = compute_steer_vector(encoder, z_proj, ref_piece["style_ids"], device)
    ref_controls = compute_controls(ref_piece["ref_raw"], ranges)

    prompt_len = min(args.prompt_len, len(target_piece["input_ids"]))
    prompt_ids = torch.tensor(
        [target_piece["input_ids"][:prompt_len]], dtype=torch.long, device=device
    )

    results = []
    for alpha in alphas:
        with torch.no_grad():
            generated = generate_steered(
                model, prompt_ids, steer_vec, layers, alpha,
                args.gen_len, args.temperature, device, args.seed,
            )
        gen_controls = compute_controls(generated, ranges)
        entry = {
            "alpha": alpha,
            "n_generated": len(generated),
            "controls": dict(zip(CONTROL_NAMES, gen_controls)) if gen_controls else None,
        }
        results.append(entry)
        c = entry["controls"]
        print(
            f"  alpha={alpha:>5}  controls={c}",
            flush=True,
        )

    summary = {
        "base_checkpoint": args.base_checkpoint,
        "conditioned_checkpoint": args.conditioned_checkpoint,
        "layers": sorted(layers),
        "reference_index": args.reference_index,
        "target_index": args.target_index,
        "reference_controls": dict(zip(CONTROL_NAMES, ref_controls)) if ref_controls else None,
        "sweep": results,
    }
    (out_dir / "steering_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'steering_sweep_summary.json'}")


if __name__ == "__main__":
    main()
