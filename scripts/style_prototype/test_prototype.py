"""Unit/sanity tests for the style-conditioning prototype components.

Prototype-scope (not part of the production test suite in tests/python/) --
run directly with pytest from within scripts/style_prototype/, or:
    python3 -m pytest scripts/style_prototype/test_prototype.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import midigpt._core as _core
from style_encoder import StyleEncoder, StyleEncoderConfig
from style_vocab import StyleVocab
from train_contrastive import info_nce_loss


def _humanize_vocab() -> "_core.Vocabulary":
    repo_root = Path(__file__).resolve().parents[2]
    enc_json = (repo_root / "models" / "humanize_encoder.json").read_text()
    cfg = _core.EncoderConfig.from_json(enc_json)
    return _core.Vocabulary(cfg)


class TestStyleVocab:
    def test_size_matches_velocity_delta_domains(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        vel_n = vocab.range(_core.TokenType.VelocityLevel)
        dd_n = vocab.range(_core.TokenType.DeltaDirection)
        delta_n = vocab.range(_core.TokenType.Delta)
        expected = (vel_n[1] - vel_n[0]) + (dd_n[1] - dd_n[0]) + (delta_n[1] - delta_n[0])
        assert sv.size == expected

    def test_round_trip(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        for local in range(sv.size):
            raw = sv.to_raw(local)
            assert sv.is_expressive(raw)
            assert sv.to_local(raw) == local

    def test_to_local_seq_filters_non_expressive(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        vel_start, _ = vocab.range(_core.TokenType.VelocityLevel)
        bar_id = vocab.encode_val(_core.TokenType.Bar, 0)
        seq = [bar_id, vel_start, bar_id, vel_start + 1]
        local_seq = sv.to_local_seq(seq)
        assert local_seq == [sv.to_local(vel_start), sv.to_local(vel_start + 1)]


class TestStyleEncoder:
    def test_forward_shape_and_unit_norm(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        cfg = StyleEncoderConfig(vocab_size=sv.size)
        enc = StyleEncoder(cfg)

        B, T = 4, 50
        token_ids = torch.randint(0, sv.size, (B, T))
        key_mask = torch.ones(B, T, dtype=torch.bool)
        key_mask[0, 30:] = False  # simulate padding on one sample

        z = enc(token_ids, key_mask)
        assert z.shape == (B, cfg.z_dim)
        assert torch.allclose(z.norm(dim=-1), torch.ones(B), atol=1e-4)
        assert not torch.isnan(z).any()

    def test_gradients_flow_to_embed_and_proj(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        cfg = StyleEncoderConfig(vocab_size=sv.size)
        enc = StyleEncoder(cfg)

        token_ids = torch.randint(0, sv.size, (2, 20))
        z = enc(token_ids)
        z.sum().backward()
        assert enc.embed.weight.grad is not None
        assert enc.embed.weight.grad.abs().sum() > 0
        assert enc.proj.weight.grad is not None
        assert enc.proj.weight.grad.abs().sum() > 0

    def test_forward_without_key_mask(self):
        vocab = _humanize_vocab()
        sv = StyleVocab(vocab)
        cfg = StyleEncoderConfig(vocab_size=sv.size)
        enc = StyleEncoder(cfg)
        z = enc(torch.randint(0, sv.size, (3, 10)))
        assert z.shape == (3, cfg.z_dim)


class TestInfoNCELoss:
    def test_perfect_alignment_beats_chance(self):
        """If z_a[i] matches z_p[i] far better than any z_p[j!=i], loss
        should be well below log(B) (chance level for B-way classification)."""
        torch.manual_seed(0)
        B, D = 8, 16
        base = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        z_a = base
        # positives = same direction as anchors + tiny noise, still far
        # closer to their own anchor than to any other anchor.
        z_p = torch.nn.functional.normalize(base + 0.01 * torch.randn(B, D), dim=-1)
        loss = info_nce_loss(z_a, z_p, temperature=0.1)
        chance = torch.log(torch.tensor(float(B))).item()
        assert loss.item() < chance * 0.5

    def test_random_unrelated_pairs_near_chance(self):
        torch.manual_seed(0)
        B, D = 8, 16
        z_a = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        z_p = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = info_nce_loss(z_a, z_p, temperature=0.5)
        chance = torch.log(torch.tensor(float(B))).item()
        # Unrelated random pairs shouldn't systematically beat chance by much.
        assert loss.item() > chance * 0.5
