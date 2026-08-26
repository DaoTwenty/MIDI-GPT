"""Unit tests for soft token injection into GPT2LMHeadModel."""

import torch

from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel


def test_soft_token_injection_forward():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=512)
    model = GPT2LMHeadModel(cfg)

    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))

    # 1. Standard discrete forward
    logits_std, kv_std = model(input_ids)
    assert logits_std.shape == (B, T, 100)

    # 2. Forward with soft token replacement at position 2 and 5
    soft_emb_2 = torch.randn(B, 64)
    soft_emb_5 = torch.randn(B, 64)

    soft_replacements = {
        2: soft_emb_2,
        5: soft_emb_5,
    }

    logits_soft, kv_soft = model(input_ids, soft_token_replacements=soft_replacements)
    assert logits_soft.shape == (B, T, 100)
    assert not torch.isnan(logits_soft).any()

    # Verify that replacing tokens changed the logits at position 2
    assert not torch.allclose(logits_std[:, 2, :], logits_soft[:, 2, :])


def test_inputs_embeds_forward():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=512)
    model = GPT2LMHeadModel(cfg)

    B, T = 2, 10
    inputs_embeds = torch.randn(B, T, 64)

    logits, _ = model(inputs_embeds=inputs_embeds)
    assert logits.shape == (B, T, 100)


def test_resize_token_embeddings():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(cfg)

    old_wte_0 = model.transformer.wte.weight[0].clone()

    # Expand vocab to 105 tokens (+5 new tokens for 12, 16, 24, 32 bars)
    model.resize_token_embeddings(105)

    assert model.cfg.vocab_size == 105
    assert model.transformer.wte.weight.shape == (105, 64)
    assert model.lm_head.weight.shape == (105, 64)
    # Existing weights preserved
    assert torch.allclose(model.transformer.wte.weight[0], old_wte_0)
