"""Padding collators for the style-conditioning prototype's two training
regimes (contrastive pretraining, joint conditioning).

Both filter/re-index raw LM token ids -> StyleVocab local ids here, not in
MidiGPTDataset -- the dataset stays vocab/style-oblivious (its
"style_ref_mask"/"anchor_mask"/"positive_mask" fields already flag which raw
ids are expressive; the collator is where those get turned into the compact
local-id space the encoder actually consumes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from style_vocab import StyleVocab


def _pad(seqs: list[list[int]]) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1) or 1
    out = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _key_mask(seqs: list[list[int]]) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1) or 1
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        mask[i, : len(s)] = True
    return mask


@dataclass
class ContrastiveCollator:
    """Pads (anchor, positive) expressive-only local-id sequences from
    MidiGPTDataset(style_pretrain_mode=True) samples to the batch max
    length."""

    style_vocab: StyleVocab
    max_len: int = 512

    def __call__(self, batch: list[dict]) -> dict:
        anchors = [
            self.style_vocab.to_local_seq(item["anchor_ids"])[: self.max_len] for item in batch
        ]
        positives = [
            self.style_vocab.to_local_seq(item["positive_ids"])[: self.max_len] for item in batch
        ]
        return {
            "anchor_ids": _pad(anchors),
            "anchor_key_mask": _key_mask(anchors),
            "positive_ids": _pad(positives),
            "positive_key_mask": _key_mask(positives),
        }


@dataclass
class JointConditioningCollator:
    """Pads MidiGPTDataset's normal humanize-shaped samples (input_ids/
    labels/style_ref_mask, from _encode_one) for the soft-prefix training
    loop: main sequence padded with -100 labels as usual, plus the
    reference-segment expressive-token subsequence (derived from
    style_ref_mask) padded separately as its own shorter tensor.
    """

    style_vocab: StyleVocab
    max_seq_len: int = 2048
    max_style_len: int = 512
    pad_value: int = -100

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [
            torch.tensor(item["input_ids"][: self.max_seq_len], dtype=torch.long) for item in batch
        ]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), self.pad_value, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            seq_len = ids.size(0)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = ids

        style_seqs: list[list[int]] = []
        has_style = torch.zeros(len(batch), dtype=torch.bool)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            mask = item["style_ref_mask"]
            local_seq = [
                self.style_vocab.to_local(t) for t, m in zip(ids, mask, strict=True) if m
            ][: self.max_style_len]
            style_seqs.append(local_seq)
            has_style[i] = bool(local_seq)

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "style_ids": _pad(style_seqs),
            "style_key_mask": _key_mask(style_seqs),
            # Samples with no available reference segment (see
            # MidiGPTDataset._encode_one's fallback) get no soft-prefix at
            # all -- the training loop must gate on this per-sample, not
            # just rely on an all-False key_mask (an all-masked-out key_mask
            # would make the pooling attention in StyleEncoder softmax over
            # -inf everywhere, i.e. NaN).
            "has_style": has_style,
        }
