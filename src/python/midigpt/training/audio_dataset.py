"""Audio-conditioned datasets — inherit from MidiGPTDatasetBase, add one step."""

from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

import midigpt._core as _core
from midigpt._types import Score
from midigpt.audio.base import BaseAudioExtractor
from midigpt.audio.synth import FastAudioSynthesizer, get_bar_intervals
from midigpt.augmentation.score_window import select_window_dispatch
from midigpt.augmentation.transpose import Transpose
from midigpt.training.dataset import MidiGPTDatasetBase
from midigpt.tokenizer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_audio_tracks(
    num_tracks: int,
    is_infill: bool,
    audio_track_prob: float,
) -> list[bool]:
    """Decide which tracks are audio context (True) vs MIDI (False).

    Ensures at least one audio track and, when infilling, at least one MIDI
    track (the generation target).
    """
    track_is_audio = [random.random() < audio_track_prob for _ in range(num_tracks)]
    if not any(track_is_audio):
        track_is_audio[random.randrange(num_tracks)] = True
    if is_infill and all(track_is_audio):
        track_is_audio[random.randrange(num_tracks)] = False
    return track_is_audio


def _extract_audio_features(
    score_slice: Score,
    track_is_audio: list[bool],
    infill_bar_indices: set[int],
    audio_extractor: BaseAudioExtractor,
    synthesizer: FastAudioSynthesizer,
    sample_rate: int,
    render_cache: dict[tuple[int, int], torch.Tensor] | None = None,
    score_idx: int = -1,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    """Synthesize audio for each audio track and extract features.

    If *render_cache* is provided, full-track waveforms are cached by
    ``(score_idx, track_idx)`` so different window selections on the same
    score reuse the cached synthesis.

    Returns (context_bar_frames, context_track_frames, audio_bar_refs).
    """
    bar_intervals = get_bar_intervals(score_slice)
    context_bar_frames: list[torch.Tensor] = []
    context_track_frames: list[torch.Tensor] = []
    audio_bar_refs: list[tuple[int, int]] = []

    with torch.no_grad():
        for t_idx in range(len(track_is_audio)):
            if not track_is_audio[t_idx]:
                continue

            cache_key = (score_idx, t_idx)
            if render_cache is not None and cache_key in render_cache:
                wav = render_cache[cache_key]
            else:
                wav = synthesizer.render_track(score_slice, t_idx)
                if render_cache is not None:
                    render_cache[cache_key] = wav

            t_frames = audio_extractor.extract_track_frames(wav, sample_rate=sample_rate)
            b_frames_list = audio_extractor.extract_bar_frames(
                wav, sample_rate=sample_rate, bar_intervals=bar_intervals,
            )
            for b_idx, b_frames in enumerate(b_frames_list):
                if b_idx not in infill_bar_indices:
                    context_bar_frames.append(b_frames)
                    context_track_frames.append(t_frames)
                    audio_bar_refs.append((t_idx, b_idx))

    return context_bar_frames, context_track_frames, audio_bar_refs


def _find_audio_bar_positions(
    token_ids: list[int],
    audio_bar_refs: list[tuple[int, int]],
    vocab: Any,
) -> list[int]:
    """Find token positions of [AudioContext] for bars in audio_bar_refs."""
    if not audio_bar_refs:
        return []
    audio_ctx_token = vocab.encode_val(_core.TokenType.AudioContext, 0)
    track_token_ids = {
        vocab.encode_val(_core.TokenType.Track, v)
        for v in range(vocab.domain_size(_core.TokenType.Track))
    }
    ref_set = set(audio_bar_refs)
    positions: list[int] = []
    cur_track, cur_bar = -1, -1
    for pos, tid in enumerate(token_ids):
        if tid in track_token_ids:
            cur_track += 1
            cur_bar = -1
        elif tid == audio_ctx_token:
            if (cur_track, cur_bar + 1) in ref_set:
                positions.append(pos)
            cur_bar += 1
    return positions


def _extract_slakh_audio_features(
    score_slice: Score,
    track_is_audio: list[bool],
    infill_bar_indices: set[int],
    stem_sources: list[Any],
    source_bar_intervals: list[tuple[float, float]],
    start_bar: int,
    num_bars: int,
    semitones: int,
    audio_extractor: BaseAudioExtractor,
    sample_rate: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    """Extract features from pre-rendered Slakh stems (cropped + pitch-shifted)."""
    import torchaudio.functional as AF

    bar_intervals = get_bar_intervals(score_slice)
    source_start_sec = source_bar_intervals[start_bar][0]
    source_end_sec = source_bar_intervals[start_bar + num_bars - 1][1]

    context_bar_frames: list[torch.Tensor] = []
    context_track_frames: list[torch.Tensor] = []
    audio_bar_refs: list[tuple[int, int]] = []

    with torch.no_grad():
        for t_idx in range(len(track_is_audio)):
            if not track_is_audio[t_idx]:
                continue
            stem = stem_sources[t_idx]
            if isinstance(stem, str):
                import torchaudio
                wav, sr = torchaudio.load(stem)
                if sr != sample_rate:
                    wav = AF.resample(wav, orig_freq=sr, new_freq=sample_rate)
                wav = wav.mean(dim=0)
            else:
                wav = stem

            start_sample = round(source_start_sec * sample_rate)
            end_sample = round(source_end_sec * sample_rate)
            wav = wav[start_sample:end_sample]

            if semitones != 0 and score_slice.tracks[t_idx].track_type != "drum":
                wav = AF.pitch_shift(wav.unsqueeze(0), sample_rate, n_steps=semitones).squeeze(0)

            t_frames = audio_extractor.extract_track_frames(wav, sample_rate=sample_rate)
            b_frames_list = audio_extractor.extract_bar_frames(
                wav, sample_rate=sample_rate, bar_intervals=bar_intervals,
            )
            for b_idx, b_frames in enumerate(b_frames_list):
                if b_idx not in infill_bar_indices:
                    context_bar_frames.append(b_frames)
                    context_track_frames.append(t_frames)
                    audio_bar_refs.append((t_idx, b_idx))

    return context_bar_frames, context_track_frames, audio_bar_refs


# ---------------------------------------------------------------------------
# AudioTransitionDataset — in-memory scores + on-the-fly synthesis
# ---------------------------------------------------------------------------


class AudioTransitionDataset(MidiGPTDatasetBase):
    """Dataset for training audio-conditioned transition infilling and autoregression.

    Overrides only the audio-specific steps: track assignment, synthesis,
    feature extraction, and audio-bar position discovery.
    """

    def __init__(
        self,
        scores: list[Score],
        tokenizer: Tokenizer,
        audio_extractor: BaseAudioExtractor,
        num_bars_choices: Sequence[int] = tuple(range(4, 68, 4)),
        infill_prob: float = 0.5,
        infill_bars_range: tuple[int, int] = (2, 4),
        audio_track_prob: float = 0.7,
        min_tracks: int = 2,
        max_tracks: int = 12,
        min_fill_ratio: float = 0.75,
        window_selection: str = "fixed",
        temperature: float = 1.0,
        sample_rate: int = 16000,
        synthesizer: FastAudioSynthesizer | None = None,
        max_seq_len: int = 2048,
    ):
        self.scores = scores
        self.audio_extractor = audio_extractor
        self.audio_track_prob = audio_track_prob
        self.sample_rate = sample_rate
        self.synthesizer = synthesizer or FastAudioSynthesizer(sample_rate=sample_rate)
        self._render_cache: dict[tuple[int, int], torch.Tensor] = {}

        super().__init__(
            tokenizer=tokenizer,
            num_bars_choices=list(num_bars_choices),
            infill_prob=infill_prob,
            infill_bar_fraction=0.5,
            max_seq_len=max_seq_len,
            max_tracks=max_tracks,
            min_tracks=min_tracks,
            min_fill_ratio=min_fill_ratio,
            window_selection=window_selection,
            temperature=temperature,
        )

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        try:
            return self._getitem_inner(idx)
        except Exception:
            return None

    def _load_score(self, idx: int) -> Score:
        return self.scores[idx]

    def prepare_score(
        self, score: Score, is_infill: bool, infill_bar_indices: set[int], idx: int,
    ) -> tuple[Score, set[int], dict]:
        num_tracks = len(score.tracks)
        track_is_audio = _assign_audio_tracks(num_tracks, is_infill, self.audio_track_prob)

        ctx_bar, ctx_track, refs = _extract_audio_features(
            score, track_is_audio, infill_bar_indices,
            self.audio_extractor, self.synthesizer, self.sample_rate,
            render_cache=self._render_cache, score_idx=idx,
        )

        for t_idx in range(num_tracks):
            if track_is_audio[t_idx]:
                score.tracks[t_idx].track_type = "audio"

        return score, infill_bar_indices, {
            "has_audio_tracks": True,
            "context_bar_frames": ctx_bar,
            "context_track_frames": ctx_track,
            "audio_bar_refs": refs,
        }

    def _encode_hook(self, token_ids: list[int], is_infill: bool, idx: int, **extra) -> dict:
        audio_bar_positions = _find_audio_bar_positions(
            token_ids, extra.get("audio_bar_refs", []), self._tokenizer._vocab,
        )
        return {"audio_bar_positions": audio_bar_positions}


# ---------------------------------------------------------------------------
# SlakhPairedStemDataset — pre-rendered stems + ground-truth MIDI
# ---------------------------------------------------------------------------


class SlakhPairedStemDataset(MidiGPTDatasetBase):
    """Dataset for pre-rendered multi-track audio stems + ground truth MIDI (Slakh2100 format).

    Differences from AudioTransitionDataset:
      - Stems come from disk (not synthesized).
      - Pitch augmentation is applied to audio stems as well as MIDI.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],  # {"score": Score, "audio_stems": list[str | Tensor]}
        tokenizer: Tokenizer,
        audio_extractor: BaseAudioExtractor,
        num_bars_choices: Sequence[int] = tuple(range(4, 68, 4)),
        infill_prob: float = 0.5,
        infill_bars_range: tuple[int, int] = (2, 4),
        audio_track_prob: float = 0.7,
        min_tracks: int = 2,
        max_tracks: int = 12,
        min_fill_ratio: float = 0.75,
        window_selection: str = "fixed",
        temperature: float = 1.0,
        transpose_range: tuple[int, int] = (-6, 6),
        sample_rate: int = 16000,
        max_seq_len: int = 2048,
    ):
        self.samples = samples
        self.audio_extractor = audio_extractor
        self.audio_track_prob = audio_track_prob
        self.sample_rate = sample_rate
        self.transpose_range = transpose_range

        super().__init__(
            tokenizer=tokenizer,
            num_bars_choices=list(num_bars_choices),
            infill_prob=infill_prob,
            infill_bar_fraction=0.5,
            max_seq_len=max_seq_len,
            max_tracks=max_tracks,
            min_tracks=min_tracks,
            min_fill_ratio=min_fill_ratio,
            window_selection=window_selection,
            temperature=temperature,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        try:
            return self._getitem_inner(idx)
        except Exception:
            return None

    def _load_score(self, idx: int) -> Score:
        return self.samples[idx]["score"]

    def prepare_score(
        self, score: Score, is_infill: bool, infill_bar_indices: set[int], idx: int,
    ) -> tuple[Score, set[int], dict]:
        item = self.samples[idx]
        stem_sources = item["audio_stems"]
        source_bar_intervals = get_bar_intervals(score)
        num_tracks = min(len(score.tracks), len(stem_sources))

        # Pitch augmentation (synchronous MIDI + Audio)
        semitones = random.randint(self.transpose_range[0], self.transpose_range[1])
        if semitones != 0:
            score = Transpose(semitones)(score)

        num_bars = max((len(t.bars) for t in score.tracks), default=0)
        track_is_audio = _assign_audio_tracks(num_tracks, is_infill, self.audio_track_prob)

        ctx_bar, ctx_track, refs = _extract_slakh_audio_features(
            score, track_is_audio, infill_bar_indices,
            stem_sources, source_bar_intervals,
            start_bar=0, num_bars=num_bars, semitones=semitones,
            audio_extractor=self.audio_extractor, sample_rate=self.sample_rate,
        )

        for t_idx in range(num_tracks):
            if track_is_audio[t_idx]:
                score.tracks[t_idx].track_type = "audio"

        return score, infill_bar_indices, {
            "has_audio_tracks": True,
            "context_bar_frames": ctx_bar,
            "context_track_frames": ctx_track,
            "audio_bar_refs": refs,
        }

    def _encode_hook(self, token_ids: list[int], is_infill: bool, idx: int, **extra) -> dict:
        audio_bar_positions = _find_audio_bar_positions(
            token_ids, extra.get("audio_bar_refs", []), self._tokenizer._vocab,
        )
        return {"audio_bar_positions": audio_bar_positions}


# ---------------------------------------------------------------------------
# 50/50 mixing wrapper — randomly picks MIDI-only vs audio per sample
# ---------------------------------------------------------------------------


class MixedMidiGPTDataset(Dataset):
    """Wraps a MIDI-only dataset and an audio dataset, picking one at random.

    Each ``__getitem__`` call flips a coin (50/50) to decide whether the
    sample comes from the MIDI-only dataset or the audio dataset.  This
    gives the model a balanced mix of pure-MIDI and audio-conditioned
    examples without duplicating data.

    The collator must handle both output formats (audio fields missing
    for MIDI-only examples).
    """

    def __init__(self, midi_dataset: Dataset, audio_dataset: Dataset):
        assert len(midi_dataset) > 0 and len(audio_dataset) > 0
        self.midi_dataset = midi_dataset
        self.audio_dataset = audio_dataset

    def __len__(self) -> int:
        return max(len(self.midi_dataset), len(self.audio_dataset))

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        if random.random() < 0.5:
            midi_idx = idx % len(self.midi_dataset)
            return self.midi_dataset[midi_idx]
        else:
            audio_idx = idx % len(self.audio_dataset)
            return self.audio_dataset[audio_idx]
