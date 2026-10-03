"""Audio-conditioned datasets — inherit from MidiGPTDatasetBase, add one step."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

log = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset

import midigpt._core as _core
from midigpt._types import Score
from midigpt.audio.render_cache import AudioRenderCache
from midigpt.audio.synth import FastAudioSynthesizer, get_bar_intervals, safe_pitch_shift
from midigpt.augmentation.score_window import select_window_dispatch
from midigpt.training.dataset import MidiGPTDatasetBase
from midigpt.tokenizer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_audio_tracks(
    num_tracks: int,
    is_infill: bool,
    audio_track_prob: float,
    single_target_prob: float = 0.1,
) -> list[bool]:
    """Decide which tracks are audio context (True) vs MIDI target (False).

    ALWAYS guarantees >=1 audio context AND >=1 MIDI target — the model must
    have something to generate in every example, in every mode (not just
    infill; the old code only forced a target when infilling, so autoregressive
    draws could end up all-audio with nothing to predict).

    With probability ``single_target_prob``, force the hardest accompaniment
    case: exactly ONE MIDI target and every other track audio context.

    ``is_infill`` is accepted for call-site compatibility but no longer gates
    the target guarantee.
    """
    if num_tracks >= 2 and random.random() < single_target_prob:
        track_is_audio = [True] * num_tracks
        track_is_audio[random.randrange(num_tracks)] = False  # exactly 1 MIDI target
        return track_is_audio
    track_is_audio = [random.random() < audio_track_prob for _ in range(num_tracks)]
    if not any(track_is_audio):
        track_is_audio[random.randrange(num_tracks)] = True   # >=1 audio context
    if all(track_is_audio):
        track_is_audio[random.randrange(num_tracks)] = False  # >=1 MIDI target (always)
    return track_is_audio


def _crop_and_shift_cached_render(
    full_wav: torch.Tensor,
    orig_bar_intervals: list[tuple[float, float]],
    start_bar: int,
    num_bars: int,
    semitones: int,
    is_drum: bool,
    sample_rate: int,
) -> torch.Tensor | None:
    """Crop a cached full-track render to a bar window and pitch-shift it.

    The cache stores one untransposed render per (score, track, soundfont);
    this adapts it to a specific (start_bar, num_bars, semitones) draw, the
    same way ``_extract_slakh_audio_features`` adapts pre-rendered stems.
    Returns None if the requested bar range falls outside what was cached.
    """
    end_bar = start_bar + num_bars - 1
    if not (0 <= start_bar < len(orig_bar_intervals) and 0 <= end_bar < len(orig_bar_intervals)):
        return None

    start_sample = round(orig_bar_intervals[start_bar][0] * sample_rate)
    end_sample = round(orig_bar_intervals[end_bar][1] * sample_rate)
    cropped = full_wav[start_sample:end_sample]
    if cropped.numel() == 0:
        return None

    if semitones != 0 and not is_drum:
        cropped = safe_pitch_shift(cropped.unsqueeze(0), sample_rate, semitones).squeeze(0)
    return cropped


def _extract_audio_features(
    score_slice: Score,
    track_is_audio: list[bool],
    infill_bar_indices: set[int],
    synthesizer: FastAudioSynthesizer,
    sample_rate: int,
    audio_cache: AudioRenderCache | None = None,
    soundfont_names: list[str] | None = None,
    score_id: str | None = None,
    start_bar: int = 0,
    semitones: int = 0,
    original_track_indices: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[int], list[tuple[float, float]], list[tuple[int, int]]]:
    """Synthesize raw audio for each audio track (CPU-side; no feature extraction here).

    When *audio_cache* is given (alongside *soundfont_names* and *score_id*),
    each audio track first tries a random soundfont's pre-synthesized
    full-track render from the cache — the same ``AudioRenderCache`` that
    ``scripts/presynthesize_audio_cache.py`` populates offline — cropping it
    to this window's bars and pitch-shifting it to match ``semitones``. Any
    cache miss (untranspiled score, uncached soundfont, out-of-range bars)
    falls back to live synthesis via *synthesizer*, exactly as when no cache
    is configured at all.

    Feature extraction is deliberately NOT done here: this function runs inside
    forked DataLoader worker processes, and the audio extractor is a real
    GPU-resident model in production — sharing a CUDA-resident module across a
    fork is exactly the bug that broke an earlier run (see
    AudioConditionedLightningModule._compute_loss for where extraction now
    happens instead, batched, on GPU, right before the adapter). Workers stay
    CPU-only and hand back raw waveforms plus enough metadata (which window
    track each waveform is, and the window's bar intervals) to slice/extract
    from later.

    Returns (audio_waveforms, audio_track_indices, bar_intervals, audio_bar_refs).
    ``audio_waveforms[i]`` is the raw waveform for window-track
    ``audio_track_indices[i]``; ``bar_intervals`` is shared by every track in
    this window (all tracks share the same bar timing).
    """
    bar_intervals = get_bar_intervals(score_slice)
    audio_waveforms: list[torch.Tensor] = []
    audio_track_indices: list[int] = []
    audio_bar_refs: list[tuple[int, int]] = []

    for t_idx in range(len(track_is_audio)):
        if not track_is_audio[t_idx]:
            continue

        wav = None
        if audio_cache is not None and soundfont_names and score_id is not None:
            orig_t_idx = (
                original_track_indices[t_idx] if original_track_indices is not None else t_idx
            )
            sf2_name = random.choice(soundfont_names)
            full_wav = audio_cache.load(score_id, orig_t_idx, sf2_name)
            orig_bar_intervals = audio_cache.load_bar_intervals(score_id)
            if full_wav is not None and orig_bar_intervals is not None:
                num_bars_window = len(score_slice.tracks[t_idx].bars)
                is_drum = score_slice.tracks[t_idx].track_type == "drum"
                wav = _crop_and_shift_cached_render(
                    full_wav, orig_bar_intervals, start_bar, num_bars_window,
                    semitones, is_drum, sample_rate,
                )

        if wav is None:
            wav = synthesizer.render_track(score_slice, t_idx)

        audio_waveforms.append(wav)
        audio_track_indices.append(t_idx)
        for b_idx in range(len(bar_intervals)):
            if b_idx not in infill_bar_indices:
                audio_bar_refs.append((t_idx, b_idx))

    return audio_waveforms, audio_track_indices, bar_intervals, audio_bar_refs


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
    sample_rate: int,
    original_track_indices: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[int], list[tuple[float, float]], list[tuple[int, int]]]:
    """Crop + pitch-shift pre-rendered Slakh stems (CPU-side; no feature extraction here).

    ``start_bar`` locates this window within the original, unwindowed score
    that ``source_bar_intervals`` was computed from — it is not always 0.
    ``stem_sources`` is indexed by the *original* score's track order, so
    ``original_track_indices`` (window position -> original track index) is
    needed to look up the right stem once window track selection has
    reordered or subset the tracks.

    Feature extraction happens later, batched, on GPU — see
    ``_extract_audio_features`` above for why.

    Returns (audio_waveforms, audio_track_indices, bar_intervals, audio_bar_refs).
    """
    import torchaudio.functional as AF

    bar_intervals = get_bar_intervals(score_slice)
    source_start_sec = source_bar_intervals[start_bar][0]
    source_end_sec = source_bar_intervals[start_bar + num_bars - 1][1]

    audio_waveforms: list[torch.Tensor] = []
    audio_track_indices: list[int] = []
    audio_bar_refs: list[tuple[int, int]] = []

    for t_idx in range(len(track_is_audio)):
        if not track_is_audio[t_idx]:
            continue
        orig_t_idx = original_track_indices[t_idx] if original_track_indices is not None else t_idx
        stem = stem_sources[orig_t_idx]
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
            wav = safe_pitch_shift(wav.unsqueeze(0), sample_rate, semitones).squeeze(0)

        audio_waveforms.append(wav)
        audio_track_indices.append(t_idx)
        for b_idx in range(len(bar_intervals)):
            if b_idx not in infill_bar_indices:
                audio_bar_refs.append((t_idx, b_idx))

    return audio_waveforms, audio_track_indices, bar_intervals, audio_bar_refs


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
        scores: list[Score] | None,
        tokenizer: Tokenizer,
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
        score_ids: list[str] | None = None,
        audio_cache: AudioRenderCache | None = None,
        raw_rows: Any | None = None,
    ):
        # raw_rows (an Arrow-backed datasets.Dataset with a "music" bytes
        # column) is the lazy alternative to `scores`: a filtered GigaMIDI
        # split averages ~2,690 notes / ~1,255 bars per row (min_tracks>=2
        # selects much larger multi-track excerpts than an unfiltered row),
        # so eagerly parsing all ~573k rows into live Score objects before
        # training starts measured at ~430GB RSS in the loading process alone
        # -- see amber OOMs 2026-08-27 through 2026-08-29. When raw_rows is
        # set, each Score is parsed on demand in _load_score, inside the
        # (already forked, already small-batch) DataLoader worker that
        # consumes it, so at most a few in-flight batches are ever live.
        self.scores = scores
        self.raw_rows = raw_rows
        self.audio_track_prob = audio_track_prob
        self.sample_rate = sample_rate
        self.synthesizer = synthesizer or FastAudioSynthesizer(sample_rate=sample_rate)
        # score_ids (e.g. the parquet `md5` column) are the cache's content
        # key; audio_cache is the same AudioRenderCache that
        # scripts/presynthesize_audio_cache.py populates offline — there is
        # no separate in-memory render cache anymore. Falls back to live
        # per-window synthesis whenever either is left unset, or on a miss.
        self.score_ids = score_ids
        self.audio_cache = audio_cache

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
        return len(self.raw_rows) if self.raw_rows is not None else len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        try:
            return self._getitem_inner(idx)
        except Exception:
            log.debug("Sample %d skipped in AudioTransitionDataset", idx, exc_info=True)
            return None

    def _load_score(self, idx: int) -> Score:
        if self.raw_rows is not None:
            return Score.from_bytes(self.raw_rows[idx]["music"])
        return self.scores[idx]

    def prepare_score(
        self, score: Score, is_infill: bool, infill_bar_indices: set[int], idx: int,
        *, semitones: int = 0, start_bar: int = 0, original_track_indices: list[int] | None = None,
    ) -> tuple[Score, set[int], dict]:
        num_tracks = len(score.tracks)
        track_is_audio = _assign_audio_tracks(num_tracks, is_infill, self.audio_track_prob)

        score_id = self.score_ids[idx] if self.score_ids is not None else str(idx)
        soundfont_names = getattr(self.synthesizer, "soundfont_names", None)

        audio_waveforms, audio_track_indices, bar_intervals, refs = _extract_audio_features(
            score, track_is_audio, infill_bar_indices,
            self.synthesizer, self.sample_rate,
            audio_cache=self.audio_cache,
            soundfont_names=soundfont_names,
            score_id=score_id,
            start_bar=start_bar,
            semitones=semitones,
            original_track_indices=original_track_indices,
        )

        for t_idx in range(num_tracks):
            if track_is_audio[t_idx]:
                score.tracks[t_idx].track_type = "audio"

        return score, infill_bar_indices, {
            "has_audio_tracks": True,
            "audio_waveforms": audio_waveforms,
            "audio_track_indices": audio_track_indices,
            "bar_intervals": bar_intervals,
            "audio_bar_refs": refs,
        }

    def _encode_hook(self, token_ids: list[int], is_infill: bool, idx: int, **extra) -> dict:
        audio_bar_positions = _find_audio_bar_positions(
            token_ids, extra.get("audio_bar_refs", []), self._tokenizer._vocab,
        )
        # MidiGPTDatasetBase._getitem_inner only merges what _encode_hook
        # returns (not prepare_score's `extra` directly) into the final item
        # dict, so audio_waveforms/audio_track_indices/bar_intervals/audio_bar_refs
        # must be re-surfaced here or AudioConditionedCollator's
        # x.get("audio_waveforms", []) silently sees an empty default —
        # audio conditioning trains as a no-op with no error.
        return {
            "audio_bar_positions": audio_bar_positions,
            "audio_waveforms": extra.get("audio_waveforms", []),
            "audio_track_indices": extra.get("audio_track_indices", []),
            "bar_intervals": extra.get("bar_intervals", []),
            "audio_bar_refs": extra.get("audio_bar_refs", []),
            "has_audio_tracks": extra.get("has_audio_tracks", False),
        }


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
        self.audio_track_prob = audio_track_prob
        self.sample_rate = sample_rate

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
            transpose_range=transpose_range,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        try:
            return self._getitem_inner(idx)
        except Exception:
            log.debug("Sample %d skipped in SlakhPairedStemDataset", idx, exc_info=True)
            return None

    def _load_score(self, idx: int) -> Score:
        return self.samples[idx]["score"]

    def prepare_score(
        self, score: Score, is_infill: bool, infill_bar_indices: set[int], idx: int,
        *, semitones: int = 0, start_bar: int = 0, original_track_indices: list[int] | None = None,
    ) -> tuple[Score, set[int], dict]:
        item = self.samples[idx]
        stem_sources = item["audio_stems"]
        # Bar intervals of the *original*, unwindowed per-sample score — the
        # stems are rendered against that full excerpt, not this window.
        source_bar_intervals = get_bar_intervals(item["score"])
        num_tracks = min(len(score.tracks), len(stem_sources))

        num_bars = max((len(t.bars) for t in score.tracks), default=0)
        track_is_audio = _assign_audio_tracks(num_tracks, is_infill, self.audio_track_prob)

        audio_waveforms, audio_track_indices, bar_intervals, refs = _extract_slakh_audio_features(
            score, track_is_audio, infill_bar_indices,
            stem_sources, source_bar_intervals,
            start_bar=start_bar, num_bars=num_bars, semitones=semitones,
            sample_rate=self.sample_rate,
            original_track_indices=original_track_indices,
        )

        for t_idx in range(num_tracks):
            if track_is_audio[t_idx]:
                score.tracks[t_idx].track_type = "audio"

        return score, infill_bar_indices, {
            "has_audio_tracks": True,
            "audio_waveforms": audio_waveforms,
            "audio_track_indices": audio_track_indices,
            "bar_intervals": bar_intervals,
            "audio_bar_refs": refs,
        }

    def _encode_hook(self, token_ids: list[int], is_infill: bool, idx: int, **extra) -> dict:
        audio_bar_positions = _find_audio_bar_positions(
            token_ids, extra.get("audio_bar_refs", []), self._tokenizer._vocab,
        )
        # MidiGPTDatasetBase._getitem_inner only merges what _encode_hook
        # returns (not prepare_score's `extra` directly) into the final item
        # dict, so audio_waveforms/audio_track_indices/bar_intervals/audio_bar_refs
        # must be re-surfaced here or AudioConditionedCollator's
        # x.get("audio_waveforms", []) silently sees an empty default —
        # audio conditioning trains as a no-op with no error.
        return {
            "audio_bar_positions": audio_bar_positions,
            "audio_waveforms": extra.get("audio_waveforms", []),
            "audio_track_indices": extra.get("audio_track_indices", []),
            "bar_intervals": extra.get("bar_intervals", []),
            "audio_bar_refs": extra.get("audio_bar_refs", []),
            "has_audio_tracks": extra.get("has_audio_tracks", False),
        }


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
