"""Audio synthesis and timing utilities for MIDI Scores."""

from __future__ import annotations

import glob
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from midigpt._types import Score, Track


def safe_pitch_shift(
    waveform: torch.Tensor, sample_rate: int, n_steps: int, work_rate: int = 4000,
) -> torch.Tensor:
    """Pitch-shift by ``n_steps`` semitones without torchaudio's resample blowup.

    ``torchaudio.functional.pitch_shift`` resamples internally via
    ``torchaudio.functional.resample``, which picks exact integer up/down
    sampling factors from ``gcd(orig_freq, new_freq)``. A semitone ratio like
    ``2**(n_steps/12)`` almost always lands on a target rate coprime with a
    real sample rate (e.g. 16000 -> 19029 Hz has gcd=1), so the resample
    kernel scales to the *sample rate itself* — a single call on an ordinary
    clip can allocate several GB. Doing the same pitch_shift call on audio
    downsampled to ``work_rate`` first (an exact, cheap ratio in and out,
    e.g. 16000/4000 = 4) bounds the internal kernel to ``work_rate``-scale
    instead, at the cost of losing content above ``work_rate / 2`` Hz during
    the shift step — acceptable for a conditioning signal, not for playback.
    """
    if n_steps == 0:
        return waveform

    orig_len = waveform.shape[-1]
    import torchaudio.functional as AF

    down = AF.resample(waveform, orig_freq=sample_rate, new_freq=work_rate)
    shifted = AF.pitch_shift(down, work_rate, n_steps=n_steps)
    up = AF.resample(shifted, orig_freq=work_rate, new_freq=sample_rate)

    if up.shape[-1] < orig_len:
        up = torch.nn.functional.pad(up, (0, orig_len - up.shape[-1]))
    else:
        up = up[..., :orig_len]
    return up


def get_bar_intervals(score: Score) -> list[tuple[float, float]]:
    """Compute exact (start_sec, end_sec) time intervals for each bar in the score.

    Tempo is in microseconds per quarter note (e.g. 500,000 = 120 BPM).
    Resolution is ticks per quarter note.
    """
    sec_per_quarter = score.tempo / 1_000_000.0
    sec_per_tick = sec_per_quarter / max(1, score.resolution)

    num_bars = max((len(t.bars) for t in score.tracks), default=0)
    intervals: list[tuple[float, float]] = []

    current_sec = 0.0
    for b_idx in range(num_bars):
        # Determine bar duration in ticks from time signature of the bar
        ts_num, ts_den = 4, 4
        for t in score.tracks:
            if b_idx < len(t.bars):
                ts_num = t.bars[b_idx].ts_numerator
                ts_den = t.bars[b_idx].ts_denominator
                break

        ticks_per_bar = int(ts_num * (score.resolution * 4.0 / max(1, ts_den)))
        bar_duration_sec = ticks_per_bar * sec_per_tick
        intervals.append((current_sec, current_sec + bar_duration_sec))
        current_sec += bar_duration_sec

    return intervals


class FastAudioSynthesizer:
    """Ultra-fast, deterministic MIDI-to-waveform synthesizer for training and augmentation.

    Renders melodic and drum tracks into clean audio waveforms using harmonic
    oscillators and ADSR envelopes.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def midi_to_hz(self, midi_pitch: int) -> float:
        return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))

    def render_track(self, score: Score, track_idx: int) -> torch.Tensor:
        """Render a single track into a mono audio waveform."""
        if track_idx >= len(score.tracks):
            return torch.zeros(int(self.sample_rate * 1.0))

        track = score.tracks[track_idx]
        intervals = get_bar_intervals(score)
        total_duration = intervals[-1][1] if intervals else 1.0
        total_samples = max(100, int(total_duration * self.sample_rate))

        waveform = torch.zeros(total_samples, dtype=torch.float32)
        sec_per_quarter = score.tempo / 1_000_000.0
        sec_per_tick = sec_per_quarter / max(1, score.resolution)

        is_drum = track.track_type == "drum"

        for b_idx, bar in enumerate(track.bars):
            if b_idx >= len(intervals):
                break
            bar_start_sec = intervals[b_idx][0]

            for note in bar.notes:
                note_onset_sec = bar_start_sec + (note.onset_ticks * sec_per_tick)
                note_dur_sec = max(0.02, note.duration_ticks * sec_per_tick)
                vel_norm = max(0.1, min(1.0, note.velocity / 127.0))

                start_sample = int(note_onset_sec * self.sample_rate)
                num_samples = int(note_dur_sec * self.sample_rate)

                if start_sample >= total_samples or num_samples <= 0:
                    continue

                end_sample = min(total_samples, start_sample + num_samples)
                actual_len = end_sample - start_sample

                t = torch.linspace(0, actual_len / self.sample_rate, actual_len, dtype=torch.float32)

                if is_drum:
                    # Noise burst + low pitch kick/snare decay
                    noise = (torch.rand(actual_len) * 2.0 - 1.0) * 0.4
                    pitch_decay = torch.sin(2.0 * math.pi * 60.0 * t * torch.exp(-15.0 * t))
                    env = torch.exp(-20.0 * t)
                    sig = (pitch_decay + noise) * env * vel_norm
                else:
                    # Harmonic rich synth tone with ADSR envelope
                    freq = self.midi_to_hz(note.pitch)
                    # Fundamental + 2 harmonics
                    sig = (
                        torch.sin(2.0 * math.pi * freq * t)
                        + 0.5 * torch.sin(4.0 * math.pi * freq * t)
                        + 0.25 * torch.sin(6.0 * math.pi * freq * t)
                    )
                    # Attack-Decay-Sustain-Release envelope
                    attack_len = min(int(0.01 * self.sample_rate), actual_len // 4)
                    release_len = min(int(0.03 * self.sample_rate), actual_len // 4)
                    env = torch.ones(actual_len, dtype=torch.float32)

                    if attack_len > 0:
                        env[:attack_len] = torch.linspace(0.0, 1.0, attack_len)
                    if release_len > 0:
                        env[-release_len:] = torch.linspace(1.0, 0.0, release_len)

                    sig = sig * env * vel_norm * 0.3

                waveform[start_sample:end_sample] += sig

        # Normalize without clipping
        max_val = waveform.abs().max()
        if max_val > 0.99:
            waveform = waveform / (max_val + 1e-4) * 0.95

        return waveform

    def render_all(self, score: Score) -> torch.Tensor:
        """Render all tracks into a combined mix waveform."""
        rendered_tracks = [self.render_track(score, i) for i in range(len(score.tracks))]
        if not rendered_tracks:
            return torch.zeros(int(self.sample_rate * 1.0))

        max_len = max(t.shape[0] for t in rendered_tracks)
        mix = torch.zeros(max_len, dtype=torch.float32)
        for t in rendered_tracks:
            mix[: t.shape[0]] += t

        max_val = mix.abs().max()
        if max_val > 0.99:
            mix = mix / (max_val + 1e-4) * 0.95
        return mix


# ---------------------------------------------------------------------------
# SoundFont-based synthesizer (FluidSynth) — per-track random timbre
# ---------------------------------------------------------------------------


def _load_fluidsynth():
    """Lazy import of fluidsynth, raising a clear error if unavailable."""
    try:
        import fluidsynth as _fs
        return _fs
    except ImportError:
        raise ImportError(
            "pip install pyfluidsynth  (requires libfluidsynth: brew install fluid-synth)"
        ) from None


class SoundFontSynthesizer:
    """Renders MIDI tracks using real SoundFont samples via FluidSynth.

    Loads a pool of SF2 files from a directory.  On each ``render_track``
    call, a soundfont is randomly selected (per-track by default) so that
    the model sees diverse timbres during training.

    Parameters
    ----------
    soundfont_dir : str | Path
        Directory containing ``.sf2`` files.  All SF2 files in the
        directory are discovered automatically.
    sample_rate : int
        Output sample rate in Hz (FluidSynth renders at 44100 internally;
        audio is resampled to this rate).
    per_track : bool
        If True (default), each track gets its own randomly chosen
        soundfont.  If False, one soundfont is chosen per ``render_track``
        call and applied to all tracks.
    """

    def __init__(
        self,
        soundfont_dir: str | Path,
        sample_rate: int = 16000,
        per_track: bool = True,
    ):
        self.sample_rate = sample_rate
        self.per_track = per_track

        # Discover SF2 files (skip corrupt ones)
        soundfont_dir = Path(soundfont_dir)
        self._sf2_paths: list[Path] = []
        for p in sorted(soundfont_dir.glob("*.sf2")):
            if p.is_file() and p.stat().st_size > 1_000_000:  # skip tiny/corrupt files
                self._sf2_paths.append(p)
        if not self._sf2_paths:
            raise FileNotFoundError(
                f"No .sf2 files found in {soundfont_dir}. "
                f"Download soundfonts from https://miditoolbox.com/posts/best-free-general-midi-soundfonts-2026"
            )

        self._fs = _load_fluidsynth()
        self._synth: Optional[object] = None
        self._loaded_sf2: dict[int, int] = {}  # path_index -> fluidsynth sfont_id
        self._active_channel = 0

    def _ensure_synth(self):
        """Lazily create the FluidSynth synthesizer."""
        if self._synth is None:
            self._synth = self._fs.Synth(samplerate=float(44100))
            self._active_channel = 0

    def _load_sf2(self, path_index: int) -> int:
        """Load a SF2 file into FluidSynth, reusing if already loaded."""
        self._ensure_synth()
        if path_index not in self._loaded_sf2:
            sfont_id = self._synth.sfload(str(self._sf2_paths[path_index]))
            self._loaded_sf2[path_index] = sfont_id
        return self._loaded_sf2[path_index]

    def _pick_sf2_index(self) -> int:
        """Randomly select a soundfont from the pool."""
        return random.randrange(len(self._sf2_paths))

    @property
    def soundfont_names(self) -> list[str]:
        """Filenames of every soundfont discovered in the pool, in stable order."""
        return [p.name for p in self._sf2_paths]

    def render_track(self, score: Score, track_idx: int, sf2_index: int | None = None) -> torch.Tensor:
        """Render a single track into a mono audio waveform using SoundFont samples.

        Args:
            sf2_index: If given, render deterministically with this soundfont
                (index into ``self.soundfont_names``) instead of picking one
                at random. Used by the pre-synthesis cache builder to cover
                every soundfont for every track.
        """
        if track_idx >= len(score.tracks):
            return torch.zeros(int(self.sample_rate * 1.0))

        track = score.tracks[track_idx]
        intervals = get_bar_intervals(score)
        total_duration = intervals[-1][1] if intervals else 1.0
        total_samples_44k = max(100, int(total_duration * 44100))
        total_samples_out = max(100, int(total_duration * self.sample_rate))

        self._ensure_synth()
        synth = self._synth

        # Reset and pick soundfont
        synth.system_reset()
        channel = 0
        sf2_idx = sf2_index if sf2_index is not None else self._pick_sf2_index()
        sfont_id = self._load_sf2(sf2_idx)

        is_drum = track.track_type == "drum"
        if is_drum:
            # GM drums are always on channel 9
            synth.program_select(9, sfont_id, 128, 0)
            drum_channel = 9
        else:
            # Map instrument to GM program (simplified: use program 0 = piano)
            # TODO: read actual program from track attributes if available
            program = 0
            if hasattr(track, "attributes") and "_program" in track.attributes:
                program = track.attributes["_program"]
            synth.program_select(channel, sfont_id, 0, program)
            drum_channel = None

        # Render all notes for this track
        sec_per_quarter = score.tempo / 1_000_000.0
        sec_per_tick = sec_per_quarter / max(1, score.resolution)
        waveform_44k = np.zeros(total_samples_44k, dtype=np.float32)
        release_buf_samples = int(0.1 * 44100)  # 100ms release tail

        for b_idx, bar in enumerate(track.bars):
            if b_idx >= len(intervals):
                break
            bar_start_sec = intervals[b_idx][0]

            for note in bar.notes:
                onset_sec = bar_start_sec + (note.onset_ticks * sec_per_tick)
                dur_sec = max(0.02, note.duration_ticks * sec_per_tick)
                vel = max(1, min(127, note.velocity))
                ch = drum_channel if is_drum else channel

                start_44k = int(onset_sec * 44100)
                end_44k = start_44k + int(dur_sec * 44100)
                end_44k = min(end_44k, total_samples_44k)

                if start_44k >= total_samples_44k:
                    continue

                synth.noteon(ch, note.pitch, vel)

                # Render the note duration
                render_len = max(0, end_44k - start_44k)
                if render_len > 0:
                    buf = synth.get_samples(render_len)
                    left = buf[0::2].astype(np.float32) / 32768.0
                    right = buf[1::2].astype(np.float32) / 32768.0
                    mono = (left + right) * 0.5
                    waveform_44k[start_44k:start_44k + len(mono)] += mono

                synth.noteoff(ch, note.pitch)

                # Render the release tail (envelope decay after noteoff)
                tail_end = min(end_44k + release_buf_samples, total_samples_44k)
                tail_len = max(0, tail_end - end_44k)
                if tail_len > 0:
                    tail_buf = synth.get_samples(tail_len)
                    left = tail_buf[0::2].astype(np.float32) / 32768.0
                    right = tail_buf[1::2].astype(np.float32) / 32768.0
                    mono = (left + right) * 0.5
                    waveform_44k[end_44k:end_44k + len(mono)] += mono

        # Resample from 44100 → target sample_rate
        if self.sample_rate != 44100:
            import torchaudio.functional as AF
            wav = torch.from_numpy(waveform_44k).unsqueeze(0)  # (1, N)
            wav = AF.resample(wav, orig_freq=44100, new_freq=self.sample_rate)
            waveform = wav.squeeze(0)
        else:
            waveform = torch.from_numpy(waveform_44k)

        # Normalize
        max_val = waveform.abs().max()
        if max_val > 0.99:
            waveform = waveform / (max_val + 1e-4) * 0.95

        return waveform

    def render_all(self, score: Score) -> torch.Tensor:
        """Render all tracks into a combined mix waveform."""
        rendered_tracks = [self.render_track(score, i) for i in range(len(score.tracks))]
        if not rendered_tracks:
            return torch.zeros(int(self.sample_rate * 1.0))

        max_len = max(t.shape[0] for t in rendered_tracks)
        mix = torch.zeros(max_len, dtype=torch.float32)
        for t in rendered_tracks:
            mix[: t.shape[0]] += t

        max_val = mix.abs().max()
        if max_val > 0.99:
            mix = mix / (max_val + 1e-4) * 0.95
        return mix
