"""Unit tests for audio feature extractors, pooling modules, and fusion adapters."""

import pytest
import torch

from midigpt._types import Bar, Note, Score, Track
from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.adapters.hierarchical_fusion import (
    ConcatLinearFusion,
    FiLMFusion,
    ResidualAddFusion,
)
from midigpt.adapters.pooling import (
    BeatGridPooler,
    MeanConvPooler,
    PerceiverResampler,
)
from midigpt.audio.synth import FastAudioSynthesizer, get_bar_intervals
from midigpt.audio.yourmt3_extractor import YourMT3Extractor
from midigpt.audio.generic_proxy_extractor import GenericProxyAudioExtractor


def test_fast_audio_synthesizer_and_bar_intervals():
    # 4-bar score
    bars = [
        Bar(
            notes=[Note(pitch=60, velocity=90, onset_ticks=0, duration_ticks=240)],
            ts_numerator=4,
            ts_denominator=4,
        )
        for _ in range(4)
    ]
    score = Score(tracks=[Track(bars=bars, instrument=0)], resolution=480, tempo=500000)

    intervals = get_bar_intervals(score)
    assert len(intervals) == 4
    # At 120 BPM in 4/4, each bar is 2.0 seconds
    assert pytest.approx(intervals[0], 0.01) == (0.0, 2.0)
    assert pytest.approx(intervals[1], 0.01) == (2.0, 4.0)

    synth = FastAudioSynthesizer(sample_rate=16000)
    wav = synth.render_track(score, 0)
    # 4 bars * 2.0s = 8.0s -> 8.0 * 16000 = 128000 samples
    assert wav.shape[0] == 8 * 16000
    assert not torch.isnan(wav).any()


def test_yourmt3_extractor():
    extractor = YourMT3Extractor(feature_dim=128, n_mels=64, n_layers=2, n_heads=4)
    dummy_wav = torch.randn(1, 16000)  # 1 second
    frames = extractor.extract_frames(dummy_wav)
    assert frames.ndim == 3  # (1, T, 128)
    assert frames.shape[-1] == 128
    assert frames.shape[1] > 0

    # Bar extraction
    bar_frames = extractor.extract_bar_frames(
        dummy_wav.squeeze(0), sample_rate=16000, bar_intervals=[(0.0, 0.5), (0.5, 1.0)]
    )
    assert len(bar_frames) == 2
    assert bar_frames[0].ndim == 2
    assert bar_frames[0].shape[-1] == 128


def test_generic_proxy_audio_extractor():
    extractor = GenericProxyAudioExtractor(feature_dim=128, n_mels=64, n_layers=2, n_heads=4)
    dummy_wav = torch.randn(1, 22050)
    frames = extractor.extract_frames(dummy_wav)
    assert frames.ndim == 3
    assert frames.shape[-1] == 128


def test_pooling_modules():
    B, T, D = 2, 50, 128
    frames = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)

    # 1. MeanConvPooler
    mean_pooler = MeanConvPooler(feature_dim=D, num_outputs=1)
    out_mean = mean_pooler(frames, mask=mask)
    assert out_mean.shape == (B, 1, D)

    # 2. PerceiverResampler
    perceiver = PerceiverResampler(feature_dim=D, num_latents=1, num_heads=4)
    out_perc = perceiver(frames, mask=mask)
    assert out_perc.shape == (B, 1, D)

    # 3. BeatGridPooler
    beat_pooler = BeatGridPooler(feature_dim=D, beats_per_bar=4)
    out_beat = beat_pooler(frames, mask=mask)
    assert out_beat.shape == (B, 4, D)


def test_hierarchical_fusion_modules():
    B, K, D = 2, 1, 128
    h_bar = torch.randn(B, K, D)
    h_track = torch.randn(B, D)

    # ConcatLinear
    concat_fuse = ConcatLinearFusion(feature_dim=D)
    fused_c = concat_fuse(h_bar, h_track)
    assert fused_c.shape == (B, K, D)

    # FiLM
    film_fuse = FiLMFusion(feature_dim=D)
    fused_f = film_fuse(h_bar, h_track)
    assert fused_f.shape == (B, K, D)

    # Residual
    res_fuse = ResidualAddFusion(feature_dim=D)
    fused_r = res_fuse(h_bar, h_track)
    assert fused_r.shape == (B, K, D)


def test_audio_to_midi_adapter_end_to_end():
    B, T, D_trans, D_model = 2, 60, 256, 128
    bar_frames = torch.randn(B, T, D_trans)
    track_frames = torch.randn(B, 300, D_trans)

    adapter = AudioToMidiAdapter(
        feature_dim=D_trans,
        model_dim=D_model,
        num_tokens_per_bar=1,
        pooling_strategy="perceiver",
        fusion_strategy="concat",
        perceiver_heads=4,
    )

    soft_embs = adapter(bar_frames, track_frames=track_frames)
    assert soft_embs.shape == (B, 1, D_model)
    assert not torch.isnan(soft_embs).any()
