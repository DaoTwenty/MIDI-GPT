"""Standalone tests for the ported `midigpt.attributes.tension` package."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from midigpt._types import Score
from midigpt.attributes.tension import (
    Tension,
    TensionDrum,
    _TENSION_CACHE,
    score_to_temp_midi,
)
from midigpt.attributes.tension._processing import (
    _INTERVAL_DISSONANCE,
    calculateChordDissonance12tet,
    normal_round,
    normalize,
)
from midigpt.attributes.tension._timing import (
    _ticks_to_seconds,
    get_bar_starts_seconds,
    interval_level_average,
)

MIDI_DIR = pathlib.Path(__file__).resolve().parents[1] / "midi"
SAMPLE_MIDI = MIDI_DIR / "Aicha.mid"


# --------------------------------------------------------------------------- #
# 1. Metadata invariants
# --------------------------------------------------------------------------- #
def test_tension_metadata():
    assert Tension.name == "tension"
    assert Tension.token_type == "Tension"
    assert Tension.level == "bar"
    assert Tension.track_type == "melodic"
    assert Tension.size == 10


def test_tension_drum_metadata():
    assert TensionDrum.name == "tension_drum"
    assert TensionDrum.token_type == "TensionDrum"
    assert TensionDrum.level == "bar"
    assert TensionDrum.track_type == "drum"
    assert TensionDrum.size == 10


# --------------------------------------------------------------------------- #
# 2. quantize() — boundary and clamping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (-2.0, 0),
        (-1.99, 0),
        (0.0, 5),
        (1.99, 9),
        (2.0, 9),
        (-10.0, 0),
        (10.0, 9),
    ],
)
def test_tension_quantize_boundaries(value, expected):
    assert Tension().quantize(value) == expected


def test_tension_quantize_monotone():
    t = Tension()
    vals = np.linspace(-2.0, 2.0, 101)
    bins = [t.quantize(float(v)) for v in vals]
    assert all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1))


def test_tension_no_bar_idx_returns_zero():
    score = Score.from_midi(str(SAMPLE_MIDI))
    assert Tension().compute(score, 0, bar_idx=None) == 0.0


# --------------------------------------------------------------------------- #
# 3. _processing unit tests
# --------------------------------------------------------------------------- #
def test_normal_round_half_up():
    assert normal_round(0.5) == 1
    assert normal_round(1.5) == 2
    assert normal_round(2.5) == 3
    assert normal_round(-0.5) == 0
    assert normal_round(-1.5) == -1


def test_normalize_zscore():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = normalize(arr)
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-9)
    assert out[0] < 0 < out[-1]


def test_normalize_constant_array_is_zero():
    arr = np.array([3.0, 3.0, 3.0])
    out = normalize(arr)
    assert np.all(out == 0.0)


def test_chord_dissonance_unison_is_zero():
    assert calculateChordDissonance12tet([60]) == 0.0
    assert calculateChordDissonance12tet([60, 60]) == 0.0


def test_chord_dissonance_semitone_max():
    assert calculateChordDissonance12tet([60, 61]) == pytest.approx(_INTERVAL_DISSONANCE[1])


def test_chord_dissonance_perfect_fifth_low():
    assert calculateChordDissonance12tet([60, 67]) == pytest.approx(_INTERVAL_DISSONANCE[7])


# --------------------------------------------------------------------------- #
# 4. _timing unit tests
# --------------------------------------------------------------------------- #
def test_ticks_to_seconds_constant_tempo():
    # 120 bpm = 500_000 mspq -> 1 quarter = 0.5 s
    tpq = 480
    out = _ticks_to_seconds([0, 480, 960], [0], [500_000], tpq)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_ticks_to_seconds_tempo_change():
    tpq = 480
    # 0..480 ticks @ 120 bpm (0.5 s), then 480..960 @ 60 bpm (1.0 s)
    out = _ticks_to_seconds([0, 480, 960], [0, 480], [500_000, 1_000_000], tpq)
    assert out == pytest.approx([0.0, 0.5, 1.5])


def test_interval_level_average_uniform_input():
    tension = np.array([1.0, 1.0, 1.0, 1.0])
    sample_times = np.array([0.0, 0.25, 0.5, 0.75])
    starts = np.array([0.0, 0.5, 1.0])
    bars, vals = interval_level_average(tension, sample_times, starts)
    assert bars.tolist() == [1, 2]
    assert vals.tolist() == pytest.approx([1.0, 1.0])


def test_interval_level_average_single_start_returns_empty():
    bars, vals = interval_level_average(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert bars.size == 0
    assert vals.size == 0


# --------------------------------------------------------------------------- #
# 5. End-to-end on a real MIDI file
# --------------------------------------------------------------------------- #
@pytest.fixture
def aicha_score():
    _TENSION_CACHE.clear()
    return Score.from_midi(str(SAMPLE_MIDI))


def _pick_busy_track(score, ttype="melodic"):
    candidates = [
        (i, sum(len(b.notes) for b in tr.bars))
        for i, tr in enumerate(score.tracks)
        if tr.track_type == ttype
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None


def test_tension_compute_returns_float(aicha_score):
    track_idx = _pick_busy_track(aicha_score, "melodic")
    assert track_idx is not None
    val = Tension().compute(aicha_score, track_idx, 0)
    assert isinstance(val, (float, int))


def test_tension_compute_quantized_in_range(aicha_score):
    track_idx = _pick_busy_track(aicha_score, "melodic")
    t = Tension()
    n = min(20, len(aicha_score.tracks[track_idx].bars))
    bins = [t.quantize(t.compute(aicha_score, track_idx, b)) for b in range(n)]
    assert all(0 <= b <= 9 for b in bins)


def test_tension_caches_per_track(aicha_score):
    _TENSION_CACHE.clear()
    track_idx = _pick_busy_track(aicha_score, "melodic")
    t = Tension()
    t.compute(aicha_score, track_idx, 0)
    assert (id(aicha_score), track_idx) in _TENSION_CACHE


def test_tension_deterministic(aicha_score):
    _TENSION_CACHE.clear()
    track_idx = _pick_busy_track(aicha_score, "melodic")
    t = Tension()
    a = [t.compute(aicha_score, track_idx, b) for b in range(5)]
    _TENSION_CACHE.clear()
    b = [t.compute(aicha_score, track_idx, b) for b in range(5)]
    assert a == b


def test_tension_drum_compute_runs(aicha_score):
    drum_idx = _pick_busy_track(aicha_score, "drum")
    if drum_idx is None:
        pytest.skip("no drum track in sample MIDI")
    val = TensionDrum().compute(aicha_score, drum_idx, 0)
    assert isinstance(val, (float, int))
    assert 0 <= TensionDrum().quantize(val) <= 9


def test_get_bar_starts_seconds_matches_score(aicha_score):
    midi_path = score_to_temp_midi(aicha_score)
    starts = get_bar_starts_seconds(midi_path)
    assert starts.size >= 2
    assert np.all(np.diff(starts) > 0)


# --------------------------------------------------------------------------- #
# 6. C++ encode -> decode round-trip for Tension and TensionDrum
# --------------------------------------------------------------------------- #
def _round_trip(midi_path, with_tension=True, window_bars=16):
    """Build a tokenizer wired with Tension+TensionDrum, encode + decode, return
    the decoded Score."""
    import midigpt._core as _core
    from midigpt.attributes.base import AttributeAnalyzer
    from midigpt.tokenizer.tokenizer import Tokenizer

    cfg = _core.EncoderConfig.from_json(
        pathlib.Path("models/ghost_config.json").read_text()
    )
    analyzer = AttributeAnalyzer(
        [Tension(), TensionDrum()] if with_tension else []
    )
    tok = Tokenizer(cfg, analyzer=analyzer)
    score = Score.from_midi(midi_path)
    for tr in score.tracks:
        tr.bars = tr.bars[:window_bars]
    opts = _core.EncodeOptions()
    opts.window_bars = window_bars
    tokens = tok.encode(score, opts=opts)
    decoded = tok.decode(tokens)
    return score, decoded, tokens, tok


def test_round_trip_tension_melodic():
    """Encode a melodic-only file; the melodic track must round-trip Tension
    tokens but no TensionDrum tokens."""
    _TENSION_CACHE.clear()
    score, decoded, tokens, _ = _round_trip(str(SAMPLE_MIDI))
    # Find the busiest melodic track in the *decoded* score
    melodic_idx = max(
        (i for i, tr in enumerate(decoded.tracks) if tr.track_type == "melodic"),
        key=lambda i: sum(len(b.notes) for b in decoded.tracks[i].bars),
    )
    track = decoded.tracks[melodic_idx]
    bar_keys = [k for k in track.attributes if k.startswith("bar_Tension_")]
    drum_keys = [k for k in track.attributes if k.startswith("bar_TensionDrum_")]
    assert len(bar_keys) > 0, "expected at least one Tension token on a melodic track"
    assert len(drum_keys) == 0, "TensionDrum must NOT appear on melodic tracks"
    # Every decoded value must be a valid quantized bin
    for k in bar_keys:
        assert 0 <= track.attributes[k] <= 9


def test_round_trip_tension_drum():
    """Encode a file with drums; the drum track must round-trip TensionDrum
    tokens but no Tension tokens."""
    _TENSION_CACHE.clear()
    score, decoded, tokens, _ = _round_trip(str(SAMPLE_MIDI))
    drum_tracks = [i for i, tr in enumerate(decoded.tracks) if tr.track_type == "drum"]
    if not drum_tracks:
        pytest.skip("no drum track in sample MIDI")
    track = decoded.tracks[drum_tracks[0]]
    bar_keys = [k for k in track.attributes if k.startswith("bar_Tension_")]
    drum_keys = [k for k in track.attributes if k.startswith("bar_TensionDrum_")]
    assert len(drum_keys) > 0, "expected at least one TensionDrum token on a drum track"
    assert len(bar_keys) == 0, "Tension must NOT appear on drum tracks"
    for k in drum_keys:
        assert 0 <= track.attributes[k] <= 9


def test_round_trip_tension_values_match_input():
    """Encoder reads track.attributes; decoder writes them back. After
    encode→decode, every original key must reappear with the same value."""
    import midigpt._core as _core
    from midigpt.attributes.base import AttributeAnalyzer
    from midigpt.tokenizer.tokenizer import Tokenizer

    _TENSION_CACHE.clear()
    cfg = _core.EncoderConfig.from_json(
        pathlib.Path("models/ghost_config.json").read_text()
    )
    analyzer = AttributeAnalyzer([Tension(), TensionDrum()])
    tok = Tokenizer(cfg, analyzer=analyzer)

    score = Score.from_midi(str(SAMPLE_MIDI))
    window_bars = 16
    for tr in score.tracks:
        tr.bars = tr.bars[:window_bars]
    opts = _core.EncodeOptions()
    opts.window_bars = window_bars
    # First encode populates track.attributes via the analyzer
    tokens = tok.encode(score, opts=opts, compute_attributes=True)
    decoded = tok.decode(tokens)

    for t_idx, (orig, dec) in enumerate(zip(score.tracks, decoded.tracks)):
        orig_tension_keys = {
            k: v for k, v in orig.attributes.items()
            if k.startswith("bar_Tension_") or k.startswith("bar_TensionDrum_")
        }
        for k, v in orig_tension_keys.items():
            assert k in dec.attributes, f"track {t_idx}: lost key {k!r} after round-trip"
            assert dec.attributes[k] == v, (
                f"track {t_idx} key {k}: encoded={v} decoded={dec.attributes[k]}"
            )
