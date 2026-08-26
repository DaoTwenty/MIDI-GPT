"""Parity tests between the ported `midigpt.attributes.tension` package and the
upstream reference implementation at
`github.com/punchline10/musical-tension-model-adjustments-for-GigaMIDI-Dataset`.

The reference repo is not a Python dependency. Set the `TENSION_MODEL_REPO`
environment variable to its path, or place it as a sibling directory of this
repo (default: `../musical-tension-model-adjustments-for-GigaMIDI-Dataset`).
All tests in this file are skipped if the repo is not found.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

import numpy as np
import pytest


def _stub_optional_deps():
    """The reference repo imports `resampy` and `matplotlib` at module-load
    time but doesn't use them on the code paths we exercise. Install minimal
    stubs so the imports succeed without dragging in numba/llvm."""
    if "resampy" not in sys.modules:
        m = types.ModuleType("resampy")
        m.resample = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("resampy stubbed in parity tests")
        )
        sys.modules["resampy"] = m
    if "matplotlib" not in sys.modules:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            mpl = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            mpl.pyplot = pyplot
            sys.modules["matplotlib"] = mpl
            sys.modules["matplotlib.pyplot"] = pyplot

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_REF = REPO_ROOT.parent / "musical-tension-model-adjustments-for-GigaMIDI-Dataset"
REF_PATH = pathlib.Path(os.environ.get("TENSION_MODEL_REPO", str(DEFAULT_REF)))
MIDI_DIR = REPO_ROOT / "tests" / "midi"
SAMPLE_MIDI = MIDI_DIR / "Aicha.mid"

if not REF_PATH.exists():
    pytest.skip(
        f"reference repo not found at {REF_PATH}; set TENSION_MODEL_REPO to enable parity tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def ref_modules():
    """Import the reference repo's modules with its directory on sys.path."""
    ref_str = str(REF_PATH)
    added = ref_str not in sys.path
    if added:
        sys.path.insert(0, ref_str)
    _stub_optional_deps()
    # Drop any cached modules from a previous test run so the reference imports win.
    for mod in [
        "tensionModel",
        "dataProcessing",
        "featureAnalysis",
        "noteObj",
        "dissonance",
        "helperFunctions",
        "tension_calculation",
    ]:
        sys.modules.pop(mod, None)
    try:
        tm = importlib.import_module("tensionModel")
        dp = importlib.import_module("dataProcessing")
        fa = importlib.import_module("featureAnalysis")
        yield {"tensionModel": tm, "dataProcessing": dp, "featureAnalysis": fa}
    finally:
        if added:
            sys.path.remove(ref_str)


@pytest.fixture(scope="module")
def busy_track_index():
    """Pick a track in SAMPLE_MIDI with notes (so the tension model has data)."""
    import symusic

    s = symusic.Score(str(SAMPLE_MIDI), ttype="tick")
    counts = [
        (i, len(tr.notes.numpy().get("time", [])))
        for i, tr in enumerate(s.tracks)
        if not tr.is_drum
    ]
    counts.sort(key=lambda x: x[1], reverse=True)
    assert counts, "no pitched tracks found in sample MIDI"
    return counts[0][0]


# --------------------------------------------------------------------------- #
# 1. compute_track_tension: ported output should equal upstream output
# --------------------------------------------------------------------------- #
def _reference_compute_track_tension(ref_modules, midi_path, track_index, is_drum):
    """Re-implement the reference repo's `compute_track_tension` from
    testTensionModel.py using its own modules, so we don't depend on the test
    script's CLI plumbing."""
    tm = ref_modules["tensionModel"]
    dp = ref_modules["dataProcessing"]
    fa = ref_modules["featureAnalysis"]

    # Constants from testTensionModel.py (must match the ported _model.py)
    SAMPLE_RATE = 10
    memoryWindowDur = 3
    memoryWeight = 2
    sliderOnset = False
    attentionalWindowDur = 3
    windowShift = 0.25
    initSlope = 1
    lag = 1
    target = []
    featureWeights_pitched = [2, 3, 3, 2, 1, 1]
    featureWeights_drums = [2, 0, 3, 2, 0, 0]

    if is_drum:
        weights = featureWeights_drums
        flags = (True, False, True, True, False, False)
    else:
        weights = featureWeights_pitched
        flags = (True, True, True, True, True, True)
    bOnsetFreq, bMelodicContour, bLoudness, bTempo, bHarmony, bDissonance = flags

    features = fa.extractFeaturesMidi(
        midi_path, SAMPLE_RATE,
        bOnsetFreq, bMelodicContour, bLoudness, bTempo, bHarmony, bDissonance,
        track_index=track_index,
    )
    if features is None or features.shape[1] == 0:
        return np.array([]), np.array([])

    for i, active in enumerate(flags):
        if active:
            features[i, :] = dp.normalize(features[i, :])

    featureLen = features.shape[1]
    t_seconds = np.arange(featureLen) / SAMPLE_RATE
    tension = tm.runModel(
        features, target, fa.featureList, weights,
        memoryWindowDur, SAMPLE_RATE,
        attentionalWindowDur, windowShift,
        midi_path, memoryWeight, initSlope, lag, sliderOnset,
    )
    return t_seconds, np.asarray(tension, dtype=float)


def test_parity_compute_track_tension_pitched(ref_modules, busy_track_index):
    from midigpt.attributes.tension._features import compute_track_tension as ported

    midi_path = str(SAMPLE_MIDI)
    t_ref, ten_ref = _reference_compute_track_tension(
        ref_modules, midi_path, busy_track_index, False
    )
    t_port, ten_port = ported(midi_path, busy_track_index, False)

    assert t_port.shape == t_ref.shape
    assert ten_port.shape == ten_ref.shape
    np.testing.assert_allclose(t_port, t_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ten_port, ten_ref, rtol=1e-9, atol=1e-9)


def test_parity_compute_track_tension_drum(ref_modules):
    import symusic

    from midigpt.attributes.tension._features import compute_track_tension as ported

    s = symusic.Score(str(SAMPLE_MIDI), ttype="tick")
    drum_indices = [i for i, tr in enumerate(s.tracks) if tr.is_drum]
    if not drum_indices:
        pytest.skip("no drum tracks in sample MIDI")
    drum_idx = drum_indices[0]

    midi_path = str(SAMPLE_MIDI)
    t_ref, ten_ref = _reference_compute_track_tension(
        ref_modules, midi_path, drum_idx, True
    )
    t_port, ten_port = ported(midi_path, drum_idx, True)

    np.testing.assert_allclose(t_port, t_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ten_port, ten_ref, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# 2. interval_level_average parity
# --------------------------------------------------------------------------- #
def _reference_interval_level_average(tension_arr, sample_times_sec, interval_starts_sec):
    """Verbatim copy of `interval_level_average` from the reference repo's
    testTensionModel.py — inlined because that script has a syntax error that
    prevents importing it directly."""
    interval_starts_sec = np.asarray(interval_starts_sec, dtype=float)
    if len(interval_starts_sec) < 2:
        return np.array([]), np.array([])
    interval_ids = np.searchsorted(interval_starts_sec[1:], sample_times_sec, side="right")
    n = len(interval_starts_sec) - 1
    vals = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)
    for i, idx in enumerate(interval_ids):
        if 0 <= idx < n:
            vals[idx] += float(tension_arr[i])
            counts[idx] += 1
    counts[counts == 0] = 1
    beat_numbers = np.arange(1, n + 1)
    beat_tension = vals / counts
    return beat_numbers, beat_tension


def test_parity_interval_level_average():
    from midigpt.attributes.tension._timing import interval_level_average as ported

    rng = np.random.default_rng(42)
    tension = rng.normal(size=200)
    sample_times = np.linspace(0.0, 20.0, 200)
    starts = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])

    bars_ref, vals_ref = _reference_interval_level_average(tension, sample_times, starts)
    bars_port, vals_port = ported(tension, sample_times, starts)

    np.testing.assert_array_equal(bars_port, bars_ref)
    np.testing.assert_allclose(vals_port, vals_ref, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. processing helpers parity
# --------------------------------------------------------------------------- #
def test_parity_normalize(ref_modules):
    from midigpt.attributes.tension._processing import normalize as ported

    rng = np.random.default_rng(7)
    arr = rng.normal(size=64)
    ref = ref_modules["dataProcessing"].normalize(arr.copy())
    port = ported(arr.copy())
    np.testing.assert_allclose(port, ref, rtol=1e-12, atol=1e-12)


def test_parity_normal_round():
    _stub_optional_deps()
    sys.path.insert(0, str(REF_PATH))
    try:
        hf = importlib.import_module("helperFunctions")
    finally:
        sys.path.remove(str(REF_PATH))
    from midigpt.attributes.tension._processing import normal_round as ported

    for v in [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.49, 3.5, 3.51, 100.7]:
        assert ported(v) == hf.normal_round(v)


def test_parity_chord_dissonance():
    _stub_optional_deps()
    sys.path.insert(0, str(REF_PATH))
    try:
        diss = importlib.import_module("dissonance")
    finally:
        sys.path.remove(str(REF_PATH))
    from midigpt.attributes.tension._processing import (
        calculateChordDissonance12tet as ported,
    )

    chords = [[60], [60, 64, 67], [60, 61], [60, 67], [60, 63, 66, 69], [48, 52, 55, 59]]
    for c in chords:
        assert ported(c) == pytest.approx(diss.calculateChordDissonance12tet(c))
