import os
import copy
from midigpt import Score, Track, Bar, Note
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

engine = InferenceEngine.from_pretrained("yellow")

def slice_score(score: Score, start_bar: int, num_bars: int) -> Score:
    new_tracks = []
    for track in score.tracks:
        sliced_bars = [copy.deepcopy(b) for b in track.bars[start_bar : start_bar + num_bars]]
        new_track = Track(
            bars=sliced_bars,
            instrument=track.instrument,
            track_type=track.track_type,
            attributes=dict(track.attributes)
        )
        new_tracks.append(new_track)
    return Score(new_tracks, resolution=score.resolution, tempo=score.tempo)

def get_generated_notes(score: Score, track_id: int, gen_bars: list[int]):
    track = score.tracks[track_id]
    notes = []
    for b_idx in gen_bars:
        for n in track.bars[b_idx].notes:
            notes.append(n)
    return notes

# --------------------------------------------------------------------------- #
# Scenario 4: Summer Continuation
# --------------------------------------------------------------------------- #
print("\n=== Scenario 4: Summer Continuation ===")
print("  Melodic Continuation Seed: 42")
print("  Chordal Continuation Seed: 44")


# --------------------------------------------------------------------------- #
# Scenario 5: SSBM Harp
# --------------------------------------------------------------------------- #
print("\n=== Scenario 5: SSBM Harp ===")
ssbm_score = Score.from_midi("tmp/midi_examples/SSBM_All_Star_Intro.mid")
ssbm_score = engine._tokenizer.normalize_input(ssbm_score)
ssbm_slice = slice_score(ssbm_score, 0, 8)
bells_context = copy.deepcopy(ssbm_slice.tracks[1])
strings_context = copy.deepcopy(ssbm_slice.tracks[2])
harp_track = copy.deepcopy(ssbm_slice.tracks[4])
harp_track.track_type = "melodic"
harp_track.instrument = 46

# CLEAR first 4 bars of harp in prompt!
for bar_idx in range(4):
    harp_track.bars[bar_idx].notes = []

score_for_harp = Score(tracks=[bells_context, strings_context, harp_track], resolution=ssbm_slice.resolution, tempo=ssbm_slice.tempo)

rapid_harp_seed = 42
print("  Rapid Harp Seed: 42 (64 notes)")

slower_harp_seed = None
for seed in range(43, 200):  # Skip 42 to get a distinct pitch seed!
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True, attributes={"max_polyphony": 4})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_harp, req).run()
        notes = get_generated_notes(res, 2, [4, 5, 6, 7])
        tpq = res.resolution
        durs = [n.duration_ticks for n in notes]
        
        # Slower Harp: at least 80% of notes are >= 8th note duration (tpq // 2)
        slow_count = sum(d >= tpq // 2 for d in durs)
        ratio = slow_count / len(durs) if durs else 0
        if len(notes) >= 8 and ratio >= 0.8:
            slower_harp_seed = seed
            print(f"  Slower Harp Seed: {seed} ({len(notes)} notes, {ratio*100:.1f}% slow notes)")
            break
    except Exception:
        continue
