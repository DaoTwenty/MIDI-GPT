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
# Scenario 1: Hotel California (Drums)
# --------------------------------------------------------------------------- #
print("\n=== Scenario 1: Drums ===")
hotel_score = Score.from_midi("tmp/midi_examples/HotelCalifornia.mid")
hotel_score = engine._tokenizer.normalize_input(hotel_score)
hotel_slice = slice_score(hotel_score, 16, 8)
guitar_context = copy.deepcopy(hotel_slice.tracks[0])
bass_context = copy.deepcopy(hotel_slice.tracks[2])
drums_target = copy.deepcopy(hotel_slice.tracks[5])
drums_target.track_type = "drum"
drums_target.instrument = 0
for bar in drums_target.bars:
    bar.notes = []
score_for_drums = Score(tracks=[guitar_context, bass_context, drums_target], resolution=hotel_slice.resolution, tempo=hotel_slice.tempo)

sparse_drums_seed = None
sparse_drums_repr = []
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=list(range(8)), ignore=False, attributes={"note_density": 1})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1, tracks_per_step=2)
    )
    try:
        res = engine.session(score_for_drums, req).run()
        notes = get_generated_notes(res, 2, list(range(8)))
        pitches = [n.pitch for n in notes]
        unique_pitches = set(pitches)
        # Relaxed drum pitch diversity for sparse: >= 3 unique pitches (e.g. Kick, Snare, Hihat/Tom)
        if len(notes) >= 12 and len(unique_pitches) >= 3:
            sparse_drums_seed = seed
            sparse_drums_repr = [(n.pitch, n.onset_ticks) for n in notes]
            print(f"  Sparse Drums Seed: {seed} ({len(notes)} notes, {len(unique_pitches)} unique pitches: {sorted(list(unique_pitches))})")
            break
    except Exception:
        continue

dense_drums_seed = None
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=list(range(8)), ignore=False, attributes={"note_density": 8})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1, tracks_per_step=2)
    )
    try:
        res = engine.session(score_for_drums, req).run()
        notes = get_generated_notes(res, 2, list(range(8)))
        pitches = [n.pitch for n in notes]
        unique_pitches = set(pitches)
        notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
        if len(notes) >= 60 and len(unique_pitches) >= 4 and notes_repr != sparse_drums_repr:
            dense_drums_seed = seed
            print(f"  Dense Drums Seed: {seed} ({len(notes)} notes, {len(unique_pitches)} unique pitches: {sorted(list(unique_pitches))})")
            break
    except Exception:
        continue


# --------------------------------------------------------------------------- #
# Scenario 2: Bach Cello
# --------------------------------------------------------------------------- #
print("\n=== Scenario 2: Bach Cello ===")
bach_score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
bach_score = engine._tokenizer.normalize_input(bach_score)
bach_slice = slice_score(bach_score, 12, 8)
violin1_context = copy.deepcopy(bach_slice.tracks[0])
violin2_context = copy.deepcopy(bach_slice.tracks[1])
viola_context = copy.deepcopy(bach_slice.tracks[2])
cello_target = copy.deepcopy(bach_slice.tracks[3])
cello_target.track_type = "melodic"
cello_target.instrument = 43
for bar in cello_target.bars:
    bar.notes = []
score_for_cello = Score(tracks=[violin1_context, violin2_context, viola_context, cello_target], resolution=bach_slice.resolution, tempo=bach_slice.tempo)

staccato_cello_seed = None
staccato_notes_repr = []
for seed in range(42, 250):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(id=3, bars=list(range(8)), ignore=False, attributes={"max_polyphony": 0, "max_note_duration": 1})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1, tracks_per_step=2)
    )
    try:
        res = engine.session(score_for_cello, req).run()
        notes = get_generated_notes(res, 3, list(range(8)))
        pitches = [n.pitch for n in notes]
        unique_pitches = set(pitches)
        
        # Check bar-to-bar diversity
        pitches_per_bar = []
        for b_idx in range(8):
            b_pitches = tuple(sorted(list(set(n.pitch for n in res.tracks[3].bars[b_idx].notes))))
            if b_pitches:
                pitches_per_bar.append(b_pitches)
        distinct_bars = len(set(pitches_per_bar))
        
        if len(notes) >= 15 and len(unique_pitches) >= 5 and distinct_bars >= 4:
            staccato_cello_seed = seed
            staccato_notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
            print(f"  Staccato Cello Seed: {seed} ({len(notes)} notes, {len(unique_pitches)} unique pitches, {distinct_bars} distinct bars)")
            break
    except Exception:
        continue

sustained_cello_seed = None
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(id=3, bars=list(range(8)), ignore=False, autoregressive=True, attributes={"max_polyphony": 0, "min_note_duration": 4})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1, tracks_per_step=2)
    )
    try:
        res = engine.session(score_for_cello, req).run()
        notes = get_generated_notes(res, 3, list(range(8)))
        pitches = [n.pitch for n in notes]
        unique_pitches = set(pitches)
        notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
        if len(notes) >= 6 and len(unique_pitches) >= 4 and notes_repr != staccato_notes_repr:
            sustained_cello_seed = seed
            print(f"  Sustained Cello Seed: {seed} ({len(notes)} notes, {len(unique_pitches)} unique pitches)")
            break
    except Exception:
        continue


# --------------------------------------------------------------------------- #
# Scenario 3: HOME Resonance Infill
# --------------------------------------------------------------------------- #
print("\n=== Scenario 3: HOME Resonance Infill ===")
res_score = Score.from_midi("tmp/midi_examples/Home-Resonance-escafandro-20220516104213-nonstop2k.com.mid")
res_score = engine._tokenizer.normalize_input(res_score)
res_slice = slice_score(res_score, 48, 8)
synth_lead = copy.deepcopy(res_slice.tracks[1])
synth_bass = copy.deepcopy(res_slice.tracks[3])
drums = copy.deepcopy(res_slice.tracks[0])
score_for_infill = Score(tracks=[synth_lead, synth_bass, drums], resolution=res_slice.resolution, tempo=res_slice.tempo)

sparse_infill_seed = None
sparse_infill_repr = []
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[3, 4], autoregressive=False, attributes={"min_polyphony": 0, "max_polyphony": 6}),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_infill, req).run()
        notes = get_generated_notes(res, 0, [3, 4])
        onsets = {}
        for n in notes:
            onsets[n.onset_ticks] = onsets.get(n.onset_ticks, 0) + 1
        max_poly = max(onsets.values()) if onsets else 0
        if 2 <= len(notes) <= 12 and max_poly <= 3:
            sparse_infill_seed = seed
            sparse_infill_repr = [(n.pitch, n.onset_ticks) for n in notes]
            print(f"  Sparse Infill Seed: {seed} ({len(notes)} notes, max_poly={max_poly})")
            break
    except Exception:
        continue

chordal_infill_seed = None
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[3, 4], autoregressive=False, attributes={"min_polyphony": 3, "max_polyphony": 9}),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_infill, req).run()
        notes = get_generated_notes(res, 0, [3, 4])
        onsets = {}
        for n in notes:
            onsets[n.onset_ticks] = onsets.get(n.onset_ticks, 0) + 1
        max_poly = max(onsets.values()) if onsets else 0
        notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
        if len(notes) >= 12 and max_poly >= 3 and notes_repr != sparse_infill_repr:
            chordal_infill_seed = seed
            print(f"  Chordal Infill Seed: {seed} ({len(notes)} notes, max_poly={max_poly})")
            break
    except Exception:
        continue


# --------------------------------------------------------------------------- #
# Scenario 4: Summer Continuation
# --------------------------------------------------------------------------- #
print("\n=== Scenario 4: Summer Continuation ===")
summer_score = Score.from_midi("tmp/midi_examples/Summer.mid")
summer_score = engine._tokenizer.normalize_input(summer_score)
summer_slice = slice_score(summer_score, 13, 8)
piano_track = copy.deepcopy(summer_slice.tracks[0])
bass_context = copy.deepcopy(summer_slice.tracks[1])
strings_context = copy.deepcopy(summer_slice.tracks[5])
score_for_ar = Score(tracks=[piano_track, bass_context, strings_context], resolution=summer_slice.resolution, tempo=summer_slice.tempo)

melodic_ar_seed = None
melodic_ar_repr = []
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True, attributes={"min_polyphony": 0, "max_polyphony": 4}),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_ar, req).run()
        notes = get_generated_notes(res, 0, [4, 5, 6, 7])
        
        # Check loop/repetition avoidance
        pitches_per_bar = []
        for b_idx in [4, 5, 6, 7]:
            b_pitches = tuple(sorted(list(set(n.pitch for n in res.tracks[0].bars[b_idx].notes))))
            if b_pitches:
                pitches_per_bar.append(b_pitches)
        distinct_bars = len(set(pitches_per_bar))
        
        if len(notes) >= 12 and distinct_bars >= 3:
            melodic_ar_seed = seed
            melodic_ar_repr = [(n.pitch, n.onset_ticks) for n in notes]
            print(f"  Melodic Continuation Seed: {seed} ({len(notes)} notes, {distinct_bars} distinct bars)")
            break
    except Exception:
        continue

chordal_ar_seed = None
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True, attributes={"min_polyphony": 2, "max_polyphony": 8}),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_ar, req).run()
        notes = get_generated_notes(res, 0, [4, 5, 6, 7])
        notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
        
        pitches_per_bar = []
        for b_idx in [4, 5, 6, 7]:
            b_pitches = tuple(sorted(list(set(n.pitch for n in res.tracks[0].bars[b_idx].notes))))
            if b_pitches:
                pitches_per_bar.append(b_pitches)
        distinct_bars = len(set(pitches_per_bar))
        
        if len(notes) >= 16 and distinct_bars >= 3 and notes_repr != melodic_ar_repr:
            chordal_ar_seed = seed
            print(f"  Chordal Continuation Seed: {seed} ({len(notes)} notes, {distinct_bars} distinct bars)")
            break
    except Exception:
        continue


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
score_for_harp = Score(tracks=[bells_context, strings_context, harp_track], resolution=ssbm_slice.resolution, tempo=ssbm_slice.tempo)

rapid_harp_seed = None
rapid_harp_repr = []
for seed in range(42, 150):
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True, attributes={"max_note_duration": 1})
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1)
    )
    try:
        res = engine.session(score_for_harp, req).run()
        notes = get_generated_notes(res, 2, [4, 5, 6, 7])
        tpq = res.resolution
        durs = [n.duration_ticks for n in notes]
        if len(notes) >= 15 and all(d <= 2 * tpq for d in durs):
            rapid_harp_seed = seed
            rapid_harp_repr = [(n.pitch, n.onset_ticks) for n in notes]
            print(f"  Rapid Harp Seed: {seed} ({len(notes)} notes)")
            break
    except Exception:
        continue

slower_harp_seed = None
for seed in range(42, 150):
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
        notes_repr = [(n.pitch, n.onset_ticks) for n in notes]
        if len(notes) >= 8 and all(d >= 2 * tpq for d in durs) and notes_repr != rapid_harp_repr:
            slower_harp_seed = seed
            print(f"  Slower Harp Seed: {seed} ({len(notes)} notes)")
            break
    except Exception:
        continue
