import copy
from midigpt import Score, Track
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

res_score = Score.from_midi("tmp/midi_examples/Home-Resonance-escafandro-20220516104213-nonstop2k.com.mid")
res_score = engine._tokenizer.normalize_input(res_score)
res_slice = slice_score(res_score, 48, 8)
synth_lead = copy.deepcopy(res_slice.tracks[1])
synth_bass = copy.deepcopy(res_slice.tracks[3])
drums = copy.deepcopy(res_slice.tracks[0])
score_for_infill = Score(tracks=[synth_lead, synth_bass, drums], resolution=res_slice.resolution, tempo=res_slice.tempo)

print("Original notes in Synth Lead bars 3 and 4:")
original_notes = []
for b_idx in [3, 4]:
    for n in score_for_infill.tracks[0].bars[b_idx].notes:
        original_notes.append(n)
print(f"Count: {len(original_notes)}")

req = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[3, 4], autoregressive=False, attributes={"min_polyphony": 0, "max_polyphony": 6}),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False)
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=1)
)

res = engine.session(score_for_infill, req).run()
generated_notes = []
for b_idx in [3, 4]:
    for n in res.tracks[0].bars[b_idx].notes:
        generated_notes.append(n)

print("\nGenerated notes in Synth Lead bars 3 and 4:")
print(f"Count: {len(generated_notes)}")
print("Identical to original:", [(n.pitch, n.onset_ticks) for n in original_notes] == [(n.pitch, n.onset_ticks) for n in generated_notes])
