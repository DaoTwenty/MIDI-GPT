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

for density in [1, 8]:
    print(f"\n--- Note Density: {density} ---")
    for seed in range(42, 47):
        req = GenerationRequest(
            tracks=[
                TrackPrompt(id=0, bars=[], ignore=False),
                TrackPrompt(id=1, bars=[], ignore=False),
                TrackPrompt(id=2, bars=list(range(8)), ignore=False, attributes={"note_density": density})
            ],
            config=InferenceConfig(model_dim=8, mask_mode="attention", seed=seed, max_attempts=1, tracks_per_step=2)
        )
        try:
            res = engine.session(score_for_drums, req).run()
            notes = [n for b in res.tracks[2].bars for n in b.notes]
            pitches = sorted(list(set(n.pitch for n in notes)))
            print(f"  Seed {seed}: notes={len(notes)}, unique_pitches={len(pitches)} ({pitches})")
        except Exception as e:
            print(f"  Seed {seed} error: {e}")
