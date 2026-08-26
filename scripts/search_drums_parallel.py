import copy
import concurrent.futures
from midigpt import Score, Track
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

def search_seed(seed, density, score_for_drums):
    # Load engine inside process to avoid CUDA/multiprocessing sharing issues
    engine = InferenceEngine.from_pretrained("yellow")
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
        return seed, len(notes), pitches
    except Exception as e:
        return seed, 0, []

if __name__ == "__main__":
    engine = InferenceEngine.from_pretrained("yellow")
    hotel_score = Score.from_midi("tmp/midi_examples/HotelCalifornia.mid")
    hotel_score = engine._tokenizer.normalize_input(hotel_score)
    
    # Slice score
    new_tracks = []
    for track in hotel_score.tracks:
        sliced_bars = [copy.deepcopy(b) for b in track.bars[16:24]]
        new_track = Track(bars=sliced_bars, instrument=track.instrument, track_type=track.track_type, attributes=dict(track.attributes))
        new_tracks.append(new_track)
    hotel_slice = Score(new_tracks, resolution=hotel_score.resolution, tempo=hotel_score.tempo)
    
    guitar_context = copy.deepcopy(hotel_slice.tracks[0])
    bass_context = copy.deepcopy(hotel_slice.tracks[2])
    drums_target = copy.deepcopy(hotel_slice.tracks[5])
    drums_target.track_type = "drum"
    drums_target.instrument = 0
    for bar in drums_target.bars:
        bar.notes = []
    score_for_drums = Score(tracks=[guitar_context, bass_context, drums_target], resolution=hotel_slice.resolution, tempo=hotel_slice.tempo)
    
    print("Searching sparse drums (density=1)...")
    seeds_to_try = list(range(42, 120))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(search_seed, s, 1, score_for_drums): s for s in seeds_to_try}
        for future in concurrent.futures.as_completed(futures):
            seed, n_count, pitches = future.result()
            if n_count >= 12 and len(pitches) >= 3:
                print(f"FOUND Sparse Drum Seed: {seed} with {n_count} notes, pitches={pitches}")
                break
                
    print("\nSearching dense drums (density=8)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(search_seed, s, 8, score_for_drums): s for s in seeds_to_try}
        for future in concurrent.futures.as_completed(futures):
            seed, n_count, pitches = future.result()
            if n_count >= 60 and len(pitches) >= 5:
                print(f"FOUND Dense Drum Seed: {seed} with {n_count} notes, pitches={pitches}")
                break
