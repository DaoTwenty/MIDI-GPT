from midigpt import Score
import os

midi_dir = "tmp/midi_examples"
files = [f for f in os.listdir(midi_dir) if f.endswith(".mid") or f.endswith(".midi")]

for filename in sorted(files):
    path = os.path.join(midi_dir, filename)
    try:
        score = Score.from_midi(path)
        print(f"\nFile: {filename}")
        print(f"Resolution: {score.resolution}, Tempo: {score.tempo}")
        print(f"Tracks: {len(score.tracks)}")
        for idx, t in enumerate(score.tracks):
            total_notes = sum(len(b.notes) for b in t.bars)
            print(f"  Track {idx}: instrument={t.instrument}, type={t.track_type}, bars={len(t.bars)}, notes={total_notes}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
