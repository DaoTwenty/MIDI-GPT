from midigpt import Score
import os

for filename in ["Never-Gonna-Give-You-Up-1.mid", "Home-Resonance-escafandro-20220516104213-nonstop2k.com.mid"]:
    path = f"tmp/midi_examples/{filename}"
    try:
        s = Score.from_midi(path)
        print(f"\n=== File: {filename} ===")
        print(f"Tempo: {s.tempo}, Resolution: {s.resolution}")
        for idx, t in enumerate(s.tracks):
            total_notes = sum(len(b.notes) for b in t.bars)
            print(f"  Track {idx}: inst={t.instrument}, type={t.track_type}, total_notes={total_notes}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
