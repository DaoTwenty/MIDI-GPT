from midigpt import Score
import os

path = "tmp/midi_examples/Home-Resonance-escafandro-20220516104213-nonstop2k.com.mid"
s = Score.from_midi(path)
total_bars = len(s.tracks[0].bars)
print(f"Total bars: {total_bars}")

# Print note counts in 8-bar segments
for start in range(0, total_bars, 8):
    end = min(start + 8, total_bars)
    segment_notes = []
    for t_idx, t in enumerate(s.tracks):
        n_count = sum(len(b.notes) for b in t.bars[start:end])
        segment_notes.append(f"T{t_idx}: {n_count}")
    print(f"Bars {start}-{end}: " + ", ".join(segment_notes))
