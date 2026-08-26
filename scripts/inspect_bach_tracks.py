from midigpt import Score

score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
start_bar = 12
print(f"Bach BWV 1052a Bars {start_bar}-{start_bar+8} Note Counts per Track:")
for t_idx, t in enumerate(score.tracks):
    notes = [len(b.notes) for b in t.bars[start_bar : start_bar + 8]]
    print(f"  Track {t_idx} (inst={t.instrument}, type={t.track_type}): total={sum(notes)} per_bar={notes}")
