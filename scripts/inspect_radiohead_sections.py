from midigpt import Score

score = Score.from_midi("tmp/midi_examples/EverythingIsInItsRightPlace.mid")
print("Radiohead Everything in Its Right Place Section Note Counts:")
for start_bar in range(0, 48, 8):
    print(f"\nBars {start_bar}-{start_bar+8}:")
    for t_idx, t in enumerate(score.tracks):
        notes = sum(len(b.notes) for b in t.bars[start_bar : start_bar + 8])
        print(f"  Track {t_idx} (inst={t.instrument}, type={t.track_type}): {notes} notes")
