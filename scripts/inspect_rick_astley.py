from midigpt import Score

score = Score.from_midi("tmp/midi_examples/Never-Gonna-Give-You-Up-1.mid")
print("Rick Astley - Never Gonna Give You Up Note Counts:")
for start_bar in range(0, 32, 8):
    print(f"\nBars {start_bar}-{start_bar+8}:")
    for t_idx, t in enumerate(score.tracks):
        notes = sum(len(b.notes) for b in t.bars[start_bar : start_bar + 8])
        if notes > 0:
            print(f"  Track {t_idx} (inst={t.instrument}, type={t.track_type}): {notes} notes")
