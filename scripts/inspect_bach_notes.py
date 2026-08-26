from midigpt import Score

score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
# Let's inspect bars 0-8
print("Bach BWV 1052a Bars 0-8 Notes:")
for t_idx in [0, 1, 2, 3]:
    t = score.tracks[t_idx]
    print(f"\nTrack {t_idx} (inst={t.instrument}, type={t.track_type}):")
    for bar_idx in range(8):
        bar = t.bars[bar_idx]
        notes_str = ", ".join(f"pitch={n.pitch}(val={n.pitch % 12}, onset={n.onset_ticks}, dur={n.duration_ticks})" for n in bar.notes)
        print(f"  Bar {bar_idx}: {notes_str}")
