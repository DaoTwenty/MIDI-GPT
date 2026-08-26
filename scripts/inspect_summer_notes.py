from midigpt import Score

score = Score.from_midi("tmp/midi_examples/Summer.mid")
# Summer bars 13-20 (8 bars)
print("Summer Piano & Bass notes (Bars 13-20):")
for t_idx in [0, 1]:
    t = score.tracks[t_idx]
    print(f"\nTrack {t_idx} (inst={t.instrument}, type={t.track_type}):")
    for bar_idx in range(13, 21):
        bar = t.bars[bar_idx]
        notes_str = ", ".join(f"pitch={n.pitch}(onset={n.onset_ticks}, dur={n.duration_ticks})" for n in bar.notes)
        print(f"  Bar {bar_idx}: {notes_str}")
