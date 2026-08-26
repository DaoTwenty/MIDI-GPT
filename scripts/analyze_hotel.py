from midigpt import Score

print("Analyzing Hotel California...")
score = Score.from_midi("tmp/midi_examples/HotelCalifornia.mid")
for t_idx, t in enumerate(score.tracks):
    print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}):")
    # Print note counts for the first 16 bars
    bar_counts = [len(b.notes) for b in t.bars[:24]]
    print(f"  Bars 0-23 note counts: {bar_counts}")

print("\nAnalyzing Everything Is In Its Right Place...")
score2 = Score.from_midi("tmp/midi_examples/EverythingIsInItsRightPlace.mid")
for t_idx, t in enumerate(score2.tracks):
    print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}):")
    bar_counts = [len(b.notes) for b in t.bars[:32]]
    print(f"  Bars 0-31 note counts: {bar_counts}")

print("\nAnalyzing Summer...")
score3 = Score.from_midi("tmp/midi_examples/Summer.mid")
for t_idx, t in enumerate(score3.tracks):
    if t_idx in [0, 1]:
        print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}):")
        bar_counts = [len(b.notes) for b in t.bars[:32]]
        print(f"  Bars 0-31 note counts: {bar_counts}")
