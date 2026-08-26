from midigpt import Score

print("Analyzing Bach (bwv1052a.mid)...")
score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
for t_idx, t in enumerate(score.tracks):
    if t_idx in [0, 3, 5]:
        print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}):")
        bar_counts = [len(b.notes) for b in t.bars[:16]]
        print(f"  Bars 0-15 note counts: {bar_counts}")

print("\nAnalyzing SSBM All Star Intro (SSBM_All_Star_Intro.mid)...")
score2 = Score.from_midi("tmp/midi_examples/SSBM_All_Star_Intro.mid")
for t_idx, t in enumerate(score2.tracks):
    print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}):")
    bar_counts = [len(b.notes) for b in t.bars]
    print(f"  Bars note counts: {bar_counts}")
