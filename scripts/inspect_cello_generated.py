from midigpt import Score

for name in ["sampling_cello_staccato.mid", "sampling_cello_sustained.mid"]:
    path = f"docs/assets/midi/{name}"
    score = Score.from_midi(path)
    print(f"\nFile: {name} (Tracks: {len(score.tracks)})")
    # Cello is Track 3
    t = score.tracks[3]
    print(f"Track 3 (inst={t.instrument}, type={t.track_type}):")
    for bar_idx in range(8):
        bar = t.bars[bar_idx]
        pitches = [n.pitch for n in bar.notes]
        print(f"  Bar {bar_idx}: {pitches}")
