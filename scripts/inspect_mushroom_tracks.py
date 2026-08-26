from midigpt import Score

score = Score.from_midi("tmp/midi_examples/MKDDMushroomCity.mid")
for t_idx, t in enumerate(score.tracks):
    total_notes = sum(len(b.notes) for b in t.bars)
    if total_notes > 0:
        max_poly = 0
        for b in t.bars:
            onsets = {}
            for n in b.notes:
                onsets[n.onset_ticks] = onsets.get(n.onset_ticks, 0) + 1
            if onsets:
                max_poly = max(max_poly, max(onsets.values()))
        print(f"Track {t_idx} (inst={t.instrument}, type={t.track_type}): total_notes={total_notes}, max_polyphony={max_poly}")
