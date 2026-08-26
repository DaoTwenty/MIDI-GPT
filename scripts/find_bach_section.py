from midigpt import Score

score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
num_bars = len(score.tracks[0].bars)

valid_sections = []
for start_bar in range(0, num_bars - 8):
    v1_notes = [len(b.notes) for b in score.tracks[0].bars[start_bar : start_bar + 8]]
    vc_notes = [len(b.notes) for b in score.tracks[3].bars[start_bar : start_bar + 8]]
    
    # We want cello to have at least 4 notes in every bar (moving bassline) and violin to also be active
    if all(n >= 4 for n in vc_notes) and all(n >= 2 for n in v1_notes):
        total_notes = sum(vc_notes) + sum(v1_notes)
        valid_sections.append((start_bar, sum(v1_notes), sum(vc_notes), v1_notes, vc_notes))

# Sort by cello note count
valid_sections.sort(key=lambda x: x[2], reverse=True)

print("Top sections for Bach (Violin + Cello):")
for start_bar, v1_tot, vc_tot, v1_notes, vc_notes in valid_sections[:10]:
    print(f"Start Bar: {start_bar}")
    print(f"  Violin 1 notes per bar: {v1_notes} (total={v1_tot})")
    print(f"  Cello notes per bar:    {vc_notes} (total={vc_tot})")
