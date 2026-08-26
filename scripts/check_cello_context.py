from midigpt import Score

s1 = Score.from_midi("docs/assets/midi/sampling_cello_staccato.mid")
s2 = Score.from_midi("docs/assets/midi/sampling_cello_sustained.mid")

print("sampling_cello_staccato tracks:", len(s1.tracks))
print("sampling_cello_sustained tracks:", len(s2.tracks))

# Compare Tracks 0, 1, 2
for idx in range(3):
    t_s1 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s1.tracks[idx].bars for n in b.notes]
    t_s2 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s2.tracks[idx].bars for n in b.notes]
    print(f"Track {idx} identical:", t_s1 == t_s2)
