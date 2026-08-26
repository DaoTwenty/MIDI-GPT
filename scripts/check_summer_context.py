from midigpt import Score

s1 = Score.from_midi("docs/assets/midi/ar_monophonic.mid")
s2 = Score.from_midi("docs/assets/midi/ar_chordal.mid")

print("ar_monophonic tracks:", len(s1.tracks))
print("ar_chordal tracks:", len(s2.tracks))

# Compare Track 1
t1_s1 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s1.tracks[1].bars for n in b.notes]
t1_s2 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s2.tracks[1].bars for n in b.notes]
print("Track 1 identical:", t1_s1 == t1_s2)

# Compare Track 2
t2_s1 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s1.tracks[2].bars for n in b.notes]
t2_s2 = [(n.pitch, n.onset_ticks, n.duration_ticks) for b in s2.tracks[2].bars for n in b.notes]
print("Track 2 identical:", t2_s1 == t2_s2)
