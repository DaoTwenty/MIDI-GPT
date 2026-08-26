from midigpt import Score

score = Score.from_midi("tmp/midi_examples/Summer.mid")
t = score.tracks[5]
notes = [len(b.notes) for b in t.bars[13:21]]
print(f"Track 5 (inst={t.instrument}, type={t.track_type}) note counts in bars 13-20: {notes}")
