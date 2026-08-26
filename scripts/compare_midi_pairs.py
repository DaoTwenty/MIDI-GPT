from midigpt import Score
import os

pairs = [
    ("docs/assets/midi/sampling_drums_sparse.mid", "docs/assets/midi/sampling_drums_dense.mid"),
    ("docs/assets/midi/sampling_cello_staccato.mid", "docs/assets/midi/sampling_cello_sustained.mid"),
    ("docs/assets/midi/infill_sparse.mid", "docs/assets/midi/infill_chordal.mid"),
    ("docs/assets/midi/ar_monophonic.mid", "docs/assets/midi/ar_chordal.mid"),
    ("docs/assets/midi/harp_monophonic.mid", "docs/assets/midi/harp_chordal.mid"),
]

for p1, p2 in pairs:
    name1 = os.path.basename(p1)
    name2 = os.path.basename(p2)
    try:
        s1 = Score.from_midi(p1)
        s2 = Score.from_midi(p2)
        
        # compare notes in all tracks
        notes1 = [[(n.pitch, n.onset_ticks, n.duration_ticks) for b in t.bars for n in b.notes] for t in s1.tracks]
        notes2 = [[(n.pitch, n.onset_ticks, n.duration_ticks) for b in t.bars for n in b.notes] for t in s2.tracks]
        
        identical = notes1 == notes2
        print(f"Pair: {name1} vs {name2} -> Identical: {identical}")
        if identical:
            print(f"  Note counts: {sum(len(t) for t in notes1)}")
    except Exception as e:
        print(f"Error comparing {name1} and {name2}: {e}")
