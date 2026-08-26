from midigpt import Score

for filename in ["sampling_drums_sparse.mid", "sampling_drums_dense.mid", "sampling_cello_staccato.mid", "sampling_cello_sustained.mid"]:
    path = f"docs/assets/midi/{filename}"
    try:
        s = Score.from_midi(path)
        notes = [n for b in s.tracks[-1].bars for n in b.notes]
        pitches = sorted(list(set(n.pitch for n in notes)))
        print(f"{filename}: total_notes={len(notes)}, unique_pitches={len(pitches)} ({pitches})")
    except Exception as e:
        print(f"Error {filename}: {e}")
