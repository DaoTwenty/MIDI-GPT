from midigpt import Score
import os

midi_dir = "tmp/midi_examples"
files = [f for f in os.listdir(midi_dir) if f.endswith(".mid") or f.endswith(".midi")]

for filename in sorted(files):
    path = os.path.join(midi_dir, filename)
    try:
        score = Score.from_midi(path)
        print(f"\n=========================================\nFile: {filename} (Tracks: {len(score.tracks)})")
        
        # Check active tracks and look for sections of 8 bars where multiple tracks are active
        num_bars = len(score.tracks[0].bars)
        print(f"Total bars: {num_bars}")
        
        # We look for a window of 8 bars
        active_windows = []
        for start_bar in range(0, num_bars - 8, 4):
            # count active tracks in this 8-bar window
            active_tracks = 0
            track_notes = []
            for t_idx, t in enumerate(score.tracks):
                notes_in_window = sum(len(b.notes) for b in t.bars[start_bar : start_bar + 8])
                if notes_in_window > 0:
                    active_tracks += 1
                    track_notes.append((t_idx, t.instrument, t.track_type, notes_in_window))
            if active_tracks >= 2:
                active_windows.append((start_bar, active_tracks, track_notes))
        
        # Sort windows by number of active tracks, and then by total notes in active tracks
        active_windows.sort(key=lambda x: (x[1], sum(n[3] for n in x[2])), reverse=True)
        
        print("Top 3 multi-track 8-bar windows:")
        for win in active_windows[:3]:
            start_bar, count, t_notes = win
            print(f"  Bars {start_bar}-{start_bar+8}: {count} active tracks")
            for t_idx, inst, t_type, notes in t_notes[:6]:
                print(f"    Track {t_idx} (inst={inst}, type={t_type}): {notes} notes")
            if len(t_notes) > 6:
                print(f"    ... and {len(t_notes) - 6} more active tracks")
                
    except Exception as e:
        print(f"Error loading {filename}: {e}")
