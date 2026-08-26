# SoundFonts

GM-compatible SF2 files for `SoundFontSynthesizer` (training audio rendering).

The synthesizer loads all `.sf2` files in this directory and randomly selects one per
track for timbre diversity. Larger pools = more timbral variety.

## Download

All five recommended soundfonts (public domain / freeware):

```bash
# FluidR3_GM — General MIDI reference (141 MB)
curl -L -o FluidR3_GM.sf2 "https://www.filehorse.com/download-file/71796/"

# GeneralUser_GS — Lightweight, clean tones (30 MB)
curl -L -o GeneralUser_GS.sf2 "https://www.schristiancollins.com/generaluser.php"

# Arachno — Orchestral / Pad emphasis (148 MB)
curl -L -o Arachno.sf2 "https://www.arachnosoft.com/main/download/soundfont/Arachno%20SoundFont%20-%20Version%201.2b.sf2"

# Timbres_of_Heaven — Vivid orchestral (377 MB)
curl -L -o Timbres_of_Heaven.sf2 "https://www.filehorse.com/download-file/71798/"

# SGM_V201 — Balanced GM set (236 MB)
curl -L -o SGM_V201.sf2 "https://www.filehorse.com/download-file/71797/"
```

**Alternative sources** (search "free GM soundfont SF2 download"):
- https://miditoolbox.com/posts/best-free-general-midi-soundfonts-2026
- https://www.keppyeditor.com/soundfonts

## Adding your own

Drop any `.sf2` file here. Requirements:
- Must be GM-compatible (128 melodic instruments + percussion)
- Minimum 1 MB (filters out corrupt / tiny placeholder files)
- Larger files are fine — they're lazily loaded on first use

## Usage

The `SoundFontSynthesizer` in `src/python/midigpt/audio/synth.py` auto-discovers
all `.sf2` files in this directory:

```python
from midigpt.audio.synth import SoundFontSynthesizer

synth = SoundFontSynthesizer(target_sr=16000)  # uses soundfonts/ by default
audio = synth.render_track(midi_track, tempo=120)  # random SF2 per call
```

FluidSynth must be installed (`brew install fluidsynth`) and the library path
may need to be set:

```bash
export DYLD_LIBRARY_PATH=/usr/local/lib
```
