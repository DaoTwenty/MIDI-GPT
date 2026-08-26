# MIDI-GPT OSC Test Studio — Implementation Plan

## Overview

A browser-based test interface for the MIDI-GPT OSC server. Inspired by mmm-studio's
MVC + canvas architecture but purpose-built for real-time, continuous MIDI streaming.

```
Browser (JS) ←──WebSocket──→ Flask Bridge ←──OSC UDP──→ MidiGPT OSC Server
                              (port 5000)      7400/7401     (oracle/specter)
```

The browser cannot send UDP, so a Flask + flask-socketio bridge handles translation
and also serves the static frontend.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  MIDI-GPT Studio │ ● Connected │ BPM: [120] │ Bar: 12 │ 4/4     │
├──────────────┬───────────────────────────┬───────────────────────┤
│  Track List  │   Piano Roll (canvas)     │   Attribute Graph     │
│  ──────────  │   multi-track, scrollable │   realized vs request │
│  cond1 🔵    │                           │   — tension (solid)   │
│  cond2 🟢    │   [streamed notes in/out] │   ··tension req (dot) │
│  agent  🟠   │                           │   — density (solid)   │
│  [+ Track]   ├───────────────────────────┤                       │
│              │  ▶ Start Session  ■ Stop  │                       │
├──────────────┴───────────────────────────┴───────────────────────┤
│  Params: temp | model_dim | buffer_bars | lookahead | tension↑   │
│  MIDI CC: [bind CC → param]                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Backend: Flask WebSocket Bridge

**File:** `studio/app.py`

- `flask-socketio` WebSocket server (port 5000)
- OSC **client** → port 7400 (sends commands to OSC server)
- OSC **listener** on port 7401 (receives replies, forwards to browser via SocketIO)
- Serves static frontend files

**SocketIO event protocol (browser → bridge → OSC server):**

| SocketIO event | Maps to OSC address |
|----------------|---------------------|
| `session:init` | `/midigpt/session/init` |
| `session:start` | `/midigpt/session/start` |
| `session:stop` | `/midigpt/session/stop` |
| `track:create` | `/midigpt/track/create` |
| `track:remove` | `/midigpt/track/remove` |
| `note` | `/midigpt/note` |
| `bar:end` | `/midigpt/bar/end` |
| `param:set` | `/midigpt/param/set` |

**SocketIO events (OSC server → bridge → browser):**

| OSC address | SocketIO event |
|-------------|----------------|
| `/midigpt/session/ready` | `session:ready` |
| `/midigpt/session/started` | `session:started` |
| `/midigpt/generated/open` | `generated:open` |
| `/midigpt/generated/note` | `generated:note` |
| `/midigpt/generated/close` | `generated:close` |
| `/midigpt/generated/features` | `generated:features` |
| `/midigpt/status` | `status` |
| `/midigpt/error` | `error` |

---

## Frontend Modules

### 1. Track Model — two conditioning modes

Each conditioning track has a **mode**:

- **Loop mode**: upload a MIDI file (or draw notes) → plays endlessly, sends
  `/midigpt/note` + `/midigpt/bar/end` on each bar boundary via LoopPlayer.
  
- **Live MIDI In mode**: select Web MIDI device + channel → routes incoming
  notes in real-time to the OSC server.

The **agent track** is read-only — it receives generated notes from the server,
displays them bar-by-bar in the piano roll, and plays audio via Spessasynth.

### 2. Piano Roll (Canvas, `PianoRollView.js`)

- Color-coded tracks stacked vertically
- Loop tracks: show the looping window cycling behind a playhead
- Agent track: notes appear bar-by-bar as each `/midigpt/generated/close` arrives
- Zoom (Shift+scroll horizontal, Cmd+scroll vertical) + pan
- Playhead synced to Web Audio transport

### 3. Attribute Graph (Canvas, `AttributeGraphView.js`)

- **X-axis:** bar number; **Y-axis:** quantized level (0–9)
- **Dotted line** = requested (from param panel targets)
- **Solid line** = realized (from `/midigpt/generated/features` messages)
- One color-coded series per attribute: `tension`, `note_density`,
  `min_polyphony`, `max_polyphony`
- Updates in real-time as bars are generated

### 4. Parameter Panel (`ParamPanelView.js`)

OSC session params (sent via `/midigpt/param/set`):

| Param | Type | Default |
|-------|------|---------|
| `temperature` | float | 1.0 |
| `model_dim` | int | 4 |
| `buffer_bars` | int | 8 |
| `lookahead_bars` | int | 2 |
| `num_anticipated_bars` | int | 1 |
| `mask_gap` | bool | false |
| `adapt_buffer` | bool | true |
| `sampling_seed` | int | -1 |
| `gen_timeout` | float | 0 |

Agent attribute targets (per-track, passed via OSC before session start):

- `tension` (0–9)
- `note_density` (0–9)
- `min_polyphony` (0–9)
- `max_polyphony` (0–9)

**MIDI CC binding:** Click any param slider label → panel enters capture mode →
wiggle a CC knob on any connected MIDI device → binding saved.
On each subsequent CC message matching that binding, `param:set` fires automatically.

### 5. Sound Engine (`AudioController.js` — Spessasynth + `default.sf2`)

- Load `default.sf2` at startup (symlinked or copied from mmm-studio)
- Per-track synthesizer instance on a separate MIDI channel (1–16)
- GM program selected from track config instrument picker
- Schedules both human loop/live notes and incoming agent notes for synthesis
- Master volume + per-track volume sliders

### 6. MIDI Controller (`MidiController.js`)

- **Web MIDI API**: enumerate input/output devices; refresh on device change
- **Live MIDI In**: per-track device + channel selector; incoming note on/off
  converted to normalized `[0,1]` onset/duration and forwarded to OscBridge
- **Loop Player**: a Web Worker (`loop-worker.js`) drives the bar clock at
  precise intervals (based on BPM + time signature); sends note + bar events
  per iteration; loops back to bar 0 on completion
- **MIDI CC capture**: global CC listener; when in binding mode, first CC
  received is registered; subsequent CCs trigger param updates

### 7. OSC Bridge (`OscBridge.js`)

- Thin wrapper over `socket.io-client`
- `send(address, ...args)` → emits `osc:out` with `{address, args}`
- `on(address, fn)` → registers handler on `osc:in` filtered by address
- Manages connection state; emits `bridge:connected` / `bridge:disconnected`

---

## File Structure

```
src/python/midigpt/osc/studio/
├── app.py                          # Flask + flask-socketio OSC bridge
├── static/
│   ├── index.html
│   ├── css/
│   │   └── style.css               # Dark theme (same design language as mmm-studio)
│   └── js/
│       ├── main.js                 # App wiring entry point
│       ├── core/
│       │   └── mvc.js              # Observable / Model / View base classes
│       ├── models/
│       │   ├── SessionModel.js     # OSC session state machine + params
│       │   └── TrackModel.js       # Track: mode, notes, instrument, MIDI config
│       ├── views/
│       │   ├── TrackListView.js    # Sidebar: create / delete / config tracks
│       │   ├── PianoRollView.js    # Canvas piano roll (multi-track, streaming)
│       │   ├── AttributeGraphView.js  # Canvas line chart (realized vs requested)
│       │   ├── TransportView.js    # Start/Stop, BPM, bar counter, state badge
│       │   └── ParamPanelView.js   # OSC params + MIDI CC binding UI
│       └── controllers/
│           ├── AppController.js    # Session lifecycle, track registration
│           ├── OscBridge.js        # socket.io ↔ OSC message protocol
│           ├── MidiController.js   # Web MIDI, loop player, MIDI CC
│           └── AudioController.js  # Spessasynth synthesis
```

---

## New Dependencies

Added to `[realtime]` extra in `pyproject.toml`:

- `flask-socketio>=5.3`
- `eventlet>=0.35` (async backend for flask-socketio)

---

## Out of Scope (first version)

- Note editing inside the piano roll (loop notes loaded from MIDI file only)
- Multiple agent tracks (OSC server supports one)
- Exporting / recording generated output
- Multi-instrument per track
