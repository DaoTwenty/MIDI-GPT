# MIDI-GPT Max MSP ↔ Server OSC Communication Protocol

**Version:** 0.2
**Transport:** OSC over UDP (bidirectional)
**Default ports:** Max → Server `7400`, Server → Max: replies to the UDP source address of incoming packets (override with `--reply-port`)

---

## 0. System Context

### What this system does

MIDI-GPT is a deep learning Transformer trained on MIDI music. It generates music token by token, conditioned on a surrounding musical context (other tracks, surrounding bars). This protocol connects it to a **live performance system** in Max MSP.

The scenario is: a musician (or multiple musicians) is performing live. Their notes are captured in Max MSP. A generative AI agent — powered by MIDI-GPT — listens to the performance and continuously generates one additional musical voice (the *agent track*), always staying a configurable number of bars ahead of playback. The result is a human-AI co-performance where the AI agent harmonically and rhythmically responds to the human input in real time.

### Roles

| Component | Role |
|-----------|------|
| **Max MSP** | Performance environment. Captures live MIDI notes, drives the bar clock, hosts the user interface (knobs, sliders), and plays back the generated agent output. |
| **MIDI-GPT Server** | Stateful generation backend. Maintains the full musical state (all tracks, all bars), manages the rolling context window, runs the model, and owns the agent track. |
| **OSC (this protocol)** | Thin communication layer between the two. Max pushes notes and parameters; the server pushes generated bars back. |

### Key constraints

- The model sees a **rolling window** of up to `model_dim` bars (typically 4–8). Older bars are discarded from the active context, but the full piece history is retained on the server for record-keeping.
- The model is **tempo-agnostic**: it does not see absolute timestamps or BPM. Musical time is quantized into subdivisions per beat (default: 12 per quarter note = 48 steps in 4/4).
- Generation is **always one or more bars at a time** for the agent track only, conditioned on all other tracks within the context window.
- The server decides *when* to generate based on `buffer_bars` and `lookahead_bars`. Max decides *what* is played by the humans and *when* bars end.
- All musical state (piece, parameters, track layout) lives on the server. Max is stateless with respect to the model — it only needs to know what notes to play back.

---

## 1. Overview

The server maintains the complete musical state: all track definitions, generation parameters, and the rolling context window. Max sends notes as they happen, advances the bar clock, and updates parameters via OSC. The server generates the agent track ahead of playback and pushes the result back to Max.

```
Max MSP                              MIDI-GPT Server
  │                                       │
  │── /session/init ──────────────────────▶│  reset state
  │◀── /capabilities ─────────────────────│  checkpoint feature flags
  │◀── /session/ready ────────────────────│
  │── /track/create ─────────────────────▶│  define tracks
  │── /param/set ──────────────────────── ▶│  set defaults
  │── /attr/set ────────────────────────── ▶│  set attribute overrides
  │── /session/start ─────────────────────▶│  begin
  │                                       │
  │ (music plays)                         │
  │── /note ────────────────────────────── ▶│  live notes land here
  │── /bar/end ──────────────────────────▶│  bar boundary triggers generation
  │◀── /generated/open ────────────────── │  start of note batch
  │◀── /generated/note ────────────────── │  individual generated notes
  │◀── /generated/close ───────────────── │  end of batch
  │◀── /generated/features ────────────── │  musical feature summary
  │◀── /sampled/attrs ─────────────────── │  attribute tokens the model sampled
  │◀── /prompt/state ──────────────────── │  debug: per-bar conditioning mask state
  │                                       │
  │── /track/remove ───────────────────── ▶│  mid-session track removal
  │── /param/set ───────────────────────▶│  knob moved → param update
  │── /session/stop ─────────────────────▶│  end session
```

---

## 2. Transport & Connection

| Property | Value |
|----------|-------|
| Protocol | OSC 1.0 over UDP |
| Max → Server port | `7400` (default; override with `--port` CLI flag) |
| Server → Max | Replies to the **UDP source address and port** of incoming packets |
| Fixed reply port | Use `--reply-port <N>` server flag to always reply to a fixed port regardless of source |
| Message encoding | Standard OSC type tags |

The server replies to the UDP source address (`ip:port`) of each incoming packet. Max does not need to declare a return address — it just needs to bind a consistent UDP port before sending `/session/init`. This means Max can run on any IP without server reconfiguration.

---

## 3. Conventions

### 3.1 Address Namespace

All messages are prefixed with `/midigpt/`.

### 3.2 Types

| OSC type | Used for |
|----------|---------|
| `i` (int32) | IDs, MIDI values, enum levels, bar/track indices |
| `f` (float32) | Normalized positions [0.0–1.0], temperature, ratios |
| `s` (string) | Names, status codes, error messages, JSON payloads |
| `T`/`F` | Boolean flags |

### 3.3 Normalized Time

Note positions and durations within a bar are sent as **float32 normalized values** in `[0.0, 1.0]`:
- `0.0` = bar start
- `1.0` = bar end (exclusive)

The server maps these to the model's quantized steps using its configured resolution (default: `resolution=12`, i.e. 48 steps per 4/4 bar). This decouples Max's timing from the model's internal quantization.

### 3.4 Track IDs

`track_id` is an **integer assigned by Max** at creation time. It persists for the session. Internally the server maps each `track_id` to a position index in the score; positions are assigned in creation order, starting from `0`.

### 3.5 Bar Index

`bar_index` is a **monotonically increasing integer** starting at `0`, incremented by Max with each `/bar/end` message. The server uses this to grow the piece and manage the rolling context window.

### 3.6 JSON Payloads

Several server→Max messages carry a single `s` argument containing a JSON string. These are marked `s:json` in the message definitions below. Clients should parse the string with a JSON library.

---

## 4. Session Lifecycle

```
UNINITIALIZED
     │
     ▼ /session/init
INITIALIZING
     │ /track/create (one or more)
     │ /param/set    (zero or more)
     │ /attr/set     (zero or more)
     │
     ▼ /session/start
RUNNING
     │ /note, /bar/end, /param/set, /attr/set, /track/create, /track/remove
     │ ◀── /generated/* (server → Max)
     │
     ▼ /session/stop
STOPPED
```

During `INITIALIZING`, the server accepts only session/track/param/attr messages. `/note` and `/bar/end` are rejected (server sends `/midigpt/error`).

During `RUNNING`, `/track/create` and `/track/remove` are allowed between bar boundaries (i.e., after `/bar/end` and before the next `/note`).

**`/session/start` preconditions** — the server validates these and returns `/midigpt/error` if violated:
- At least one agent track and one conditioning track must have been created.
- `buffer_bars >= 2`
- `buffer_bars <= model_dim`

---

## 5. Messages: Max → Server

### 5.1 Session Management

---

#### `/midigpt/session/init`

```
/midigpt/session/init  s:session_name
```

Resets all server state. Begins initialization phase. The server captures the UDP source address of this packet and uses it as the return address for all subsequent server→Max messages.

| Argument | Type | Description |
|----------|------|-------------|
| `session_name` | `s` | Human-readable label (for logging) |

Server responds with `/midigpt/capabilities` (checkpoint feature flags) followed by `/midigpt/session/ready`.

---

#### `/midigpt/session/start`

```
/midigpt/session/start
```

Transitions from INITIALIZING → RUNNING. Server begins waiting for notes and bar advances. Preconditions: see §4.

Server responds with `/midigpt/session/started` or `/midigpt/error`.

---

#### `/midigpt/session/stop`

```
/midigpt/session/stop
```

Gracefully terminates the session. Server flushes any in-progress generation, transitions to STOPPED.

---

### 5.2 Track Management

---

#### `/midigpt/track/create`

```
/midigpt/track/create  i:track_id  i:instrument  i:track_type  i:is_agent
```

Defines a track. Can be sent during INITIALIZING or RUNNING (between bar boundaries).

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Unique track identifier assigned by Max |
| `instrument` | `i` | `0–139` | GM instrument number (see note below) |
| `track_type` | `i` | `10`=STANDARD, `11`=DRUM | Mirrors `midi::TRACK_TYPE` |
| `is_agent` | `i` | `0` or `1` | `1` = this is the generated track; only one agent track allowed |

**Instrument values:** 0–127 = General MIDI melodic instruments; 128–139 = individual drum instruments.

When called during RUNNING, the server inserts an empty track with `bar_index` empty bars to align with the current piece length.

---

#### `/midigpt/track/remove`

```
/midigpt/track/remove  i:track_id
```

Removes a track from the piece. Cannot remove the agent track while session is RUNNING (server returns error). Can be called during RUNNING between bar boundaries.

---

#### `/midigpt/track/set_ignore`

```
/midigpt/track/set_ignore  i:track_id  i:ignored
```

Marks a conditioning track as ignored (`1`) or active (`0`) for future generation calls. Ignored tracks remain in the piece but are excluded from the model's conditioning context.

---

### 5.3 Note Input

---

#### `/midigpt/note`

```
/midigpt/note  i:track_id  i:pitch  i:velocity  f:onset  f:duration  i:bar_index
```

Sends a complete note event to the server. Notes for the agent track (`is_agent=1`) are ignored — the server owns that track's content.

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Which track this note belongs to |
| `pitch` | `i` | `0–127` | MIDI pitch |
| `velocity` | `i` | `1–127` | MIDI velocity (note-on only; note-offs are implicit via `duration`) |
| `onset` | `f` | `[0.0, 1.0)` | Normalized position within bar |
| `duration` | `f` | `(0.0, 1.0]` | Normalized duration; clamped to bar end if `onset + duration > 1.0` |
| `bar_index` | `i` | `≥ 0` | Which bar this note belongs to |

Notes can arrive out of order (Max may buffer). The server accumulates notes per `(track_id, bar_index)` until that bar is finalized by `/bar/end`.

---

### 5.4 Bar Control

---

#### `/midigpt/bar/end`

```
/midigpt/bar/end  i:bar_index  i:ts_numerator  i:ts_denominator
```

Signals that bar `bar_index` has ended and all notes for that bar have been sent. This is the bar clock. The server finalizes the bar, extends the piece, and may trigger generation depending on `buffer_bars` and `lookahead_bars`.

| Argument | Type | Description |
|----------|------|-------------|
| `bar_index` | `i` | The bar that just ended |
| `ts_numerator` | `i` | Time signature numerator (e.g. `4`) |
| `ts_denominator` | `i` | Time signature denominator (e.g. `4`) |

Time signature applies to all tracks for this bar.

---

### 5.5 Parameter Updates

All parameter updates take effect on the **next generation call**. Parameters are persistent unless sent via the one-shot variant.

---

#### `/midigpt/param/set`

```
/midigpt/param/set  s:param_name  <value>
```

Persistent parameter update. Value type depends on parameter (see §7).

---

#### `/midigpt/param/set_once`

```
/midigpt/param/set_once  s:param_name  <value>
```

One-shot parameter update. Applied to the next generation call only, then reverted to the current persistent value. Use this for volatile parameters like `temperature` where forgetting to reset could corrupt generation.

---

#### `/midigpt/param/reset`

```
/midigpt/param/reset  s:param_name
```

Resets a parameter to its default value.

---

#### `/midigpt/param/reset_all`

```
/midigpt/param/reset_all
```

Resets all parameters to defaults.

---

#### `/midigpt/track/param/set`

```
/midigpt/track/param/set  i:track_id  s:param_name  <value>
```

Sets a per-track parameter on a specific track (see §7.2).

---

### 5.6 Attribute Overrides

Attribute overrides steer the model's generation toward a specific musical characteristic (e.g., density, polyphony) for the agent track. These are distinct from parameters: they inject target attribute tokens directly into the model's input rather than modifying the sampling strategy.

Which attributes are available depends on the checkpoint. Check the `capabilities` JSON received after `/session/init` (see §6.1).

---

#### `/midigpt/attr/set`

```
/midigpt/attr/set  s:attr_name  i:value
```

Sets an attribute override. Persists until explicitly cleared.

| Argument | Type | Description |
|----------|------|-------------|
| `attr_name` | `s` | Attribute name (see §8) |
| `value` | `i` | Attribute level (≥ 0). Send `-1` or any negative value to clear this attribute override |

---

#### `/midigpt/attr/reset_all`

```
/midigpt/attr/reset_all
```

Clears all active attribute overrides. The model will freely sample all attributes on the next generation call.

---

## 6. Messages: Server → Max

### 6.1 Session Responses

| Message | Arguments | Sent when |
|---------|-----------|-----------|
| `/midigpt/capabilities` | `s:json` | Immediately after `/session/init` (before `/session/ready`) |
| `/midigpt/session/ready` | — | `/session/init` accepted |
| `/midigpt/session/started` | — | `/session/start` accepted |
| `/midigpt/session/stopped` | — | `/session/stop` complete |

**`/midigpt/capabilities` JSON fields:**

```json
{
  "tension":                true,
  "note_density":           true,
  "min_polyphony":          true,
  "max_polyphony":          true,
  "min_note_duration":      true,
  "max_note_duration":      true,
  "supports_token_mask":    true,
  "supports_attention_mask":      true,
  "supports_attention_approx":    true,
  "supports_attention_skip":      true,
  "supports_remove":        true
}
```

Boolean `true`/`false` flags for each attribute and mask mode. Hide UI controls for features that are `false` — sending an unsupported `attr_name` or `mask_mode` value will result in an error response.

---

### 6.2 Generated Notes

The server pushes notes back to Max as a **flat list**, framed by open/close delimiters. The batch may contain a single note, a handful of notes, or a full bar's worth. Max treats them all the same: each note carries enough information (`track_id`, `bar_index`, `onset`) to be placed in the right position.

---

#### `/midigpt/generated/open`

```
/midigpt/generated/open  i:track_id  i:bar_index  i:note_count
```

Signals the start of a note batch. `note_count` tells Max how many `/generated/note` messages follow. A batch with `note_count=0` is valid (silent bar).

---

#### `/midigpt/generated/note`

```
/midigpt/generated/note  i:track_id  i:bar_index  i:pitch  i:velocity  f:onset  f:duration
```

A single generated note. Field semantics are symmetric with the incoming `/midigpt/note` message.

| Argument | Type | Range | Description |
|----------|------|-------|-------------|
| `track_id` | `i` | `≥ 0` | Agent track ID |
| `bar_index` | `i` | `≥ 0` | Bar this note belongs to |
| `pitch` | `i` | `0–127` | MIDI pitch |
| `velocity` | `i` | `1–127` | MIDI velocity |
| `onset` | `f` | `[0.0, 1.0)` | Normalized onset within bar |
| `duration` | `f` | `(0.0, 1.0]` | Normalized duration |

---

#### `/midigpt/generated/close`

```
/midigpt/generated/close  i:track_id  i:bar_index
```

Signals end of batch. Max can safely schedule or commit all notes from this batch once it receives this message.

---

#### `/midigpt/generated/features`

```
/midigpt/generated/features  i:track_id  i:bar_index
    f:note_density  f:mean_pitch  f:mean_velocity
    i:max_polyphony  f:mean_duration
    i:min_polyphony  i:min_note_duration  i:max_note_duration
```

Musical features of the generated batch, sent immediately after `/generated/close`. Omitted if the batch is empty (silent bar).

| Feature | Type | Description |
|---------|------|-------------|
| `note_density` | `f` | Notes per beat |
| `mean_pitch` | `f` | Average MIDI pitch |
| `mean_velocity` | `f` | Average MIDI velocity |
| `max_polyphony` | `i` | Maximum simultaneous notes at any onset |
| `mean_duration` | `f` | Mean note duration (normalized) |
| `min_polyphony` | `i` | Minimum simultaneous notes at any onset |
| `min_note_duration` | `i` | Shortest note duration (in ticks) |
| `max_note_duration` | `i` | Longest note duration (in ticks) |

`min_note_duration` and `max_note_duration` are raw tick values (not normalized), matching the model's internal quantization. They bracket the sampled `MinNoteDuration`/`MaxNoteDuration` attribute tokens.

---

### 6.3 Attribute Feedback

---

#### `/midigpt/sampled/attrs`

```
/midigpt/sampled/attrs  s:json
```

Sent once per generation call, after the last `/generated/close`. Reports which attribute tokens the model actually sampled for the agent track. Useful for driving visualizations or confirming that attribute overrides were honored.

**JSON fields:**

```json
{
  "start_bar":   4,
  "end_bar":     5,
  "attrs": {
    "note_density":      3,
    "min_polyphony":     1,
    "max_polyphony":     2,
    "onset_polyphony":   2,
    "min_note_duration": 2,
    "max_note_duration": 4
  }
}
```

`start_bar`/`end_bar` define the generation window (half-open interval). `attrs` contains only the attribute names that were present in the decoded score; missing keys were not tokenized.

---

### 6.4 Prompt State (Debug)

---

#### `/midigpt/prompt/state`

```
/midigpt/prompt/state  s:json
```

Sent once per generation call, after the inference engine builds the prompt. Carries the per-bar conditioning mask layout visible to the model — useful for debugging and for studio visualizations.

**JSON fields:**

```json
{
  "target_bar":  4,
  "start_bar":   0,
  "end_bar":     5,
  "mask_mode":   "token",
  "human_masked_bars": [4],
  "agent_gen_bars":    [4],
  "agent_masked_bars": []
}
```

Bar indices are absolute (same coordinate space as `bar_index` in `/bar/end`).

---

### 6.5 Status & Errors

---

#### `/midigpt/status`

```
/midigpt/status  s:state
```

Emitted when the server's internal activity changes. `state` is one of: `"generating"`, `"ready"`.

---

#### `/midigpt/error`

```
/midigpt/error  i:code  s:message
```

| Code | Meaning |
|------|---------|
| `1` | Invalid session state (e.g. `/note` before `/session/start`) |
| `2` | Unknown track ID |
| `3` | Duplicate track ID |
| `4` | Invalid parameter name or value |
| `5` | Generation failed (model error or timeout) |
| `6` | Agent track missing |
| `7` | Multiple agent tracks defined |
| `8` | Note sent for agent track (ignored with warning) |

---

## 7. Parameter Reference

### 7.1 Global Parameters (`/midigpt/param/set`)

| Name | Type | Default | Range | One-shot recommended | Description |
|------|------|---------|-------|----------------------|-------------|
| `buffer_bars` | `i` | `4` | `2–64` | No | Bars to accumulate before the first generation is triggered; must be `≤ model_dim` |
| `lookahead_bars` | `i` | `1` | `1–8` | No | How many bars ahead of the playhead the server targets for generation |
| `num_anticipated_bars` | `i` | `1` | `1–8` | No | Number of bars generated in a single inference call |
| `model_dim` | `i` | `8` | `1–16` | No | Rolling context window size in bars; must be `≥ buffer_bars` |
| `temperature` | `f` | `1.0` | `0.1–5.0` | **Yes** | Softmax temperature for token sampling |
| `temperature_escalation` | `f` | `1.0` | `1.0–3.0` | No | Multiplies temperature on each retry attempt (helps escape repeated failures) |
| `top_p` | `f` | `1.0` | `0.0–1.0` | **Yes** | Nucleus sampling cutoff (1.0 = disabled) |
| `top_k` | `i` | `0` | `0–10000` | **Yes** | Top-k sampling (0 = disabled) |
| `mask_p` | `f` | `0.0` | `0.0–1.0` | No | Probability of masking each token in the conditioning context |
| `mask_k` | `i` | `0` | `0–10000` | No | Number of top-probability conditioning tokens to mask (0 = disabled) |
| `mask_mode` | `s` | `"token"` | see below | No | Mechanism used to apply conditioning masks (checkpoint-dependent; check `capabilities`) |
| `warmup_policy` | `s` | `"a_empty"` | see below | No | Strategy for filling the agent prefix during warmup (before steady-state) |
| `adapt_buffer` | `T/F` | `True` | bool | No | Dynamically lower the effective buffer during warmup so generation starts earlier |
| `novelty_check` | `T/F` | `True` | bool | No | Reject a generated bar if it repeats recent agent output |
| `silence_check` | `T/F` | `True` | bool | No | Reject a completely silent generated bar |
| `sampling_seed` | `i` | `-1` | any | **Yes** | RNG seed (-1 = random) |
| `max_attempts` | `i` | `3` | `1–10` | No | Max retry attempts per inference call before giving up |
| `gen_timeout` | `f` | `0.0` | `≥ 0` | No | Inference hard timeout in seconds (0 = disabled) |
| `polyphony_hard_limit` | `i` | `0` | `0–128` | No | Post-generation filter: drop notes beyond this simultaneous count (0 = off) |
| `density_hard_limit` | `i` | `0` | `0–1000` | No | Post-generation filter: drop notes beyond this per-bar count (0 = off) |

**`mask_mode` values:**

| Value | Description |
|-------|-------------|
| `"token"` | Insert `[MASK]` tokens (requires `supports_token_mask` in capabilities) |
| `"attention"` | Zero attention weights to masked positions (exact) |
| `"attention_approx"` | Approximate attention masking (faster) |
| `"attention_skip"` | Skip masked positions entirely (fastest exact mode) |
| `"remove"` | Remove masked tokens from the sequence |

**`warmup_policy` values:**

| Value | Description |
|-------|-------------|
| `"a_empty"` | Fill agent prefix bars with silence; model sees them as empty context |
| `"a_masked"` | Mask agent prefix bars so the model doesn't see them at all |
| `"b"` | Autoregress the full agent prefix in one bootstrap pass before generating the target bar |
| `"b_collapse"` | Like `"b"` but trims leading bars to fit within a single model window |

---

### 7.2 Per-Track Parameters (`/midigpt/track/param/set`)

#### Agent track (any type)

| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `temperature` | `f` | `1.0` | `0.1–5.0` | Per-track sampling temperature override |
| `min_pitch` | `i` | `0` | `0–127` | Minimum MIDI pitch constraint |
| `max_pitch` | `i` | `127` | `0–127` | Maximum MIDI pitch constraint |

#### Conditioning tracks

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ignore` | `i` | `0` | `1` = exclude this track from model conditioning |
| `temperature` | `f` | `1.0` | Reserved (not currently used for conditioning) |

---

## 8. Attribute Reference (`/midigpt/attr/set`)

Attribute overrides inject a specific discrete level as the target value for the corresponding attribute token on the agent track. Level 0 always means "any / model chooses freely".

Which attributes are supported depends on the checkpoint. Check the `capabilities` JSON after `/session/init`.

| Name | Level 0 | Level range | Description |
|------|---------|-------------|-------------|
| `note_density` | ANY | 1–N | Target note density per bar (higher = more notes) |
| `min_polyphony` | ANY | 1–6 | Minimum simultaneous notes at any onset |
| `max_polyphony` | ANY | 1–6 | Maximum simultaneous notes at any onset |
| `min_note_duration` | ANY | 1–6 | Minimum note duration quantile (1 ≈ 32nd note … 6 ≈ whole note) |
| `max_note_duration` | ANY | 1–6 | Maximum note duration quantile |
| `tension` | ANY | 1–N | Harmonic tension level |

Send a negative value (e.g., `-1`) to `/midigpt/attr/set` to clear a specific attribute override and let the model sample it freely.

---

## 9. Generation Triggering Logic (Server-Side Reference)

*This section describes server behavior for protocol completeness — not an OSC specification.*

**Warmup phase:** until `bars_completed + lookahead_bars >= buffer_bars` (or `bars_completed >= buffer_bars` if `adapt_buffer=False`), no generation is triggered. This ensures the model has enough context before the first inference call.

**Steady state:** the server generates bar `B = bars_completed + lookahead_bars` for the agent track when that threshold is first crossed after each `/bar/end`. The generation window covers `num_anticipated_bars` bars starting at `B`.

The server constructs:
- A rolling **score window** of up to `model_dim` bars across all tracks
- A **generation request** with the agent track's target bars marked for autoregressive filling and conditioning tracks masked past the playhead
- An **inference config** from the current global parameters

On completion the generated bars are merged back into the agent track state and sent to Max via `/generated/*`.

If inference is already in progress when a new generation would be triggered, the new request is dropped (the queue depth is 1). The `/midigpt/status "generating"` / `"ready"` messages bracket each inference call.

---

## 10. Example Flow

### Session initialization

```
Max → Server:  /midigpt/session/init "live-performance"
Server → Max:  /midigpt/capabilities '{"tension":true,"note_density":true,...}'
Server → Max:  /midigpt/session/ready

Max → Server:  /midigpt/track/create 0 0 10 0      # track 0: piano, melodic, conditioning
Max → Server:  /midigpt/track/create 1 25 10 0     # track 1: acoustic guitar, conditioning
Max → Server:  /midigpt/track/create 2 0 10 1      # track 2: piano, AGENT

Max → Server:  /midigpt/param/set "buffer_bars" 4
Max → Server:  /midigpt/param/set "lookahead_bars" 1
Max → Server:  /midigpt/param/set "model_dim" 8
Max → Server:  /midigpt/param/set "temperature" 1.0

Max → Server:  /midigpt/session/start
Server → Max:  /midigpt/session/started
```

### Running — bars 0–3 (warmup, no generation yet)

```
Max → Server:  /midigpt/note 0 60 80 0.0 0.25 0
Max → Server:  /midigpt/note 1 55 90 0.0 1.0 0
Max → Server:  /midigpt/bar/end 0 4 4
# bars_completed=1; buffer_bars=4; 1+1 < 4 → no generation
...
Max → Server:  /midigpt/bar/end 3 4 4
# bars_completed=4; 4+1 >= 4 → generation of bar 4 (agent) triggered
```

### Generated notes response

```
Server → Max:  /midigpt/status "generating"
Server → Max:  /midigpt/prompt/state '{"target_bar":4,"start_bar":0,"end_bar":5,...}'
Server → Max:  /midigpt/generated/open 2 4 5
Server → Max:  /midigpt/generated/note 2 4 65 82 0.0 0.25
Server → Max:  /midigpt/generated/note 2 4 69 78 0.25 0.25
Server → Max:  /midigpt/generated/note 2 4 72 90 0.0 0.5
Server → Max:  /midigpt/generated/note 2 4 65 85 0.5 0.25
Server → Max:  /midigpt/generated/note 2 4 67 80 0.75 0.25
Server → Max:  /midigpt/generated/close 2 4
Server → Max:  /midigpt/generated/features 2 4 2.0 67.4 83.0 3 0.28 1 12 48
Server → Max:  /midigpt/sampled/attrs '{"start_bar":4,"end_bar":5,"attrs":{"note_density":3,"max_polyphony":2}}'
Server → Max:  /midigpt/status "ready"
```

### Attribute override — steer toward sparse texture

```
Max → Server:  /midigpt/attr/set "note_density" 1     # level 1 = sparse
Max → Server:  /midigpt/attr/set "max_polyphony" 1    # max 1 note at a time
# Takes effect on the next generation call
```

### Clear attribute override

```
Max → Server:  /midigpt/attr/set "note_density" -1    # -1 clears override
# Or clear everything at once:
Max → Server:  /midigpt/attr/reset_all
```

### One-shot temperature spike

```
Max → Server:  /midigpt/param/set_once "temperature" 1.6
# Next generation uses 1.6, then reverts to 1.0
```

### Track removal mid-session

```
Max → Server:  /midigpt/bar/end 7 4 4     # bar boundary first
Max → Server:  /midigpt/track/remove 1
# Server removes track; future generation ignores it
```

---

## 11. CLI Reference

```
midigpt-server --ckpt PATH --port 7400 --host 0.0.0.0
               [--reply-port N]
               [--buffer N] [--lookahead N] [--model_dim N]
               [--gen_timeout SECS] [--max_attempts N]
               [--log_level DEBUG|INFO|WARNING|ERROR]
```

| Flag | Description |
|------|-------------|
| `--ckpt` | Path to packed `.pt` checkpoint bundle (required) |
| `--port` | UDP port to listen on; default `7400` |
| `--host` | Bind address; default `0.0.0.0` |
| `--reply-port` | Always reply to this port regardless of sender's source port |
| `--buffer` | Override `buffer_bars` at startup |
| `--lookahead` | Override `lookahead_bars` at startup |
| `--model_dim` | Override `model_dim` at startup |
| `--gen_timeout` | Override `gen_timeout` at startup |
| `--max_attempts` | Override `max_attempts` at startup |
| `--log_level` | Logging verbosity |

---

## 12. Open Questions / Future Extensions

| Topic | Note |
|-------|------|
| **Pre-recorded loops** | A `/track/load_bar` message could inject pre-computed bars into a conditioning track instead of live notes. |
| **Multiple agent tracks** | Currently one agent track only. Multi-agent (co-generate drums + melody) would require coordinated bar scheduling. |
| **MIDI channel binding** | Max could optionally announce a MIDI channel per track (`/track/bind_channel`) for automatic note routing. |
| **Piece persistence** | A `/session/save` / `/session/load` message pair could dump/restore the full piece state for session continuity. |
| **Per-bar attribute overrides** | Current `/attr/set` applies to the entire generation step window. A per-bar variant could target individual bars. |
| **Tension attribute** | `tension` is listed in capabilities but bar-level tension is not yet decoded into `/sampled/attrs`. |
