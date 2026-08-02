# Tools

`generate_vocal_track` and `check_vocal_track_status` are a start/poll
pair, not two independent tools — see why in
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md#why-generation-is-a-startpoll-pair-not-one-blocking-call).
Every job-based tool below (`generate_vocal_track`,
`generate_vocal_track_takes`, `split_vocal_stems`, `analyze_reference_audio`,
`transcribe_instrumental_to_midi`, `edit_audio_track`,
`prepare_voice_reference`) shares the same `check_vocal_track_status`
poll tool and job registry.

**No tool ever returns audio inline** — every result is a real,
absolute file path plus small scalar metadata (durations, seeds,
BPM/key, note counts), never raw bytes/base64/embedded audio content,
regardless of how large the underlying file is. See
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for why. `get_midi_notes` is
the one tool that returns real content instead of just a path (actual
note data) — it's paginated and hard-capped server-side
(`MAX_MIDI_NOTES_PER_PAGE`) regardless of what a caller requests.

## `generate_vocal_track(caption, lyrics, reference_audio_path=None, reference_youtube_url=None, advanced_settings=None, output_format="wav", remix_source_path=None, remix_source_youtube_url=None, remix_strength=0.5, remix_melody_retention=None, remix_no_fsq=False, song_title=None, split_stems=False, lora_path=None) -> dict`

Starts generating a complete original track — vocals and instrumentation
together — via ACE-Step 1.5. **Returns immediately** with a `job_id`; it
does not wait for generation to finish. Renders a complete song, not an
isolated vocal — don't call `split_vocal_stems` unless the user
explicitly asks for stems.

**Arguments**

- `caption` *(required)* — genre, mood, instrumentation, and vocal-style
  description, e.g. `"melodic dubstep, female vocals, dreamy, atmospheric,
  150 BPM"`. Never name a real artist — describe their characteristic
  sound instead.
- `lyrics` *(required)* — full original lyrics using ACE-Step's section
  tags: `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`, `[outro]`, or
  `[instrumental]`/`[inst]` for a section with no vocals. Must be
  original, not reproduced from a real song.
- `reference_audio_path` *(optional)* — a local audio file to use as a
  style reference. Confirmed to risk detuning the generated vocal — warn
  the user before using it, don't offer it unprompted. Mutually
  exclusive with `reference_youtube_url`.
- `reference_youtube_url` *(optional)* — same as `reference_audio_path`,
  downloaded automatically first. Same caveats.
- `advanced_settings` *(optional)* — a map of exact ACE-Step UI field
  names to override values, e.g. `{"Guidance Scale": 8.5, "Seed":
  "12345"}`. Every field not listed keeps ACE-Step's own default.
- `output_format` *(optional, default `"wav"`)* — one of `"wav"`, `"flac"`,
  `"mp3"`, `"opus"`, `"aac"`, `"wav32"`. Defaults to uncompressed WAV for
  DAW editing; pass `"mp3"` when file size matters more than edit
  quality.
- `remix_source_path` / `remix_source_youtube_url` *(optional)* —
  **actively discouraged.** Switches to ACE-Step's Remix mode. Confirmed
  to risk both vocal detuning and the source's own recognizable
  melody/lyrics bleeding into the output even with completely different
  supplied lyrics — a real copyright exposure, not just a quality issue.
  Mutually exclusive with each other and with `reference_audio_path`/
  `reference_youtube_url`. Prefer describing the desired vibe in
  `caption` instead.
- `remix_strength` *(optional, default `0.5`)* — 0.0-1.0. Confirmed not
  to reliably stop the source's own content from bleeding through even
  at low values (0.3 tested, still bled through).
- `remix_melody_retention` *(optional)* — 0.0-1.0 override for Remix
  mode's "Cover Strength (Melody Retention)". Confirmed not to fix
  garbled vocals — leave unset absent a specific, confirmed reason.
- `remix_no_fsq` *(optional, default `False`)* — Remix mode override.
  Confirmed to trade garbled vocals for a near-exact copy of the source
  instead, not a fix.
- `song_title` *(optional)* — a real title for this song, used to name
  the output file instead of a bare timestamp/UUID. Always provide one
  (write your own if the user hasn't given one).
- `split_stems` *(optional, default `False`)* — if `True`, also splits
  the finished mix into `vocals_path`/`instrumental_path` in this same
  job (via the same vocal-Roformer model `split_vocal_stems` uses).
  Prefer this over a separate `split_vocal_stems` call when you already
  know you'll want stems. Only does the vocals/instrumental split — for
  drums/bass/guitar/piano/other, call `split_vocal_stems` separately
  with `extra_stems` afterward.
- `lora_path` *(optional)* — path to a trained LoRA/LoKr adapter
  directory to load and enable before generating. Leave unset for
  normal generation.

**Returns**

```json
{ "job_id": "5e9d2b3a-..." }
```

**Raises** `SongForgeMCPError` — `MISSING_PARAMETER`/`INVALID_PARAMETER`
(empty/malformed caption or lyrics, both reference sources given at
once, both remix sources given at once, remix combined with reference,
a given path doesn't exist or isn't real audio, `output_format` not
supported), `VALUE_OUT_OF_RANGE` (`remix_strength`/`remix_melody_retention`
outside 0.0-1.0). Validation happens synchronously before the job
starts.

## `check_vocal_track_status(job_id, wait_seconds=25.0) -> list`

Polls a job started by any job-based tool on this server. If still
running, this blocks server-side for up to `wait_seconds` (capped at
25s) before replying — call again immediately on a `"running"` result
rather than adding your own delay between polls.

**Arguments**

- `job_id` *(required)* — the `job_id` a prior job-starting call
  returned.
- `wait_seconds` *(optional, default 25.0)* — how long this call may
  block if still running before returning `"running"` anyway.

**Returns** — a list whose first item is one of:

```json
{ "status": "running", "progress": 0.4, "message": "Generating - 40%" }
```
```json
{ "status": "error", "error": "[SYNTHESIS_FAILED] ACE-Step reported: ..." }
```
```json
{ "status": "complete", "...": "fields depend on which tool started the job — see that tool's own docs" }
```

No inline audio is ever attached — every path returned is real and
absolute, pass it directly to other tools.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND` (unrecognized
`job_id` — jobs live in memory only and don't survive a server
restart).

## `generate_vocal_track_takes(caption, lyrics, num_takes=3, reference_audio_path=None, reference_youtube_url=None, advanced_settings=None, output_format="wav", song_title=None) -> dict`

Same as `generate_vocal_track`, but runs `num_takes` (2-5) independent
generations sequentially, each with its own random seed (always
overridden per-take regardless of what `advanced_settings["Seed"]`
says), and measures each result with `analyze_reference_audio`'s same
analysis (BPM, key, mode) so results can be compared. Runs
sequentially, not in parallel — this server's GPU is a single shared
resource. Takes roughly `num_takes` times as long as a single
generation.

**Returns** `{"job_id": str}` — poll `check_vocal_track_status` exactly
as for `generate_vocal_track`. On completion, returns `{"takes": [...]}`,
one entry per take in order: `{"audio_path", "seed", "diagnostics", "measured": {"bpm", "key", "mode", "key_confidence"}}`.
A take that individually fails is recorded as `{"seed", "error"}`
instead — one bad take doesn't fail the whole job.

**Raises** `SongForgeMCPError` — same as `generate_vocal_track`, plus
`VALUE_OUT_OF_RANGE` if `num_takes` isn't between 2 and 5.

## `split_vocal_stems(audio_path, model=None, extra_stems=None) -> dict`

Starts splitting a full mix into a vocals-only stem and an
instrumental-only stem via `audio-separator`. **Returns immediately**
with a `job_id` — poll `check_vocal_track_status` exactly as for
`generate_vocal_track`. Idempotent per (`audio_path`, `model`) pair —
splitting the same file with the same model again returns the existing
stems instantly instead of re-running; a different model always
re-runs.

**Arguments**

- `audio_path` *(required)* — a file this server previously produced
  (typically `generate_vocal_track`'s `audio_path`). Must be inside
  this server's own output folder.
- `model` *(optional)* — `audio-separator` model filename as a literal
  string, e.g. `"vocals_mel_band_roformer.ckpt"` (default) or
  `"model_bs_roformer_ep_368_sdr_12.9628.ckpt"` (faster, slightly lower
  vocal isolation — try this first if turnaround matters more than
  quality, or as a retry if the default's results are bad on a
  dense/loud mix). Call `list_separator_models()` for the full ranked
  catalog — any `"filename"` value it returns is valid here.
- `extra_stems` *(optional)* — a Demucs model filename (e.g.
  `"htdemucs_6s.yaml"`) to also split the non-vocal instruments out of
  the mix. Runs as a second, independent pass on the same original
  `audio_path` — not chained off the instrumental stem, since Demucs
  models are trained on full mixes with vocals present. `"htdemucs_6s.yaml"`
  is the only model in the catalog with guitar/piano outputs at all
  (drums/bass/guitar/piano/other); `"htdemucs_ft.yaml"` gives
  higher-accuracy drums/bass/other with no guitar/piano. Known real
  limitation confirmed by ear (2026-08-03): guitar content on a real
  rock/pop track split across the `guitar` and `other` buckets rather
  than cleanly isolating — a precision limit of this model, not
  something a retry fixes. If this second pass fails, the whole job
  reports an error even if the vocals/instrumental split itself would
  have succeeded.

**Returns**

```json
{ "job_id": "5e9d2b3a-..." }
```

On completion:

```json
{
  "vocals_path": "C:\\...\\stems\\..._(vocals)_....wav",
  "instrumental_path": "C:\\...\\stems\\..._(other)_....wav",
  "model": "vocals_mel_band_roformer.ckpt",
  "extra_stems": {
    "drums_path": "...", "bass_path": "...", "guitar_path": "...",
    "piano_path": "...", "other_path": "..."
  },
  "extra_stems_model": "htdemucs_6s.yaml"
}
```

`extra_stems`/`extra_stems_model` are only present when `extra_stems`
was requested.

Separation quality is good but not perfect — some bleed between vocals
and instrumentation (both directions; confirmed on a real EDM track:
vocal bleed into the piano/keys stem) is an observed limitation of
these models, not something a retry will fix.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path exists but isn't inside this server's output folder, or isn't a
real audio file), `SEPARATOR_NOT_CONFIGURED` (see
[`docs/INSTALLATION.md`](INSTALLATION.md)), `SUBPROCESS_TIMEOUT`,
`SEPARATION_FAILED`.

## `list_separator_models(vocal_only=True) -> dict`

Lists `audio-separator`'s available models, sorted by vocal SDR score
descending. Use to pick or suggest a model for `split_vocal_stems`'s
`model` or `extra_stems` params when the defaults aren't separating
cleanly. Fast, synchronous, no job polling needed.

**Arguments**

- `vocal_only` *(optional, default `True`)* — exclude models with no
  vocal-stem score at all (denoise/deverb/crowd-removal models etc.).
  Set `False` to see the full catalog, including Demucs models for
  `extra_stems`.

**Returns** `{"models": [{"filename", "arch", "scores", "friendly_name", "vocal_sdr"}, ...]}`
— `filename` is what to pass to `split_vocal_stems`'s `model` or
`extra_stems` params.

## `play_audio(audio_path) -> dict`

Plays a file this server previously produced, on demand. A completed
`generate_vocal_track` already auto-plays its result via the OS's
default app — this tool is the fallback for when that didn't work or
wasn't seen, or for replaying an earlier result later in the
conversation.

**Arguments**

- `audio_path` *(required)* — a file this server previously produced
  (`audio_path`, `vocals_path`, `instrumental_path`, or any
  `extra_stems` path from an earlier completed job). Must resolve
  inside this server's own output folder.

**Returns** `{"status": "playing", "audio_path": ...}`.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path outside the output folder, or isn't a real audio file).

## `list_generated_tracks(limit=50) -> list[dict]`

Lists finished tracks sitting in this server's output folder, newest
first — for when a past generation's `job_id` has been lost (jobs are
in-memory only) or you want to see what's already been made. Doesn't
include split-out stems, only full-mix renders.

**Returns** a list of `{"path", "filename", "size_bytes", "modified_at", "duration_seconds"}` (duration is best-effort, `None` if it couldn't be read).

## `list_recent_jobs(limit=20) -> list[dict]`

Lists recent background jobs from any of this server's job-based
tools, newest first — for when a `job_id` has been lost
mid-conversation.

**Returns** a list of `{"job_id", "status", "progress", "message", "created_at"}`
— call `check_vocal_track_status` for a job's full result.

## `delete_generated_track(audio_path) -> dict`

Moves a file this server previously produced (a full-mix render or a
stem) into a `.trash` subfolder — **not a permanent delete.**
Deliberately reversible: a calling model being wrong about what's safe
to remove should never be able to permanently destroy a generation. To
actually reclaim disk space, empty `.trash` yourself via File Explorer.

**Returns** `{"status": "moved_to_trash", "audio_path": ..., "trash_path": ...}`.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path outside the output folder), `SUBPROCESS_FAILED` (the underlying
OS move failed).

## `edit_audio_track(audio_path, trim_start_seconds=None, trim_end_seconds=None, fade_in_seconds=None, fade_out_seconds=None, output_format=None) -> dict`

Starts trimming and/or fading a file this server previously produced.
Writes a **new** file — never overwrites the original. `output_format`
is `"wav"`, `"flac"`, or `"mp3"` (defaults to the source file's own
format if it's one of these three, otherwise `"wav"`); mp3 output
requires `ffmpeg` on PATH (see `docs/INSTALLATION.md`).

**Returns** `{"job_id": str}` — poll `check_vocal_track_status` exactly
as for `generate_vocal_track`. On completion, returns `{"audio_path", "duration_seconds", "output_format"}`.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path outside the output folder, unsupported `output_format`, negative
fade values), `VALUE_OUT_OF_RANGE` (trim range outside the file's
duration, fades longer than the trimmed clip),
`SUBPROCESS_FAILED`/`SUBPROCESS_TIMEOUT` (mp3 conversion via ffmpeg).

## `analyze_reference_audio(audio_path) -> dict`

Starts measuring real BPM and musical key from an audio file (tempo
tracking + key-profile correlation) — not a guess. Feed the result into
`generate_vocal_track`'s `advanced_settings`, e.g.
`{"BPM (Beats Per Minute)": result["bpm"], "Key": f"{result['key']} {result['mode'].title()}"}`.
Does **not** detect chord progression — ACE-Step has no chord-sequence
input; identify the song and look up a real chord chart instead if one
is needed.

**Arguments**

- `audio_path` *(required)* — any real audio file, **not** restricted
  to this server's own output folder (same broader validation as
  `reference_audio_path` above).

**Returns** `{"job_id": str}` — poll `check_vocal_track_status` exactly
as for `generate_vocal_track`. On completion, returns `{"bpm", "key", "mode", "key_confidence", "duration_seconds"}`.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(not a real audio file).

## `transcribe_instrumental_to_midi(audio_path) -> dict`

Starts converting an instrumental audio file into MIDI via basic-pitch,
then splitting it into separate bass/melody/chords tracks by pitch
register and note-overlap density — **not** real
per-instrument-class separation, but genuinely separate,
DAW-assignable parts instead of one flat blob. No drums/percussion
captured, pitched content only. Idempotent — calling again on an
already-transcribed file returns the existing MIDI files instantly.

Typical flow: `generate_vocal_track` → `split_vocal_stems` for the
instrumental → this tool.

**Arguments**

- `audio_path` *(required)* — a file this server previously produced
  (typically an instrumental stem). Must be inside this server's own
  output folder.

**Returns** `{"job_id": str}` — poll `check_vocal_track_status` exactly
as for `generate_vocal_track`. On completion, returns `midi_path`/`note_count`
for the flat transcription, plus `bass_midi_path`, `melody_midi_path`,
`chords_midi_path` (each with its own `_note_count`) for the split
tracks.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path outside the output folder, not real audio), plus a transcription
failure wraps the underlying model error.

## `get_midi_notes(midi_path, offset=0, max_results=500) -> dict`

Returns the actual note data (pitch, start, end, velocity) from a MIDI
file this server produced — real ground truth, not a summary. Use
before recreating, describing, or importing a transcribed MIDI file's
content anywhere — without calling this there is no real data to work
from, only a file path and note count; "recreating" notes without it
produces fabricated content.

Paginated since a real transcription can have hundreds of notes (510-821
observed in a single split track).

**Arguments**

- `midi_path` *(required)* — a `.mid` file this server previously
  produced. Must be inside this server's own output folder.
- `offset` *(optional, default 0)* — index of the first note to
  return, sorted by start time.
- `max_results` *(optional, default 500)* — maximum notes to return in
  one call. **Hard-capped at 500 server-side regardless of a higher
  value requested** — page with `offset` for more.

**Returns** `{"total_note_count", "returned_count", "offset", "notes": [{"pitch", "start", "end", "velocity"}, ...]}`.

**Raises** `SongForgeMCPError` — `FILE_NOT_FOUND`, `INVALID_PARAMETER`
(path outside the output folder, not a real MIDI file).

## `prepare_voice_reference(voice_name, youtube_url=None, local_audio_path=None) -> dict`

Checks whether reference vocal clips already exist for a named voice,
or prepares a new one from a YouTube link or a local audio file already
on disk.

Call with only `voice_name` first. Two possible results: `{"status": "found", ...}`
(includes `total_duration_seconds`/`meets_recommended_minimum` — one
song typically yields only 1-2 minutes of usable vocal, so a single
clip essentially never meets the recommended minimum of 600s), or
`{"status": "not_found", "message": ...}`.

Once given a `youtube_url` **or** a `local_audio_path` (exactly one),
call again with `voice_name` plus that one source to actually prepare
it.

**Arguments**

- `voice_name` *(required)* — the voice to check/prepare, e.g. `"Annika Wells"`.
- `youtube_url` *(optional)* — a YouTube link featuring this voice.
  Mutually exclusive with `local_audio_path`.
- `local_audio_path` *(optional)* — path to an audio file already on
  disk (any format/location — not restricted to this server's output
  folder, same broader validation as `reference_audio_path`). Mutually
  exclusive with `youtube_url`.

**Returns** `{"job_id": str}` when a source is given — poll
`check_vocal_track_status` exactly as for `generate_vocal_track`. On
completion, returns the same found-style status
(`clips`, `total_duration_seconds`, `meets_recommended_minimum`).
Returns the found/not_found dict directly (no job) when called with
only `voice_name`.

**Raises** `SongForgeMCPError` — `INVALID_PARAMETER` (both
`youtube_url` and `local_audio_path` given, or `local_audio_path` isn't
real audio).

## Example request

> "Write me an original melodic dubstep track in the style of Illenium —
> a female vocal, dreamy and a little melancholic, about holding on to a
> memory. Full verse/chorus/bridge structure. Give me the vocal
> separated out afterward so I can build my own instrumental around it."

Claude would work out the caption and full original lyrics, confirm them
with you, call `generate_vocal_track`, tell you generation has started,
poll `check_vocal_track_status` every 20-30 seconds — narrating progress
to you between calls — and once complete, call `split_vocal_stems` on
the result.
