# Architecture

How SongForge-MCP turns a style description and lyrics into a finished
track.

## Overview

```
┌──────────────┐    stdio     ┌───────────────────┐   browser automation   ┌──────────────┐
│  MCP Client  │◄────────────►│  SongForge-MCP  │◄───────────────────────►│  ACE-Step    │
│(AI assistant)│  (JSON-RPC)  │      FastMCP       │      (Playwright)       │  1.5 server  │
└──────────────┘              └───────────────────┘                        └──────────────┘
                                        │
                                        │ subprocess
                                        ▼
                                ┌───────────────────┐
                                │  audio-separator   │
                                │ (stem separation)  │
                                └───────────────────┘
```

## Why browser automation, not a direct API call

ACE-Step 1.5 exposes a Gradio web interface backed by a generation
endpoint that takes roughly seventy positional parameters, most of them
unlabeled in its own API schema. Reconstructing correct values for all of
them from the underlying source code was attempted directly three
separate times and produced unusable (static/noise) output every time,
even after independently fixing two real environment issues along the
way — the failure was in the reconstructed defaults themselves, not the
environment.

Driving ACE-Step's actual web interface programmatically instead — the
same interface a person would use, via [Playwright](https://playwright.dev/) —
sidesteps the problem entirely: every field this server doesn't
explicitly set keeps the interface's own real, working default. This is
the reason `generate_vocal_track`'s `advanced_settings` parameter takes
exact UI field labels rather than a fixed parameter list — it's the same
principle applied to intentional overrides.

`songforge_mcp/acestep_client.py` owns this integration, including two
reliability behaviors that were not obvious from a single successful
run:

- A short delay is enforced after clicking "Generate" before anything
  else happens. Navigating away immediately can abort the request on the
  server side before it is fully registered.
- The completion timeout is generous rather than tuned to the common
  case — supplying reference audio measurably shifts ACE-Step onto a
  slower internal code path, and a timeout sized for the fast path fails
  the slow one.

## What reference audio actually conditions on

ACE-Step's own UI text describes reference audio as controlling "vocal
timbre, mixing style, and overall atmosphere" — implying a narrow,
voice-only signal. Tracing the actual mechanism in ACE-Step's source
tells a different story: `infer_refer_latent()` in
`acestep/core/generation/handler/conditioning_embed.py` runs the
reference clip through `tiled_encode` — the same VAE audio encoder used
for real audio elsewhere in the pipeline — producing
`refer_audio_acoustic_hidden_states` that the diffusion model is
conditioned on directly. That's a real acoustic latent of the reference
clip's actual sound, not a narrow speaker/timbre embedding. Practically,
this means a reference clip's instrumentation and production texture
(not just its vocal character) plausibly influences the generation too —
this project's own docs previously claimed otherwise, copying ACE-Step's
UI description without verifying it against the code.

What's still unconfirmed: exactly how strongly this acoustic conditioning
competes against the `caption` text's own description when they pull in
different directions (e.g. a caption naming one instrument against a
reference clip dominated by another). That would need an actual
side-by-side listening comparison, which hasn't been run in this project.
Until it has, `generate_vocal_track`'s own docs treat `caption` as the
reliable lever for a specific instrument being prominent, and reference
audio as secondary reinforcement rather than a guarantee.

## Why output_format needs its own Settings-panel dance

`output_format` (default `"wav"`, overriding ACE-Step's own `"mp3"` UI
default) can't be set the same way as a normal `advanced_settings`
field. Reading ACE-Step's own source
(`acestep/ui/gradio/interfaces/user_preferences.py` and
`generation_advanced_output_controls.py`) showed the Audio Format control
is a global, `localStorage`-persisted user preference living in the
Settings panel (behind a collapsed "Audio Output & Post-processing"
accordion), not a field on the per-generation form — and it isn't a
native `<select>` either, it's a combobox (`role="listbox"` input) that
opens a `role="option"` list on click. `ACEStepClient._set_output_format`
drives that exact sequence: open Settings, expand the accordion, click
the combobox, click the matching option, close Settings. Every part of
this was confirmed against the live server before being written, not
assumed from source alone — reading the DOM (`get_by_role("option")`
returned the actual six format choices) and then running one real short
generation with the format set to WAV, which produced a genuine `.wav`
file on disk, not just a UI value change. The `_OUTPUT_FORMAT_EXTENSIONS`
mapping matters for finding the resulting file afterward: ACE-Step's own
`audio_utils.py` saves `wav32` with a plain `.wav` extension too, not
`.wav32`.

## Why generation is a start/poll pair, not one blocking call

`generate_vocal_track` originally blocked for the full duration of ACE-Step
cold-start plus generation and reported progress via MCP's
`ctx.report_progress`. A real call against Claude Desktop was measured
being cancelled by the client after a fixed **240-second** timeout, with
**zero** progress notifications having reached the client in that window.
Checking FastMCP's own source (`Context.report_progress` in
`mcp/server/fastmcp/server.py`) confirmed why: it's a silent no-op unless
the client supplied a `progressToken` on the original request, which
Claude Desktop doesn't. So the mechanism this server relied on to keep a
long call alive was never actually functioning with this client — not a
display issue, a delivery issue.

Since ACE-Step's own cold-start plus generation time is real (a checkpoint
load alone can take a while, and generation with reference audio has been
observed taking up to ~9 minutes) and can't be shortened, the fix is
architectural: `generate_vocal_track` now starts the work as a background
`asyncio.create_task` (`songforge_mcp/job_registry.py` tracks it by
`job_id` in memory) and returns in well under a second. A separate
`check_vocal_track_status(job_id, wait_seconds=25.0)` tool is what the
calling model polls — and it long-polls: if the job is still running, it
blocks server-side for up to `wait_seconds` before replying, rather than
returning "running" instantly. This exists specifically so the calling
model can just call it again immediately on a `"running"` result without
adding its own delay — every poll is a full model turn (a tool call plus
whatever text it generates), so minimizing how many polls a multi-minute
generation needs directly minimizes cost, independent of what gets
narrated.

The first version of this fix over-corrected in the other direction: the
injected instructions originally told the calling model to narrate every
single poll, on the reasoning that MCP's progress-notification mechanism
doesn't reach this client so chat narration was the only substitute. In
practice that meant five, ten, or more chat messages per generation for
information that hadn't meaningfully changed — expensive and, per direct
user feedback, "way too much feedback... spamming." The instructions now
call for at most a handful of messages per generation (start, maybe one
"still going" if it's taking a while, then done/error) — narration was
never the fix for the timeout bug in the first place; the job/poll split
itself was. MCP progress notifications are still emitted opportunistically
alongside all this (harmless for clients that do support `progressToken`)
but were never relied on for correctness.

`split_vocal_stems` was deliberately left as a single blocking call —
real separations were measured at 5-15 seconds, comfortably inside any
reasonable client timeout, so the added complexity of a job/poll pattern
isn't justified there.

Jobs live in memory only and do not survive a server restart — a
`job_id` from before a restart will 404 (`FILE_NOT_FOUND`) on
`check_vocal_track_status`, which is an acceptable tradeoff for this
server's own local/prototype use.

## Why a specific real artist's name is never passed through

`generate_vocal_track`'s documentation and this server's injected
instructions both direct the calling model to translate a named real
artist into descriptive genre/production language rather than pass the
name itself into `caption` or `lyrics`. Genre- and style-level prompting
("melodic dubstep, Illenium-adjacent atmosphere") is well-supported,
unremarkable use of a music generation model. Targeting a specific,
identifiable real person's voice or likeness by name is a materially
different and more legally sensitive act, particularly for a deceased
individual whose estate controls those rights — this server treats that
distinction as a hard line, not a style preference.

## Where generated output actually lives

`Paths.OUTPUT_DIR` (renders) and `Paths.REFERENCE_AUDIO_CACHE` default to
an `output/` folder inside this repo checkout
(`songforge_mcp_shared/constants._default_output_root()`), not the OS
temp directory. That was the original default and turned out to be a
real problem in practice: system cleanup tools (Windows Storage Sense,
etc.) can and do purge temp, backup software typically skips it, and it
isn't a location a user would think to check for a track they just asked
for and want to keep — confirmed the hard way when a completed track had
to be located by reading server logs instead of just knowing where to
look. Overridable via `SONGFORGE_OUTPUT_DIR` for anyone who wants
output somewhere else (e.g. a personal music archive folder), but the
built-in default deliberately isn't a path outside this repo either
(such as a sibling project folder specific to one developer's machine) —
this repo is meant to be cloned standalone by people with their own,
different directory layouts, so the default has to be self-contained.

## File access boundaries

Every path parameter accepted from a calling model is validated before
it's ever opened, uploaded to a browser session, or handed to a
subprocess (`songforge_mcp_shared/protocol.py`). Two distinct checks
exist, deliberately different in strictness:

- **`reference_audio_path`** (`generate_vocal_track`) — `validate_audio_file_path`
  resolves the real path, requires it to actually be a file, checks the
  extension against an allow-list, and confirms it parses as genuine
  audio content via `soundfile` rather than trusting the extension alone.
  It does **not** restrict which directory the file lives in — pointing
  at a personal sample library anywhere on disk is the legitimate use
  case this parameter exists for.
- **`audio_path`** (`split_vocal_stems`, `play_audio`,
  `delete_generated_track`, `edit_audio_track`) — `validate_output_dir_audio_path`
  applies the same checks, plus one more: the resolved path must live
  inside `Paths.OUTPUT_DIR`, the folder this server itself writes
  generated audio into. There's no legitimate reason for these
  parameters to point anywhere else — their only real input is a file
  `generate_vocal_track` (or a prior `split_vocal_stems`/`edit_audio_track`
  call) already produced — so a calling model (whether misled by a
  manipulated prompt or simply given a wrong path) cannot make this
  server read, play, edit, or delete arbitrary files elsewhere on the
  filesystem. `delete_generated_track` layers a further restriction of
  its own on top: it never actually erases anything, only moves the file
  into a `.trash` subfolder — deliberately reversible, since a calling
  model being wrong about what's safe to remove should never be able to
  permanently destroy a generation.

**No tool ever returns audio inline** (no MCP `AudioContent`/base64/raw
bytes) — every completed job's result is paths and small scalar
metadata only (durations, seed values, BPM/key, note counts, etc.),
regardless of how large the underlying file is. This was tried the
other way first (returning generated audio inline via FastMCP's `Audio`
helper) and reverted: not every client's chat UI renders that block
type (observed on Claude Desktop as an "unsupported format" message
even though generation had genuinely succeeded — confirmed by checking
the server's own log, which showed no error, and the output file, which
existed and played correctly), and it meant pushing a full audio
payload back through the conversation on every completion regardless.
`generate_vocal_track`'s job instead best-effort auto-launches the
finished file in the OS's own default player (`open_with_default_app`,
`songforge_mcp_shared/constants.py`) as a convenience — a failure there
(e.g. no default app registered) is swallowed and never turns a
successful generation into a reported error, since the real deliverable
is the `audio_path` already sitting in `job.result` either way.
`get_midi_notes` is the one tool that returns real content instead of
just a path (raw note data, not a summary) — it exists because nothing
else exposes ground-truth MIDI content to the calling model, but even
there the response is capped server-side (`MAX_MIDI_NOTES_PER_PAGE`,
`songforge_mcp_shared/constants.py`) regardless of what `max_results` a
caller requests, so a large transcription can't produce an unbounded
response.

## Cross-platform process launching

`acestep_client.py` starts ACE-Step's Gradio server as a detached
background process (it takes minutes to load its checkpoint and must
keep running independent of any single tool call). This uses Python's
native `subprocess` detachment flags —
`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows,
`start_new_session=True` on POSIX — rather than shelling out to
PowerShell's `Start-Process`. This is both a portability fix (PowerShell
isn't available on Linux/macOS) and a hardening fix: the previous
approach built a shell command line by interpolating paths into an
f-string, which is an unnecessary injection surface for something that
native `subprocess` handles directly with an argument list.

The same reasoning applies to `.venv` layout and the separator
executable name, which are OS-detected (`.venv/Scripts/python.exe` +
`audio-separator.exe` on Windows vs. `.venv/bin/python` +
`audio-separator` on POSIX) rather than hardcoded.

### Why every subprocess call passes `no_window_popen_kwargs()`

On Windows, redirecting a console-subsystem child process's stdout/stderr
to pipes (`subprocess.run(..., capture_output=True)`) does **not** stop it
from getting its own visible console window — Windows creates one for any
console-subsystem process whose parent doesn't already have one to
inherit, which is exactly this server's situation when launched by a GUI
app like Claude Desktop. Left unfixed, every `yt-dlp` and
`audio-separator` invocation would flash up a window a user has no reason
to expect, easily read as something suspicious running in the
background. `songforge_mcp_shared/constants.no_window_popen_kwargs()`
adds `creationflags=subprocess.CREATE_NO_WINDOW` on Windows (a no-op on
POSIX, where this isn't a thing) and is passed to every `subprocess.run`/
`Popen` call in this codebase. This is separate from, and in addition to,
the `DETACHED_PROCESS` flags already used for the ACE-Step server launch
itself — both are set together there for defense in depth. Any window
that still appears *during* generation rather than at ACE-Step server
startup would be coming from a process ACE-Step itself spawns internally
(e.g. a multiprocessing/vLLM worker), which is outside this server's code
and not something this project can fix from here.

## Why separation runs in its own environment

`songforge_mcp/separator_client.py` subprocesses into `audio-separator`
(BS-Roformer) rather than importing it, and does so from a dedicated
environment entirely separate from ACE-Step's. The two tools have
unrelated, independently-versioned dependencies; installing them into a
shared environment in earlier testing produced silent version conflicts
that were more expensive to diagnose than simply keeping them apart.

## Package layout

```
songforge_mcp/
├── main.py                  # FastMCP entry point
├── tool_registry.py         # Auto-discovers tools/ modules
├── acestep_client.py        # Playwright-driven ACE-Step integration
├── job_registry.py          # In-memory background job tracking for generation
├── separator_client.py      # Stem separation via audio-separator
├── youtube_reference.py     # Reference-audio download via yt-dlp
├── instructions/
│   └── 00_core.md           # Injected system-prompt instructions
└── tools/                   # generate_vocal_track, check_vocal_track_status,
                              # split_vocal_stems

songforge_mcp_shared/
├── constants.py              # Paths, timeouts, safety limits
├── error_codes.py            # SongForgeMCPError + ErrorCode
└── protocol.py                # Input validation, output helpers
```

## Design decisions carried over from v1

- **Composition stays out of this codebase.** This server renders what
  it's given; deciding what to write is the calling model's job, done in
  conversation with the user before any tool is called.
- **Typed errors.** `SongForgeMCPError` + `ErrorCode` give the calling
  model specific, machine-readable failure reasons rather than a generic
  message.
- **External tools stay external.** Neither ACE-Step nor audio-separator
  is a Python dependency of this project — both are separately-installed
  tools this server automates or subprocesses into, matching how this
  server previously integrated with DiffSinger.

See [`TOOLS.md`](TOOLS.md) for the tool reference and
[`INSTALLATION.md`](INSTALLATION.md) for setup.
