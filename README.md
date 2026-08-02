<!-- mcp-name: io.github.xDarkzx/songforge-mcp -->
<h1 align="center">SongForge-MCP</h1>

<p align="center">
  <strong>Generate original AI music through Claude, powered by ACE-Step 1.5</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License" /></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-compatible-purple.svg" alt="MCP Compatible" /></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.3.0-orange.svg" alt="v0.3.0" /></a>
  <a href="https://pypi.org/project/songforge-mcp/"><img src="https://img.shields.io/pypi/v/songforge-mcp" alt="PyPI" /></a>
  <a href="https://github.com/xDarkzx/SongForge-MCP/actions/workflows/ci.yml"><img src="https://github.com/xDarkzx/SongForge-MCP/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/ace-step/ACE-Step-1.5"><img src="https://img.shields.io/badge/GPU-12GB%2B%20VRAM-red.svg" alt="GPU 12GB+ VRAM" /></a>
  <a href="https://github.com/sponsors/xDarkzx"><img src="https://img.shields.io/badge/Sponsor-30363D?logo=githubsponsors&logoColor=EA4AAA" alt="Sponsor" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="docs/TOOLS.md">Tools Reference</a> &bull;
  <a href="docs/ARCHITECTURE.md">Architecture</a> &bull;
  <a href="CHANGELOG.md">Changelog</a> &bull;
  <a href="CONTRIBUTING.md">Contributing</a> &bull;
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

**Ask Claude for a song in plain English — "an upbeat pop track about
road trips, with lyrics about freedom" — and get back a real, finished
audio file.** No music software, no AI/ML knowledge, and no account or
subscription required. Everything runs on your own computer.

**If this is useful to you, a star helps other people find it** — that's the whole marketing budget for this project. Want to help keep it maintained? Click the **Sponsor** badge up top.

## What is this, exactly?

This is an **MCP server** — a small local program that gives Claude
Desktop (Anthropic's Claude app) a new ability it doesn't have out of
the box: generating actual music. "MCP" (Model Context Protocol) is
just the standard way Claude apps plug in extra tools; you don't need
to understand it to use this, only to install it once (see below).

Under the hood, this server drives [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) —
a genuinely capable, open-source AI music generation model, entirely
credited to its original authors; this project doesn't train or own
any model, it just automates using one well. ACE-Step's own interface
exposes around seventy interdependent generation settings, and getting
a good result out of it depends on a number of non-obvious defaults and
interactions between them. SongForge-MCP removes that layer entirely:
you describe the track you want in conversation, and Claude calls this
server to produce it, handling every underlying setting correctly on
your behalf. You never touch a settings panel.

## Runs entirely on your own machine

There is no cloud service behind this, no subscription, and no account
to create. Generation happens on your own GPU, using models downloaded
once to your own disk. Your prompts, lyrics, and generated audio are
never sent to any third-party server for processing, and nothing about
your usage is logged or transmitted anywhere. The one exception is
strictly opt-in: if you point a generation at a YouTube link for
reference-audio style matching, that specific request naturally reaches
YouTube to fetch the audio — nothing else in this server makes any
outbound network call. Everything else, including every generation
itself, stays entirely local.

### Works With

SongForge-MCP works with any AI client that supports the [Model Context Protocol](https://modelcontextprotocol.io):

- [Claude Desktop](https://claude.ai/download) — Anthropic's desktop app
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — CLI agent
- [Cursor](https://www.cursor.com/) — AI code editor with MCP support
- [LM Studio](https://lmstudio.ai/) — run local models with MCP tool access
- Any other [MCP-compatible client](https://modelcontextprotocol.io/clients)

---

## Quick Start

**Recommended hardware:** an NVIDIA GPU with **≥12GB VRAM** (≥20GB to run
without CPU offload), plus **~40GB free disk space** for the ACE-Step 1.5
XL-SFT model this server is configured to use. See
[System requirements](docs/INSTALLATION.md#system-requirements) for the
full breakdown — these numbers come from ACE-Step 1.5's own published
requirements, not a guess.

### 1. Get SongForge-MCP

**Option A:** Click the green **Code** button above → **Download ZIP** → extract to a folder.

**Option B:** Clone with git (lets you `git pull` for updates later — and
you'll need git installed anyway, since the installer itself uses it in
step 2 to fetch ACE-Step 1.5):

A fresh Windows install doesn't ship with git — check first:
```powershell
git --version
```
If that says "not recognized", install it, then close and reopen your terminal:
```powershell
winget install --id Git.Git -e --source winget
```
(`winget` itself ships with Windows 11 and up-to-date Windows 10. If `winget` isn't found either, grab the installer directly from [git-scm.com](https://git-scm.com/download/win) — click through it with the default options.)

macOS/Linux almost always have git already — `git --version` to check, or `brew install git` / `sudo apt install git` if not.

```bash
git clone https://github.com/xDarkzx/SongForge-MCP.git
cd SongForge-MCP
```

**Option C:** Already have Python? Install the published package straight from PyPI:
```bash
pip install songforge-mcp
```
This gives you the `songforge-mcp` command directly — but it only installs
this server itself, not ACE-Step 1.5 or the separator environment (the
PyPI package deliberately doesn't bundle either, since ACE-Step's ~28GB
checkpoint and its own dependency stack aren't something a Python package
install should silently pull in). You'll still need to run the installer
below (or follow [`docs/INSTALLATION.md`](docs/INSTALLATION.md) manually)
to set those up and point `SONGFORGE_ACESTEP_HOME` at a real checkout.

### 2. Run the installer

**Windows:** either double-click `install.bat` in File Explorer, or — if you're already in a terminal from the `git clone` step above — just keep going in the same window:
```powershell
.\install.bat
```

**macOS / Linux:**
```bash
bash install.sh
```

This provisions everything end to end: this server's own environment,
a full ACE-Step 1.5 checkout with the exact dependency versions it
needs, and an isolated environment for vocal/instrumental separation —
including Python itself on Windows if it isn't already on your system.
It downloads a genuinely large amount of data (the ACE-Step model is
tens of GB), so this step takes a while on a typical connection — good
time to make a coffee. See [`docs/INSTALLATION.md`](docs/INSTALLATION.md)
for exactly what each step does and how to recover if one fails.

Near the end, it asks: **"Configure Claude Desktop for SongForge-MCP? (y/n)"**
— say yes. You don't need to touch any configuration files yourself.

<details>
<summary>Manual install / other MCP clients (Cursor, Claude Code, etc.)</summary>

Install manually with `pip install -e ".[dev]"` from the repo folder,
then add this to your client's MCP config, pointing at the executable
inside this project's own `.venv`:

```json
{
  "mcpServers": {
    "songforge": {
      "command": "/path/to/SongForge-MCP/.venv/Scripts/songforge-mcp.exe"
    }
  }
}
```

(On Linux/macOS, point at `.venv/bin/songforge-mcp` instead.) Check your
client's MCP documentation for its config file location. See
[`docs/INSTALLATION.md`](docs/INSTALLATION.md) for the full manual setup,
including the environment variables this server needs.

</details>

### 3. Restart Claude Desktop

Fully quit and reopen it so it picks up the new server.

### 4. Ask for a track

```
"Generate me an upbeat pop track about summer road trips."
"Write me a melodic dubstep song, dreamy female vocals, about holding on to a memory."
"Split that last track into vocal and instrumental stems."
```

The first generation takes longer than usual — ACE-Step's ~28GB
checkpoint downloads and loads into memory on first use. After that,
it's much faster. See [`docs/TOOLS.md`](docs/TOOLS.md) for the full
tool reference and more example requests.

---

## Features

| Tool | What it does |
|------|---------------|
| `generate_vocal_track` | Produces a complete original track (vocals + instrumentation together) from a style description and full lyrics, in whatever genre/mood you ask for — not EDM-only, ACE-Step genuinely supports pop, rock, trap, R&B, folk, and many regional styles |
| `check_vocal_track_status` | Polls a generation/split/analysis/transcription job through to completion — every long-running tool here returns a `job_id` immediately rather than blocking |
| `split_vocal_stems` | Separates a generated track into an isolated vocal stem and an isolated instrumental stem, ready to drop into your own production |
| `analyze_reference_audio` | Measures real BPM/key/duration from an audio file — used to verify a generation actually landed on the intended tempo/key before calling it done |
| `transcribe_instrumental_to_midi` | Converts an instrumental stem to MIDI, heuristically split into bass/melody/chords parts, for a DAW starting point |
| `get_midi_notes` | Returns real pitch/timing/velocity note data from a transcribed MIDI file — paginated for large transcriptions |
| `play_audio` | Plays a previously generated file on demand (a completed generation already auto-plays; this is the fallback/replay option) |
| `prepare_voice_reference` | Builds a named library of reference vocal clips (from a local file or YouTube) for consistent style-matching across generations |
| `generate_vocal_track_takes` | Generates 2-5 independent takes of the same song (different seed each time) and measures each one's BPM/key so you can actually compare and pick, instead of accepting whatever the first attempt produced |
| `list_generated_tracks` | Lists finished tracks already sitting in the output folder, newest first — for when a `job_id` has been lost |
| `list_recent_jobs` | Lists recent background jobs across every tool here, newest first — same reason as above |
| `delete_generated_track` | Moves a file to a `.trash` subfolder — reversible by design, not a permanent delete |
| `edit_audio_track` | Trims and/or fades a previously generated file, optionally converting format, writing a new file without touching the original |

**Reference-audio style matching** — point a generation at a specific
vocal sample or a YouTube link, and the result adopts that voice's
timbre and performance character. It does not copy the reference's
melody, rhythm, or lyrics — those still come entirely from what you ask
Claude to write.

**Guardrails, not guesswork.** Every setting this server doesn't
explicitly need is left at ACE-Step's own proven-correct default, rather
than reconstructed by guesswork — the approach that produced reliable
results after every other approach silently produced noise. An
`advanced_settings` override is available for the cases that do need a
specific ACE-Step setting changed.

> See **[`docs/TOOLS.md`](docs/TOOLS.md)** for full signatures, arguments, and example requests for every tool.

---

## Works well alongside Reaper-MCP

If your production workflow is built around REAPER, the vocal and
instrumental stems this server produces are well suited to import
directly into a REAPER project using [Reaper-MCP](https://github.com/xDarkzx/Reaper-MCP),
a companion MCP server for AI-assisted composition, mixing, and
mastering in REAPER. Both servers can be connected to the same Claude
Desktop session, allowing a track to be generated here and then placed,
arranged, and produced further without leaving the conversation.

---

## Architecture

```
┌──────────────┐     stdio      ┌──────────────┐   browser automation   ┌──────────────┐
│  MCP Client  │◄──────────────►│ SongForge-MCP│◄───────────────────────►│  ACE-Step    │
│(AI assistant)│    (JSON-RPC)  │   FastMCP    │  (Playwright, Gradio)   │   1.5 (UI)   │
└──────────────┘                └──────────────┘                        └──────────────┘
```

This server drives ACE-Step's own Gradio UI via Playwright browser
automation rather than its raw internal API — three separate attempts
at reconstructing correct values for that API's ~70 mostly-unlabeled
parameters all produced static/noise, while driving the actual UI (fill
the fields, click Generate, poll for completion) worked reliably. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full reasoning,
the job/poll pattern used for every long-running tool, and cross-process
locking to prevent two generations from racing for the same GPU.

### Project structure

```
SongForge-MCP/
├── songforge_mcp/
│   ├── main.py                     # FastMCP server entry (`songforge-mcp` command)
│   ├── tool_registry.py            # Auto-discovers tool modules
│   ├── acestep_client.py           # Playwright-driven ACE-Step UI automation
│   ├── separator_client.py         # Vocal/instrumental separation (audio-separator)
│   ├── media_player.py             # Auto-plays a finished generation
│   ├── midi_transcription.py       # Instrumental → MIDI transcription
│   ├── voice_reference_library.py  # Named voice-reference clip library
│   ├── youtube_reference.py        # Reference-audio download from YouTube
│   ├── audio_analysis.py           # BPM/key/duration measurement
│   ├── job_registry.py             # Shared job/poll registry for long-running tools
│   ├── shared_state.py             # Single shared client/registry instances
│   ├── instructions/
│   │   └── 00_core.md              # System-prompt instructions injected into MCP
│   └── tools/                      # One module per tool group, auto-registered
│       ├── generate_tools.py       # generate_vocal_track(_takes), check_vocal_track_status
│       ├── separate_tools.py       # split_vocal_stems
│       ├── analysis_tools.py       # analyze_reference_audio
│       ├── midi_tools.py           # transcribe_instrumental_to_midi, get_midi_notes
│       ├── playback_tools.py       # play_audio
│       ├── voice_reference_tools.py # prepare_voice_reference
│       ├── track_management_tools.py # list_generated_tracks, list_recent_jobs, delete_generated_track
│       └── audio_edit_tools.py     # edit_audio_track
├── songforge_mcp_shared/
│   ├── constants.py                # Paths, timeouts, safety limits
│   ├── error_codes.py              # SongForgeMCPError + ErrorCode enum
│   └── protocol.py                 # Input validation, path safety checks
├── docs/
│   ├── INSTALLATION.md             # Detailed setup for all platforms
│   ├── TOOLS.md                    # Complete tool reference
│   ├── ARCHITECTURE.md             # UI-automation reasoning, job/poll pattern, locking
│   └── TECH_STACK_RESEARCH.md      # Why ACE-Step 1.5 was chosen
├── tests/
├── install.bat / install.sh        # One-click installers
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `"ACESTEP_NOT_CONFIGURED"` or `"SEPARATOR_NOT_CONFIGURED"` errors | The relevant environment variable isn't set or doesn't point to a valid install. Re-run `install.bat`/`install.sh`, or check [`docs/INSTALLATION.md`](docs/INSTALLATION.md) and set it manually, then restart Claude Desktop. |
| A generation silently never completes | Check `<ACE-Step folder>/mcp_server_stderr.txt` for what ACE-Step's own process is doing — the same log a human would check running it manually. |
| GPU/CUDA errors on generation | Confirm your GPU driver actually supports the pinned PyTorch/CUDA combination the installer used — a hardware-compatibility question outside this server's control. |
| Claude Desktop doesn't see SongForge-MCP | Restart Claude Desktop after installing/editing config. On Windows, note that a **Microsoft Store** install of Claude Desktop uses a different, sandboxed config path than the usual `%APPDATA%\Claude` — `install.bat` detects this automatically, but if editing by hand, check `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude_desktop_config.json` too. |
| A completed track doesn't visibly play | It auto-plays in your OS's default media player — on Windows, a window opened by a background process can flash in the taskbar instead of popping up, a Windows restriction, not a bug. Ask Claude to `play_audio` the file again, or check the taskbar. |
| Generation seems unusually slow / times out | ACE-Step has its own internal generation timeout, separate from this server's. Very long lyrics or a long requested duration can push past it — ask for a shorter song (~3-4 minutes is the sweet spot) or fewer candidate takes. |
| `"command not found: songforge-mcp"` | Run the installer again, or manually: `pip install -e .` from the repo folder, or `pip install songforge-mcp`. |

---

## Development

```bash
git clone https://github.com/xDarkzx/SongForge-MCP.git
cd SongForge-MCP
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # .venv/bin/pip on Linux/macOS

pytest -v
```

The test suite is fully mocked/stubbed — it doesn't require a GPU or a
running ACE-Step instance. See [CONTRIBUTING.md](CONTRIBUTING.md) for
how to add a new tool and full contribution guidelines.

---

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** — install and configure, step by step, for every platform
- **[Tools Reference](docs/TOOLS.md)** — what each tool does, its arguments, and example requests
- **[Architecture](docs/ARCHITECTURE.md)** — how this server actually drives ACE-Step under the hood
- **[Tech Stack Research](docs/TECH_STACK_RESEARCH.md)** — why ACE-Step 1.5 was chosen over the alternatives
- **[Contributing](CONTRIBUTING.md)** — how to add tools and contribute
- **[Changelog](CHANGELOG.md)** — version history and release notes

## Status

Implemented and verified end-to-end against a live ACE-Step 1.5 instance.
Vocal/instrumental separation is functional but not perfect — some bleed
between stems is a known limitation of the current separation model, not
a bug in this server.

## Support

If SongForge-MCP has helped with your music generation, consider sponsoring:

<p align="center">
  <a href="https://github.com/sponsors/xDarkzx">
    <img src="https://img.shields.io/badge/Sponsor-30363D?style=for-the-badge&logo=githubsponsors&logoColor=EA4AAA" alt="GitHub Sponsors" />
  </a>
</p>

Your support helps keep this project maintained and free for everyone.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
