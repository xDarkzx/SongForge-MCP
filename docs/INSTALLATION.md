# Installation

## System requirements

This server automates ACE-Step 1.5 — its hardware needs are what actually
matter here, not this server's own footprint (which is trivial). Numbers
below are from [ACE-Step 1.5's own published requirements](https://github.com/ace-step/ACE-Step-1.5)
(`docs/en/INSTALL.md` and `docs/en/GPU_COMPATIBILITY.md` in that repo),
not this project's own estimate.

| | Requirement |
|---|---|
| OS | Windows, Linux, or macOS |
| Python | 3.11 or 3.12 for the ACE-Step checkout specifically (this server's own venv is more permissive, but ACE-Step itself requires 3.11-3.12) |
| GPU | An NVIDIA CUDA GPU is what this project's installers target and what has actually been tested. ACE-Step 1.5 also supports AMD ROCm, Intel XPU, Apple Silicon (MPS), and CPU-only — but `install.bat`/`install.sh` don't set those up automatically; see ACE-Step's own `docs/en/INSTALL.md` if you need one of them |
| VRAM | Both installers now detect available VRAM automatically (`scripts/detect_gpu_checkpoint.py`, run right after ACE-Step's own venv/torch install) and pick the right-sized checkpoint for it, setting `SONGFORGE_ACESTEP_CHECKPOINT` accordingly — **below 8GB**: `acestep-v15-turbo` (2B), **8-20GB**: `acestep-v15-sft` (2B), **20GB+**: `acestep-v15-xl-sft` (4B, this project's original hardcoded default). This never overrides a `SONGFORGE_ACESTEP_CHECKPOINT` you've already set yourself, and any detection failure (no GPU, non-NVIDIA, any error) falls back to the XL-SFT default unchanged — see `docs/2026-08-03-gpu-checkpoint-detection-design.md` for the full reasoning behind each threshold |
| Disk | ~40GB free is checked by the installer before it proceeds — the XL-SFT checkpoint alone is ~28GB and downloads on first generation, on top of PyTorch/CUDA and this project's other dependencies |
| No GPU at all | Both installers detect the absence of an NVIDIA GPU and fall back to CPU builds so the install itself won't fail, but this has not been performance-tested by this project — expect generation to be dramatically slower, likely impractically so for the XL-SFT configuration these installers set up |
| `ffmpeg` on PATH | Needed for `reference_youtube_url`/`remix_source_youtube_url` (yt-dlp's audio extraction) and for `edit_audio_track`'s mp3 output (soundfile can only write wav/flac directly). Not installed by either installer — get it from [ffmpeg.org](https://ffmpeg.org/download.html) or your package manager (`winget install ffmpeg`, `apt install ffmpeg`, `brew install ffmpeg`) if those features error out. |

Detection requires an NVIDIA GPU (same as the rest of this installer) —
on AMD/Intel/Apple Silicon/no-GPU hardware, detection fails silently and
you fall back to the XL-SFT default regardless of actual VRAM. Running
ACE-Step 1.5 directly (outside this server) with its own GPU-tier
auto-configuration remains the better starting point for that hardware.

## Installing

Two installer scripts provision everything this server needs in one pass:
this server's own environment, a full ACE-Step 1.5 checkout with the
exact dependency versions it requires, and an isolated environment for
vocal/instrumental separation.

- **Windows:** `install.bat`
- **Linux / macOS:** `install.sh` (`bash install.sh`)

Run the one matching your OS once from this folder. What follows explains
what each step does and how to recover if a step fails partway through —
both scripts are safe to re-run; completed steps are skipped or cleanly
repeated.

The two scripts differ in two places, both called out below: Python
bootstrap (Windows installs Python for you; Linux/macOS just tells you
the right command) and environment-variable persistence (Windows sets
them permanently via `setx`; Linux/macOS prints `export` lines for you to
add to your shell profile, since there's no cross-shell equivalent of
`setx`).

## What the installer does

### 1. Python

**Windows:** if `python` isn't found on your system `PATH`, the installer
downloads and silently installs Python 3.11 for the current user and adds
it to `PATH`. If this step runs, **restart your terminal and re-run
`install.bat`** so the updated `PATH` takes effect before the rest of the
install continues.

**Linux/macOS:** `install.sh` checks for `python3` (3.10+) and, if
missing, prints the right install command for your distro
(`apt-get`/`dnf`/`pacman`) or `brew` on macOS, then exits — installing
Python unattended isn't safe to automate across that many package
managers. Install it and re-run the script.

### 2. Disk space check

ACE-Step 1.5's model checkpoint is approximately 28 GB and downloads on
its first run. The installer checks for at least 40 GB free before
proceeding and stops with a clear message if there isn't enough — running
this download on a nearly-full drive has caused real problems in testing.
If it stops here, free up space (clearing package manager caches and
stale virtual-machine disk images are usually the largest wins) and
re-run the installer.

### 3. This server's own environment

Creates `.venv` in this folder and installs `songforge-mcp` along with
its dependencies (`mcp`, `playwright`, `yt-dlp`), then downloads the
Chromium browser Playwright needs.

`songforge-mcp` is also published on [PyPI](https://pypi.org/project/songforge-mcp/) —
`pip install songforge-mcp` is an option if you'd rather not clone this
repo just for the server package itself (e.g. managing it inside your
own existing Python environment). This doesn't replace the rest of this
installer, though: ACE-Step 1.5 and the separator environment (steps 4-5
below) are separate checkouts this package doesn't bundle, and still
need to be set up either way — the installer scripts remain the
recommended path for a first install.

### 4. ACE-Step 1.5

If `SONGFORGE_ACESTEP_HOME` isn't already set, the installer clones
[`ace-step/ACE-Step-1.5`](https://github.com/ace-step/ACE-Step-1.5) into
a sibling folder and sets up its own virtual environment inside that
checkout, including:

Since ACE-Step needs Python 3.11-3.12 specifically (unlike this server's
own venv, which accepts any 3.10+), the installer no longer blindly reuses
whatever `python`/`python3` happens to resolve on `PATH` for this step —
that broke silently under pyenv-win, where the global/shim default can be
a different, incompatible version ([#1](https://github.com/xDarkzx/SongForge-MCP/issues/1)).
It now probes for a qualifying 3.11/3.12 interpreter first (the Windows
`py` launcher's `-3.12`/`-3.11`, or versioned `python3.12`/`python3.11`
binaries on Linux/macOS) and stops with a clear message telling you how to
install one if none is found.

- PyTorch, pinned to a specific version matched to a known-good CUDA
  build. **This pin matters** — a newer PyTorch version breaks the
  prebuilt accelerated-attention component ACE-Step depends on, which in
  turn breaks its language-model backend in a way that's harder to
  diagnose than simply running slower. Don't "helpfully" upgrade it.
- ACE-Step's own requirements, including its language-model component.

On Linux, the installer detects an NVIDIA GPU (via `nvidia-smi`) and
installs the same pinned CUDA build as Windows. Without a detected NVIDIA
GPU (including all of macOS, which has no CUDA support), it falls back to
a plain CPU PyTorch build instead. That CPU fallback has not been
performance-tested by this project — it's the correct package per
PyTorch's own published wheel support, but expect generation to be much
slower than on a CUDA GPU.

This is the slowest part of the install — expect it to take a while on
a typical connection. The 28 GB checkpoint itself is *not* downloaded
here; it downloads automatically the first time a track is generated.

If you already have an ACE-Step 1.5 checkout you'd like to reuse, set
`SONGFORGE_ACESTEP_HOME` to point at it before running the installer
and this step is skipped.

### 5. Vocal/instrumental separator

Creates a separate, isolated environment (`.separator_env`) for
`audio-separator`, the tool used by `split_vocal_stems`. This is kept
independent from ACE-Step's own environment deliberately — the two have
unrelated dependency requirements, and past experience in this project
showed that combining environments with different needs silently breaks
one or the other.

### 6. Environment variables

The installer needs three environment variables set so this server can
find everything it just installed:

| Variable | Points to |
|---|---|
| `SONGFORGE_ACESTEP_HOME` | The ACE-Step 1.5 checkout |
| `SONGFORGE_SEPARATOR_PYTHON` | The separator environment's Python interpreter |
| `SONGFORGE_YTDLP_PYTHON` | This server's own Python interpreter (yt-dlp is installed alongside it) |
| `SONGFORGE_OMNIZART_PYTHON` *(optional)* | The Omnizart environment's Python interpreter — only set if you opted into installing it (interactive prompt in both installers, or run `scripts/install_omnizart.py` directly later). Enables `transcribe_instrumental_to_midi_omnizart`, a second, more accurate (on real material tested so far) transcription engine alongside the default basic-pitch one. Genuinely heavier and more fragile to install than everything else here — several compatibility patches are applied automatically for an old dependency (madmom) that predates current Python/numpy; see the install script's own docstring for exactly what and why. |

Two more, optional, control output location and which ACE-Step
checkpoint gets used:

| Variable | Points to |
|---|---|
| `SONGFORGE_OUTPUT_DIR` *(optional)* | Where generated tracks and stems are saved. Defaults to an `output/` folder inside this repo checkout if unset — not the OS temp directory, since temp can be purged by system cleanup tools and isn't somewhere you'd think to look for a track you asked for and want to keep. Set this if you'd rather output land somewhere else (e.g. a personal music archive folder). |
| `SONGFORGE_ACESTEP_CHECKPOINT` *(optional)* | Which ACE-Step model to load. Defaults to `acestep-v15-xl-sft` (this project's tested configuration — higher quality, needs more VRAM, see the requirements table above). Other real options include `acestep-v15-turbo` (faster, lower quality) and `acestep-v15-base` (slower, but the only one supporting ACE-Step's Extract/Lego/Complete modes — see [ACE-Step's own docs](https://github.com/ace-step/ACE-Step-1.5) for what those do). Changing this is a manual, deliberate choice — this server never switches or downloads a different checkpoint on its own, since that means killing whatever generation is running and a multi-GB download with no confirmation step in between. Set it, then restart this server (and Claude Desktop) for it to take effect — a not-yet-downloaded checkpoint downloads automatically on first use after that, the same way the default one does. |

**Windows:** `install.bat` sets these as permanent user environment
variables via `setx` automatically.

**Linux/macOS:** `install.sh` prints the three `export` lines at the end
of the run — add them to your shell profile (`~/.bashrc`, `~/.zshrc`, or
equivalent) yourself. The script doesn't edit that file for you.

**Restart your terminal (and Claude Desktop, if it's running) after
installation** so these take effect.

## Configuring Claude Desktop

Both installers now offer to do this step for you automatically (they'll
ask "Configure Claude Desktop for SongForge-MCP? (y/n)" near the end of
the run) — say yes and skip to [First run](#first-run) below. If you said
no, are using a different MCP client, or just want to do it by hand:
add this server to Claude Desktop's MCP configuration, pointing at the
executable created inside this project's own `.venv`.

**Windows only, if you installed Claude Desktop from the Microsoft
Store:** its config file is not at the usual `%APPDATA%\Claude` path —
Store apps run sandboxed and redirect their own AppData to a virtualized
per-package folder. The real file is under
`%LOCALAPPDATA%\Packages\Claude_<hash>\LocalCache\Roaming\Claude\claude_desktop_config.json`
(`install.bat` detects and handles this automatically; only relevant if
editing by hand). Editing the normal `%APPDATA%\Claude` path on a
Store install produces a valid file that Claude Desktop simply never
reads, with no error to indicate anything went wrong — confirmed the
hard way.

```json
{
  "mcpServers": {
    "songforge-mcp": {
      "command": "C:\\path\\to\\SongForge-MCP\\.venv\\Scripts\\songforge-mcp.exe"
    }
  }
}
```

On Linux/macOS, point at the `.venv/bin/songforge-mcp` script instead:

```json
{
  "mcpServers": {
    "songforge-mcp": {
      "command": "/path/to/SongForge-MCP/.venv/bin/songforge-mcp"
    }
  }
}
```

Restart Claude Desktop after saving. See [`docs/TOOLS.md`](TOOLS.md) for
how to use it once connected.

## First run

The first time you ask for a track, expect a longer wait than usual —
ACE-Step's ~28 GB checkpoint downloads and its model loads into memory on
first use. Subsequent generations are much faster since the model server
stays running in the background between requests.

## Troubleshooting

- **"ACESTEP_NOT_CONFIGURED" or "SEPARATOR_NOT_CONFIGURED" errors** — the
  relevant environment variable isn't set or doesn't point to a valid
  install. Re-run `install.bat`, or check the table above and set the
  variable manually, then restart Claude Desktop.
- **A generation silently never completes** — check
  `<ACE-Step folder>/mcp_server_stderr.txt` for what ACE-Step's own
  process is doing; this is the same log a human would check if running
  it manually.
- **GPU/CUDA errors on generation** — confirm your GPU driver actually
  supports the pinned PyTorch/CUDA combination from step 4. This is a
  hardware-compatibility question outside this server's control.
