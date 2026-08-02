"""Shared constants: ACE-Step checkout paths, timeouts, and safety limits."""
import os
import platform
import subprocess
from pathlib import Path

# This project's own root (the folder containing pyproject.toml), not
# wherever the process happens to be invoked from. Used as the default
# home for generated output — see _default_output_root() below.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_output_root() -> str:
    """Where generated tracks/stems live if SONGFORGE_OUTPUT_DIR isn't
    set. Deliberately a folder inside this repo checkout, not the OS temp
    directory: temp is fine for scratch state, but a system's cleanup
    tools (Windows Storage Sense, etc.) can and do purge it, backup
    software typically skips it, and it's not somewhere a user would
    think to look for a track they just asked for and want to keep — all
    of which were real problems with the previous tempfile-based default.
    Also deliberately NOT a path outside this repo (e.g. a sibling
    project folder): this repo is meant to be cloned standalone by people
    who won't have this developer's particular multi-project directory
    layout, so the default must be self-contained."""
    return str(_PROJECT_ROOT / "output")


def _resolve_output_root() -> str:
    return os.environ.get("SONGFORGE_OUTPUT_DIR", _default_output_root())


def ensure_private_dir(path: str) -> None:
    """Create a directory if missing, and best-effort restrict it to the
    owning user only (0700). Mirrors reaper_mcp_shared.constants."""
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def no_window_popen_kwargs() -> dict:
    """Extra kwargs for subprocess.run/Popen that stop a console window
    from flashing up on Windows. `capture_output`/pipe redirection alone
    does NOT prevent this — a console-subsystem child process (python.exe,
    yt-dlp, audio-separator.exe) still gets a real, visible console window
    of its own on Windows whenever its parent has none to inherit (which
    this server's parent process typically doesn't, being launched by a
    GUI app), unless CREATE_NO_WINDOW is set explicitly. No-op on POSIX,
    where this isn't a thing."""
    if platform.system() == "Windows":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def open_with_default_app(path: str) -> None:
    """Opens a file with the OS's default associated application - used
    to auto-play a completed generation so the user doesn't have to find
    it in File Explorer themselves. Best-effort by design: callers must
    not let a failure here break an otherwise-successful tool response,
    since the file path is always returned regardless and remains the
    real result - this is a convenience on top of that, not a
    replacement for it. `os.startfile` is Windows-only in the standard
    library and resolves the OS's actual default app the same way
    double-clicking the file would; `open`/`xdg-open` cover macOS/Linux
    the same way."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class Paths:
    # Root of a separately-cloned ace-step/ACE-Step-1.5 checkout. Must be
    # set by the user at install time (see docs/INSTALLATION.md) — ACE-Step
    # is not pip-installable, and its checkpoints (~28GB) are managed by its
    # own app, not this server.
    ACESTEP_HOME = os.environ.get("SONGFORGE_ACESTEP_HOME", "")
    # Override with SONGFORGE_OUTPUT_DIR to redirect generated output
    # anywhere else (e.g. a personal archive folder) — see
    # _default_output_root() for why the built-in default lives inside
    # this repo rather than the OS temp directory.
    OUTPUT_ROOT = _resolve_output_root()
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "renders")
    REFERENCE_AUDIO_CACHE = os.path.join(OUTPUT_ROOT, "reference_audio")
    # A library of extracted vocal-only clips, organized one subfolder per
    # named voice (e.g. reference_voices/annika-wells/) - distinct from
    # REFERENCE_AUDIO_CACHE, which is keyed by YouTube video ID for
    # single-generation reference audio, not accumulated per-voice over
    # multiple clips.
    REFERENCE_VOICES_DIR = os.path.join(OUTPUT_ROOT, "reference_voices")


class Separator:
    # The model previously in use (model_bs_roformer_ep_317_sdr_12.9755.ckpt,
    # audio-separator's own default) left vocals bleeding into the
    # instrumental stem on dense/loud mixes - heavy synths/distortion
    # masking vocal frequencies badly enough that some vocal passages were
    # misclassified as instrumental entirely, not just degraded. A sibling
    # BS-Roformer checkpoint (same architecture/training lineage, just a
    # different epoch) was considered but rejected as the default: if the
    # failure is architectural, a same-family checkpoint likely inherits
    # the same weakness rather than actually testing a different approach.
    # vocals_mel_band_roformer.ckpt: the highest vocal SDR (12.6) of every
    # model in audio-separator's full catalog (checked directly via
    # `audio-separator --list_models`, not assumed) - beats every
    # BS-Roformer checkpoint (best: 12.1) and every MDX-Net option
    # (best: 10.4), while also being a different architecture from the
    # BS-Roformer model that was actually failing on dense mixes, so this
    # isn't just a marginally-different same-family checkpoint. Still
    # unproven on this project's actual dense mixes until tested for real.
    DEFAULT_MODEL = os.environ.get(
        "SONGFORGE_SEPARATOR_MODEL", "vocals_mel_band_roformer.ckpt"
    )
    # Best-scoring BS-Roformer checkpoint (12.1 vocal SDR, vs. the old
    # default's 11.8) - kept as a same-family fallback to A/B against, in
    # case the architecture change above turns out worse in practice.
    ALT_MODEL = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"


class GradioServer:
    HOST = "127.0.0.1"
    PORT = int(os.environ.get("SONGFORGE_ACESTEP_PORT", "7866"))
    URL = f"http://{HOST}:{PORT}"
    # Which ACE-Step checkpoint to launch with - "acestep-v15-xl-sft" is
    # this project's tested default (higher quality, what its own docs
    # are written around). Deliberately a manual, restart-required env
    # var rather than something this server can switch at runtime: a
    # calling model changing this on its own would mean killing whatever
    # generation is in progress and a multi-GB download with no user
    # confirmation in between - a real, explicit decision the user makes
    # themselves, not something automated. See docs/INSTALLATION.md for
    # the other checkpoints ACE-Step supports and their tradeoffs.
    CHECKPOINT = os.environ.get("SONGFORGE_ACESTEP_CHECKPOINT", "acestep-v15-xl-sft")


class Timeouts:
    # ACE-Step generation time varies a lot, and not just with luck: a
    # real "timed out" complaint traced back to a genuine, documented
    # hardware constraint, not a bug. ACE-Step's own GPU_COMPATIBILITY.md
    # places this project's 12-16GB VRAM tier as "marginal" for the XL
    # DiT model this server uses - it requires real CPU offload +
    # quantization on this tier, and that tier still permits requesting
    # up to 8-minute songs. Long lyrics push toward longer requested
    # durations and more LM/lyric-conditioning work, both of which add
    # real time on a tier that's already trading speed for VRAM fit via
    # offloading - this compounds, it doesn't just add a little. 900s
    # was measured against short/typical clips (~10-100s with no
    # reference audio, ~9 min observed with reference audio) and never
    # validated against a long-lyrics full-length song on this specific
    # tier - raised to give real headroom for that documented case.
    # Still a generous ceiling chosen by reasoning about known
    # constraints, not a precisely tuned/benchmarked value.
    SERVER_STARTUP = 300.0
    GENERATION = 1800.0
    YOUTUBE_DOWNLOAD = 120.0
    # The ~5-15s this was originally sized around was observed with the
    # OLD default separator model (model_bs_roformer_ep_317_sdr_12.9755.ckpt).
    # The new default (Separator.DEFAULT_MODEL, vocals_mel_band_roformer.ckpt,
    # swapped for its vocal-isolation quality - see that constant's comment)
    # is a different, apparently slower architecture: a real timeout was
    # observed at 194s on a longer file, already past the old 180s ceiling.
    # Raised with real headroom rather than a precisely re-measured value,
    # same reasoning as GENERATION above - kept apart from GENERATION's much
    # larger budget so a genuinely stuck separator still fails in a bounded
    # time, not GENERATION's 1800s, since separation is not diffusion-based
    # and should never legitimately need that long.
    SEPARATION = 600.0
    # edit_audio_track's mp3 path shells out to ffmpeg for the one format
    # soundfile/libsndfile can't write directly - a few minutes of audio
    # transcodes in well under this on any real hardware.
    FFMPEG_CONVERT = 120.0


MAX_CAPTION_LENGTH = 1000     # characters
MAX_LYRICS_LENGTH = 4000      # characters, generous for a full multi-section song
