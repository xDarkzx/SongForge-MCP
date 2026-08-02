"""Playwright-driven client for ACE-Step 1.5's Gradio UI.

ACE-Step 1.5's raw gradio_client API (`/generation_wrapper`, 72 mostly-
unlabeled positional parameters) proved too fragile to drive reliably —
three separate attempts at reconstructing correct default values from
source code all produced static/noise output, even after fixing real
environment bugs along the way (disk full, wrong torch version). Driving
the actual Gradio UI headlessly instead — filling only Music Caption and
Lyrics, clicking Generate, and leaving every other setting at the UI's
own real defaults — produced correct output on the first try. This
client encapsulates that approach, plus reliability fixes found the hard
way during testing:

1. Closing the browser too soon after clicking Generate can abort the
   request before the server ever registers it — several generations
   silently never ran at all until a wait was added after the click.
2. Reference-audio generations fall back to a much slower backend
   (~10s/step vs ~1.3-2.5s/step) — the completion timeout must be
   generous, not tuned to the no-reference-audio case.
3. FastMCP tools run inside asyncio — Playwright's *sync* API raises
   ("using Playwright Sync API inside the asyncio loop") if called from
   an async tool function, so this client uses the async API throughout.
4. ACE-Step's own generation is a single-queue, GPU-bound operation — a
   process-wide lock serializes calls so a second request can't launch a
   competing browser session and drive concurrent GPU work (or race to
   launch a second server process) while one is already in flight.
"""
import asyncio
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Awaitable, Callable

import psutil
import soundfile as sf
from playwright.async_api import async_playwright

from songforge_mcp_shared.constants import GradioServer, Paths, Timeouts, ensure_private_dir
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# ACE-Step's own reference-audio processing (confirmed in its
# docs/en/Tutorial.md and acestep/core/generation/handler/io_audio.py's
# process_reference_audio): it randomly samples three 10-second chunks
# from the front/middle/back thirds of whatever file it's given and
# concatenates them into the actual 30-second conditioning signal. For a
# multi-minute song that isn't in one key throughout (i.e. most real
# songs — verse/chorus key contrast is completely normal songwriting),
# those three chunks can land in genuinely different sections, stitching
# together an internally contradictory conditioning signal before the
# model ever generates a single frame. Confirmed via a controlled A/B
# test in this project (identical caption/lyrics, with vs without a
# multi-minute reference) that this can badly break vocal tuning — and
# via measuring the reference clip's own chroma/key profile that it does
# shift across its length. Trimming to one short, contiguous window
# before ACE-Step ever sees the file forces its front/middle/back
# sampling to draw only from that same narrow span.
_REFERENCE_TRIM_WINDOW_SECONDS = 32.0


def _trim_reference_for_key_coherence(path: str, window_seconds: float = _REFERENCE_TRIM_WINDOW_SECONDS) -> str:
    """Returns a path to a short, single-window clip suitable for ACE-Step's
    reference-audio input. If `path` is already short enough that
    front/middle/back sampling can't span more than one section, returns
    `path` unchanged. Otherwise trims a `window_seconds` window from the
    middle of the file (avoiding likely intro/outro fades at the very
    start/end) and writes it to a new file in Paths.REFERENCE_AUDIO_CACHE."""
    info = sf.info(path)
    duration = info.frames / info.samplerate
    if duration <= window_seconds:
        return path

    start_seconds = max(0.0, (duration - window_seconds) / 2)
    start_frame = int(start_seconds * info.samplerate)
    stop_frame = start_frame + int(window_seconds * info.samplerate)
    data, samplerate = sf.read(path, start=start_frame, stop=stop_frame)

    ensure_private_dir(Paths.REFERENCE_AUDIO_CACHE)
    trimmed_path = os.path.join(
        Paths.REFERENCE_AUDIO_CACHE, f"trimmed_{os.path.splitext(os.path.basename(path))[0]}.wav"
    )
    sf.write(trimmed_path, data, samplerate)
    return trimmed_path


def _slugify_title(title: str, max_length: int = 60) -> str:
    """Filesystem-safe slug for a song title (e.g. "Hearts on a Wire!" ->
    "hearts-on-a-wire") - lowercase, alphanumerics and hyphens only,
    truncated to max_length. Empty/all-punctuation input yields ""."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return slug[:max_length].strip("-")


def _venv_python_path(acestep_home: str) -> str:
    if _IS_WINDOWS:
        return os.path.join(acestep_home, ".venv", "Scripts", "python.exe")
    return os.path.join(acestep_home, ".venv", "bin", "python")

ProgressCallback = Callable[[float, str], Awaitable[None]]

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ETA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*s\b")

# ACE-Step's Audio Format control lives in the Settings panel's "Audio
# Output & Post-processing" section (acestep/ui/gradio/interfaces/
# generation_advanced_output_controls.py + user_preferences.py in the
# ACE-Step checkout) - it's a global, localStorage-persisted preference,
# not a per-generation form field, which is why it needs its own Settings-
# panel dance rather than going through _set_field_by_label like a normal
# advanced_settings override. Confirmed end-to-end against a live server:
# setting this and generating produced a real file with the matching
# extension on disk, not just a UI value change.
OUTPUT_FORMATS = frozenset({"flac", "mp3", "opus", "aac", "wav", "wav32"})
_OUTPUT_FORMAT_OPTION_LABELS = {
    "flac": "FLAC",
    "mp3": "MP3",
    "opus": "Opus",
    "aac": "AAC",
    "wav": "WAV (16-bit)",
    "wav32": "WAV (32-bit Float)",
}
# wav32 still saves with a plain .wav extension (confirmed in ACE-Step's
# own acestep/audio_utils.py: `ext = ".wav" if format == "wav32" else
# f".{format}"`) - not a real, separate on-disk extension.
_OUTPUT_FORMAT_EXTENSIONS = {
    "flac": "flac",
    "mp3": "mp3",
    "opus": "opus",
    "aac": "aac",
    "wav": "wav",
    "wav32": "wav",
}


class ACEStepClient:
    def __init__(self, acestep_home: str | None = None):
        self.acestep_home = acestep_home if acestep_home is not None else Paths.ACESTEP_HOME
        ensure_private_dir(Paths.OUTPUT_DIR)
        self._lock = asyncio.Lock()

    def _require_configured(self) -> None:
        if not self.acestep_home or not os.path.isdir(self.acestep_home):
            raise SongForgeMCPError(
                ErrorCode.ACESTEP_NOT_CONFIGURED,
                "SONGFORGE_ACESTEP_HOME is not set or does not point to a valid "
                "directory. See docs/INSTALLATION.md.",
            )

    def _is_server_up(self) -> bool:
        try:
            with socket.create_connection((GradioServer.HOST, GradioServer.PORT), timeout=2):
                return True
        except OSError:
            return False

    @staticmethod
    def _try_acquire_lock_file(lock_path: str, stale_after_seconds: float) -> bool:
        """Cross-process advisory lock via atomic exclusive file creation —
        the first process to successfully create the file "wins". The
        winner's PID is written into the file so a later, blocked process
        can check whether the holder is actually still alive (via
        `psutil.pid_exists`) and reclaim immediately if not — this is the
        common case in practice: force-closing Claude Desktop mid-
        generation kills the holding process without ever running its
        `finally` cleanup, and there is no reason to make the next attempt
        wait out the full staleness window when the holder is provably
        gone. Falls back to age-based staleness (`stale_after_seconds`)
        only when the PID can't be read or belongs to a live-but-unrelated
        process (e.g. PID reuse) — a real crash still can't wedge this
        lock forever. Shared implementation behind `_acquire_launch_lock`
        (server startup) and `_acquire_generate_lock` (an actual
        generation run) — two different critical sections with two
        different staleness windows, same underlying mutex mechanism."""
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            holder_dead = False
            try:
                with open(lock_path, "r") as f:
                    holder_pid = int(f.read().strip())
                holder_dead = not psutil.pid_exists(holder_pid)
            except (OSError, ValueError):
                pass

            if not holder_dead:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                except OSError:
                    return False
                if age <= stale_after_seconds:
                    return False

            try:
                os.remove(lock_path)
            except OSError:
                pass
            return ACEStepClient._try_acquire_lock_file(lock_path, stale_after_seconds)

    @staticmethod
    def _acquire_launch_lock(lock_path: str) -> bool:
        """Cross-process advisory lock so two separate OS processes never
        both conclude "ACE-Step isn't running yet" and each launch a
        competing server. This is a real, confirmed failure mode, not a
        theoretical one: this project's own MCP server (which warms up
        ACE-Step on its own startup) and a separately-run diagnostic
        script each independently launched acestep_v15_pipeline.py within
        the same few minutes, and both loaded the ~9GB XL-SFT model onto
        the same 12GB GPU — 95% VRAM utilization, and the next real
        generation timed out. `self._lock` (asyncio.Lock) only prevents
        this within one process; it does nothing across process
        boundaries."""
        return ACEStepClient._try_acquire_lock_file(lock_path, Timeouts.SERVER_STARTUP + 60)

    @staticmethod
    def _acquire_generate_lock(lock_path: str) -> bool:
        """Cross-process advisory lock covering an entire generation run
        (from opening the browser to reading the finished file) — not just
        server startup. Confirmed real failure mode (2026-07-23): Claude
        Desktop restarted its MCP connection after a slow generation
        looked stuck, without the old server process actually exiting,
        leaving two independent songforge-mcp processes each with their
        own `self._lock` (asyncio.Lock, process-local only) and their own
        job registry. A `generate_vocal_track` call landing on one process
        and a `check_vocal_track_status` call landing on the other means
        the status check queries a registry that never received the job —
        and both processes could independently drive Playwright against
        the same single-GPU ACE-Step server at once. This lock makes that
        physically impossible regardless of how many separate processes
        exist: only one generation runs system-wide at a time."""
        return ACEStepClient._try_acquire_lock_file(lock_path, Timeouts.GENERATION + 60)

    async def ensure_server_running(self, on_progress: ProgressCallback | None = None) -> None:
        """Start ACE-Step's Gradio server as a detached process if it isn't
        already running. Detached (not a plain subprocess this process
        waits on) because the server takes minutes to load its checkpoint
        and must keep running independent of any single tool call."""
        self._require_configured()
        if await asyncio.to_thread(self._is_server_up):
            return

        venv_python = _venv_python_path(self.acestep_home)
        pipeline_script = os.path.join(self.acestep_home, "acestep", "acestep_v15_pipeline.py")
        if not os.path.isfile(venv_python) or not os.path.isfile(pipeline_script):
            raise SongForgeMCPError(
                ErrorCode.ACESTEP_NOT_CONFIGURED,
                f"Expected ACE-Step venv/entrypoint not found under {self.acestep_home}. "
                "Re-run install (see docs/INSTALLATION.md).",
            )

        stdout_log = os.path.join(self.acestep_home, "mcp_server_stdout.txt")
        stderr_log = os.path.join(self.acestep_home, "mcp_server_stderr.txt")
        ensure_private_dir(Paths.OUTPUT_ROOT)
        lock_path = os.path.join(Paths.OUTPUT_ROOT, ".acestep_launch.lock")
        launched_by_us = False

        try:
            deadline = time.time() + Timeouts.SERVER_STARTUP
            while time.time() < deadline:
                if await asyncio.to_thread(self._is_server_up):
                    if on_progress:
                        await on_progress(0.0, "ACE-Step server is up — starting generation")
                    return

                if not launched_by_us:
                    launched_by_us = await asyncio.to_thread(self._acquire_launch_lock, lock_path)
                    if launched_by_us:
                        if on_progress:
                            await on_progress(
                                0.0,
                                "ACE-Step server not running — starting it now (this can take "
                                "several minutes on first load)",
                            )
                        args = [
                            venv_python, pipeline_script,
                            "--port", str(GradioServer.PORT),
                            "--server-name", GradioServer.HOST,
                            "--language", "en",
                            "--config_path", GradioServer.CHECKPOINT,
                            "--lm_model_path", "acestep-5Hz-lm-0.6B",
                            "--init_service", "true",
                        ]
                        # Launched detached (not via asyncio.create_subprocess_exec,
                        # which this process would otherwise remain a parent/waiter
                        # of) - the server takes minutes to load its checkpoint and
                        # must keep running independent of this tool call and this
                        # MCP server process both. Native subprocess detachment
                        # flags, rather than shelling out to PowerShell's
                        # Start-Process, avoid both a Windows-only dependency and an
                        # f-string-built shell command line.
                        detach_kwargs: dict = (
                            {
                                "creationflags": (
                                    subprocess.DETACHED_PROCESS
                                    | subprocess.CREATE_NEW_PROCESS_GROUP
                                    | subprocess.CREATE_NO_WINDOW
                                )
                            }
                            if _IS_WINDOWS
                            else {"start_new_session": True}
                        )
                        # ACE-Step's own generation watchdog (ACESTEP_GENERATION_TIMEOUT,
                        # default 600s if unset) is separate from this client's own
                        # Timeouts.GENERATION polling patience above - the server was
                        # never told about our more generous budget, so it could kill a
                        # legitimately-still-running generation (e.g. reference-audio's
                        # slower backend, or a long-duration request) at 600s while this
                        # client was still happily waiting up to Timeouts.GENERATION.
                        # Must match here explicitly rather than rely on ambient
                        # environment, since nothing else sets this for a freshly
                        # spawned server process.
                        launch_env = os.environ.copy()
                        launch_env["ACESTEP_GENERATION_TIMEOUT"] = str(int(Timeouts.GENERATION))
                        try:
                            with open(stdout_log, "wb") as out_f, open(stderr_log, "wb") as err_f:
                                await asyncio.to_thread(
                                    subprocess.Popen,
                                    args,
                                    cwd=self.acestep_home,
                                    stdout=out_f,
                                    stderr=err_f,
                                    stdin=subprocess.DEVNULL,
                                    env=launch_env,
                                    **detach_kwargs,
                                )
                        except OSError as e:
                            raise SongForgeMCPError(
                                ErrorCode.SUBPROCESS_FAILED, f"failed to launch ACE-Step server: {e}"
                            ) from e
                    else:
                        if on_progress:
                            await on_progress(
                                0.0, "Another process is already starting the ACE-Step server — waiting for it"
                            )

                if on_progress:
                    remaining = int(deadline - time.time())
                    await on_progress(0.0, f"Waiting for ACE-Step server to finish loading (~{remaining}s left before timeout)")
                await asyncio.sleep(3)
        finally:
            if launched_by_us:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

        raise SongForgeMCPError(
            ErrorCode.SUBPROCESS_TIMEOUT,
            f"ACE-Step server did not start within {Timeouts.SERVER_STARTUP}s "
            f"(checkpoint loading can be slow on first run). Check {stderr_log}.",
        )

    # Several Optional Parameters fields are paired with an "X Auto"
    # checkbox that must be unchecked before the actual value field
    # becomes interactive at all (confirmed live: filling "BPM (Beats Per
    # Minute)" while "BPM Auto" is checked times out, since the field
    # exists in the DOM but isn't enabled) - the auto-checkbox's own label
    # isn't a simple derivation of the field's label (it's a short name,
    # not "<field label> Auto"), so this is an explicit lookup rather than
    # a string transformation.
    _AUTO_CHECKBOX_FOR_LABEL = {
        "BPM (Beats Per Minute)": "BPM Auto",
        "Key": "Key Auto",
        "TimeSig": "TimeSig Auto",
        "Vocal Language": "Language Auto",
        "Audio Duration (seconds)": "Duration Auto",
    }

    # "Key" specifically has no working accessible name at all - its real
    # DOM label text is "Key" immediately concatenated with its long info
    # tooltip with no separator, so the computed accessible name is that
    # whole run-on string, not "Key" alone; get_by_label("Key", exact=True)
    # never matches anything, confirmed live. Its placeholder text is a
    # reliable, unambiguous anchor instead.
    _PLACEHOLDER_FALLBACK_FOR_LABEL = {
        "Key": "e.g. C Major, A Minor, F# Minor",
    }

    @staticmethod
    async def _set_field_by_label(page, label: str, value) -> None:
        """Generic escape hatch for ACE-Step's ~70 advanced settings
        without hardcoding each one — locate a UI control by its exact
        visible label and set it. Deliberately only touches fields the
        caller explicitly names; every other field is left exactly at
        the UI's own default (the lesson from three failed attempts at
        reconstructing all 72 raw-API defaults by hand)."""
        placeholder = ACEStepClient._PLACEHOLDER_FALLBACK_FOR_LABEL.get(label)
        if placeholder:
            field = page.get_by_placeholder(placeholder, exact=True).first
        else:
            field = page.get_by_label(label, exact=True).first
        try:
            count = await field.count()
            visible = bool(count) and await field.is_visible()
        except Exception:
            count = 0
            visible = False
        if not visible:
            # Several real fields (BPM, Key, Audio Duration among them) live
            # behind the collapsed "Optional Parameters" accordion. Gradio
            # keeps them present in the DOM even while collapsed (count=1),
            # just not visible - confirmed live, and the reason an earlier
            # version of this fix (gated on count == 0) never actually
            # fired: count was already 1, just invisible, so the expansion
            # click never ran and .fill() kept timing out waiting for an
            # element that was never going to become visible. Checking
            # is_visible() rather than count, and doing this BEFORE the
            # auto-checkbox lookup below (that checkbox lives behind the
            # same accordion, so checking it first would find it present
            # but also invisible and skip unchecking).
            try:
                await page.get_by_role("button", name="Optional Parameters", exact=False).click(timeout=3000)
                await page.wait_for_timeout(500)
                count = await field.count()
                visible = bool(count) and await field.is_visible()
            except Exception:
                pass

        auto_checkbox_label = ACEStepClient._AUTO_CHECKBOX_FOR_LABEL.get(label)
        if auto_checkbox_label:
            try:
                auto_checkbox = page.get_by_label(auto_checkbox_label, exact=True).first
                if await auto_checkbox.count() and await auto_checkbox.is_checked():
                    await auto_checkbox.uncheck(timeout=3000)
                    await page.wait_for_timeout(300)
            except Exception:
                pass

        if count == 0:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"advanced setting {label!r} was not found in the UI — check the exact "
                f"label text (case/wording sensitive) in the ACE-Step Playground",
            )
        if isinstance(value, bool):
            if value:
                await field.check(timeout=5000)
            else:
                await field.uncheck(timeout=5000)
            actual = await field.is_checked()
            if actual != value:
                raise SongForgeMCPError(
                    ErrorCode.SYNTHESIS_FAILED,
                    f"advanced setting {label!r} did not take effect - set to {value!r} "
                    f"but the field now reads {actual!r}",
                )
            return
        try:
            await field.select_option(str(value), timeout=3000)
            await ACEStepClient._verify_field_value(field, label, value)
            return
        except SongForgeMCPError:
            raise
        except Exception:
            pass
        try:
            await field.fill(str(value), timeout=5000)
            await ACEStepClient._verify_field_value(field, label, value)
            return
        except SongForgeMCPError:
            raise
        except Exception:
            pass
        # Gradio Radio groups render as separate labeled inputs rather than
        # a single fillable/selectable element - fall back to clicking the
        # specific option whose own text matches the requested value.
        try:
            radio_option = page.get_by_label(str(value), exact=True).first
            await radio_option.check(timeout=5000)
            if not await radio_option.is_checked():
                raise SongForgeMCPError(
                    ErrorCode.SYNTHESIS_FAILED,
                    f"advanced setting {label!r} did not take effect - clicked the "
                    f"{value!r} option but it did not end up checked",
                )
        except SongForgeMCPError:
            raise
        except Exception as e:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"could not set advanced setting {label!r} to {value!r} - tried fill, "
                f"select, and radio-check, all failed: {e}",
            ) from e

    @staticmethod
    async def _verify_field_value(field, label: str, expected) -> None:
        """Reads a field back after fill()/select_option() and confirms the
        value actually stuck, instead of trusting a non-throwing call as
        success. Real, confirmed bug class this guards against: a Gradio
        combobox (e.g. the Track Name dropdown during Extract-mode
        testing) where .fill() succeeds by typing into the filter input
        without ever registering a real backend selection - the UI looks
        right but the value silently never applied. Numeric values get a
        tolerant float comparison since Gradio number inputs commonly
        reformat ("150" -> "150.0"); text comparison is case-insensitive
        since ACE-Step may normalize casing internally without that being
        a real failure."""
        try:
            actual = await field.input_value(timeout=3000)
        except Exception:
            return  # not a value-bearing input (e.g. a button) - nothing to verify

        expected_str = str(expected).strip()
        if actual.strip().lower() == expected_str.lower():
            return
        try:
            if abs(float(actual) - float(expected_str)) < 1e-6:
                return
        except ValueError:
            pass
        raise SongForgeMCPError(
            ErrorCode.SYNTHESIS_FAILED,
            f"advanced setting {label!r} did not take effect - set to {expected!r} but "
            f"the field now reads {actual!r}",
        )

    @staticmethod
    async def _set_output_format(page, output_format: str) -> None:
        """Sets the Audio Format this generation will be saved as. Unlike
        a normal advanced_settings field, this control lives in the
        Settings panel (not the main generation form) behind a collapsed
        "Audio Output & Post-processing" accordion, and it isn't a native
        <select> - it's a combobox (role="listbox" input) that opens a
        list of role="option" elements on click. Both of those were
        confirmed by inspecting the live DOM, not assumed from source
        alone."""
        label = _OUTPUT_FORMAT_OPTION_LABELS[output_format]
        await page.get_by_role("button", name="Settings", exact=False).first.click()
        await page.wait_for_timeout(500)
        await page.get_by_text("Audio Output", exact=False).first.click()
        await page.wait_for_timeout(300)
        format_field = page.get_by_label("Audio Format", exact=True).first
        await format_field.click()
        await page.wait_for_timeout(200)
        await page.get_by_role("option", name=label, exact=True).click()
        await page.wait_for_timeout(200)
        # Close the Settings panel rather than leaving it open over the
        # generation form for the rest of this session.
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)

    @staticmethod
    async def _switch_to_remix_mode(
        page,
        remix_source_path: str,
        remix_strength: float,
        melody_retention: float | None,
        no_fsq: bool = False,
    ) -> None:
        """Switches Generation Mode to "Remix" and sets its Source Audio +
        Remix Strength + (optionally) Cover Strength / Melody Retention +
        (optionally) no_fsq. Unlike "Reference Audio" (Custom mode),
        Remix's source audio is used whole and unmodified — confirmed
        from ACE-Step's own source (process_src_audio in
        acestep/core/generation/handler/io_audio.py just normalizes to
        stereo 48kHz, no front/middle/back sampling-and-stitching) — so
        it doesn't have the internal-inconsistency problem Reference
        Audio does for anything longer than ~30s. Remix Strength controls
        how closely the result follows the source's own melody/structure;
        Melody Retention is a genuinely separate control ACE-Step's own
        docs describe as "how much of the original melody is preserved...
        0 = no melody retention (pure style transfer from your caption)"
        — its own default (0.2) is already inside ACE-Step's documented
        recommended range for the SFT model (0.1-0.25). `no_fsq`
        (confirmed from ACE-Step's own source,
        generation_tab_secondary_controls.py: "Use source-audio latents
        directly instead of FSQ-quantized audio codes", default False)
        bypasses a finite discrete-code quantization step in how the
        source's structure is encoded — a real, previously-untested
        candidate for the garbled/gibberish-vocal symptom, since lossy
        quantization is a plausible source of exactly that kind of
        reconstruction artifact."""
        await page.get_by_label("Remix", exact=True).first.check()
        await page.wait_for_timeout(500)

        src_section = page.locator("text=Source Audio").locator(
            "xpath=ancestor::*[contains(@class,'block')][1]"
        )
        await src_section.locator("input[type='file']").first.set_input_files(remix_source_path)
        await page.wait_for_timeout(6000)

        strength_block = page.locator("text=Remix Strength").locator(
            "xpath=ancestor::*[contains(@class,'block')][1]"
        )
        await strength_block.locator("input[type='number']").first.fill(str(remix_strength))
        await page.wait_for_timeout(300)

        if melody_retention is not None:
            retention_block = page.locator("text=Cover Strength (Melody Retention)").locator(
                "xpath=ancestor::*[contains(@class,'block')][1]"
            )
            await retention_block.locator("input[type='number']").first.fill(str(melody_retention))
            await page.wait_for_timeout(300)

        if no_fsq:
            checkbox = page.get_by_label("no_fsq", exact=True)
            await checkbox.first.check()
            await page.wait_for_timeout(300)

    @staticmethod
    async def _ensure_accordion_open(page, header_name: str, content_locator) -> None:
        """Opens the accordion headed by header_name, but only if
        content_locator (something inside it that's only visible when it's
        expanded) isn't already visible. Accordion headers are toggle
        buttons, not one-way "open" buttons — clicking one that's already
        open closes it instead. This bit _load_lora for real:
        _set_output_format's trailing Escape key press only dismisses its
        dropdown, not the Settings accordion itself (confirmed live —
        Settings stayed open, marked '▼', afterward), so _load_lora's old
        unconditional click on "Settings" closed it, hiding the nested
        "LoRA Adapter" accordion and timing out on a button that no longer
        existed. Checking real state instead of assuming it fixes this
        regardless of call order."""
        if await content_locator.is_visible():
            return
        await page.get_by_role("button", name=header_name, exact=False).first.click(timeout=5000)
        await content_locator.wait_for(state="visible", timeout=5000)

    async def _load_lora(self, page, lora_path: str) -> None:
        """Loads a trained LoRA/LoKr adapter before generation. The control
        lives two accordions deep: the outer "Settings" accordion, then a
        nested "LoRA Adapter" accordion inside it — confirmed by reading
        generation_advanced_primary_controls.py's build_lora_controls(),
        not guessed; get_by_label("LoRA Path") alone never resolves it
        (count=0) regardless of accordion state, so this locates the field
        by its placeholder text instead, the same fix already needed for
        ACE-Step's "Key" field. Also checks "Use LoRA" — loading alone
        does not enable it for generation."""
        lora_adapter_header = page.get_by_role("button", name="LoRA Adapter", exact=False).first
        await self._ensure_accordion_open(page, "Settings", lora_adapter_header)
        await page.wait_for_timeout(300)

        path_field = page.get_by_placeholder("./lora_output/final/adapter", exact=True).first
        await self._ensure_accordion_open(page, "LoRA Adapter", path_field)
        await page.wait_for_timeout(300)

        await path_field.fill(lora_path, timeout=5000)
        await page.wait_for_timeout(300)

        await page.get_by_role("button", name="Load LoRA", exact=False).first.click(timeout=5000)
        await page.wait_for_timeout(3000)

        use_lora_checkbox = page.get_by_label("Use LoRA", exact=True).first
        if not await use_lora_checkbox.is_checked():
            await use_lora_checkbox.check(timeout=5000)
            await page.wait_for_timeout(300)

    @staticmethod
    async def _read_progress_text(page) -> str:
        """Best-effort scrape of ACE-Step's own progress display (observed
        format: "Generating music (batch size: N)... - 52.0%" alongside a
        separate "elapsed/estimated s" readout). Returns an empty string if
        nothing progress-shaped is currently visible - this is scraped
        from human-readable UI text, not a stable API, so callers must
        treat a miss as "no update available" rather than an error."""
        try:
            texts = await page.locator("text=/Generating music/i").all_inner_texts()
        except Exception:
            texts = []
        if not texts:
            return ""
        return texts[0]

    async def generate(
        self,
        caption: str,
        lyrics: str,
        reference_audio_path: str | None = None,
        advanced_settings: dict | None = None,
        output_format: str = "wav",
        remix_source_path: str | None = None,
        remix_strength: float = 0.5,
        remix_melody_retention: float | None = None,
        remix_no_fsq: bool = False,
        song_title: str | None = None,
        lora_path: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Drive ACE-Step's Gradio UI to generate one song. `lora_path`,
        if given, loads a trained LoRA/LoKr adapter (e.g. a directory like
        `.../lokr_output/my_voice_v2/final`) before generating, and
        enables it via ACE-Step's own "Use LoRA" toggle — loading alone
        does not apply it. `song_title`,
        if given, is slugified into the output filename (e.g. "Hearts on
        a Wire" -> `hearts-on-a-wire_<timestamp>.wav`) so generated files
        are identifiable by name instead of only a timestamp/UUID — pick a
        real title for the song, not a placeholder. `advanced_settings`
        is an optional {exact UI label: value} map for overriding specific
        settings (e.g. {"Guidance Scale": 8.5, "Seed": "12345", "Think": False})
        when default output isn't right — every field not named here is left
        untouched at the UI's own default. `output_format` must be one of
        OUTPUT_FORMATS; defaults to "wav" (uncompressed, better for further
        editing than ACE-Step's own "mp3" UI default). `remix_source_path`,
        if given, switches to ACE-Step's Remix mode instead of Custom mode
        (mutually exclusive with `reference_audio_path` — different
        generation modes) — the source audio is used whole, unlike
        Reference Audio's broken front/middle/back sampling, and
        `remix_strength` (0.0-1.0, ACE-Step's own documented range)
        controls how closely the result follows its melody/structure.
        `remix_melody_retention`, if given, overrides ACE-Step's own
        "Cover Strength (Melody Retention)" (0.0-1.0, default 0.2, already
        inside ACE-Step's own recommended low range for the SFT model) —
        an experimental knob for cases where new lyrics don't fit the
        source's locked melodic contour. `remix_no_fsq`, if True, bypasses
        ACE-Step's FSQ (finite scalar quantization) discrete-code encoding
        of the source's structure in favor of using its continuous
        latents directly — a real, previously-untested candidate for
        garbled/gibberish-vocal output, since lossy quantization is a
        plausible source of exactly that artifact. `on_progress`, if
        given, is awaited with (fraction_complete: float in [0, 1],
        message: str) periodically throughout — server startup,
        generation percentage when ACE-Step exposes one, and elapsed time
        otherwise. Returns {"audio_path": str, "generation_seconds": float}."""
        async with self._lock:
            ensure_private_dir(Paths.OUTPUT_ROOT)
            lock_path = os.path.join(Paths.OUTPUT_ROOT, ".acestep_generate.lock")
            deadline = time.time() + Timeouts.GENERATION
            acquired = False
            announced_wait = False
            while time.time() < deadline:
                if await asyncio.to_thread(self._acquire_generate_lock, lock_path):
                    acquired = True
                    break
                if on_progress and not announced_wait:
                    await on_progress(
                        0.0, "Another generation is already in progress on this machine — waiting for it to finish"
                    )
                    announced_wait = True
                await asyncio.sleep(3)
            if not acquired:
                raise SongForgeMCPError(
                    ErrorCode.SUBPROCESS_TIMEOUT,
                    "timed out waiting for another in-progress generation to finish",
                )
            try:
                return await self._generate_locked(
                    caption, lyrics, reference_audio_path, advanced_settings, output_format,
                    remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq,
                    song_title, lora_path, on_progress,
                )
            finally:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    async def _generate_locked(
        self,
        caption: str,
        lyrics: str,
        reference_audio_path: str | None,
        advanced_settings: dict | None,
        output_format: str,
        remix_source_path: str | None,
        remix_strength: float,
        remix_melody_retention: float | None,
        remix_no_fsq: bool,
        song_title: str | None,
        lora_path: str | None,
        on_progress: ProgressCallback | None,
    ) -> dict:
        await self.ensure_server_running(on_progress)
        start_time = time.time()

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1600, "height": 2400})
                try:
                    await page.goto(GradioServer.URL, timeout=60000)
                except Exception as e:
                    raise SongForgeMCPError(
                        ErrorCode.SUBPROCESS_FAILED, f"could not reach ACE-Step UI: {e}"
                    ) from e
                await page.wait_for_timeout(4000)

                await self._set_output_format(page, output_format)

                if lora_path:
                    if not os.path.isdir(lora_path):
                        raise SongForgeMCPError(
                            ErrorCode.INVALID_PARAMETER, f"lora_path does not exist: {lora_path}"
                        )
                    await page.get_by_label("Custom", exact=True).check()
                    await page.wait_for_timeout(500)
                    await self._load_lora(page, lora_path)

                if remix_source_path:
                    if not os.path.isfile(remix_source_path):
                        raise SongForgeMCPError(
                            ErrorCode.INVALID_PARAMETER,
                            f"remix_source_path does not exist: {remix_source_path}",
                        )
                    await self._switch_to_remix_mode(
                        page, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq
                    )

                await page.get_by_label("Music Caption").click()
                await page.get_by_label("Music Caption").fill(caption)
                await page.get_by_label("Lyrics").click()
                await page.get_by_label("Lyrics").fill(lyrics)

                reference_audio_confirmed = False
                if reference_audio_path:
                    if not os.path.isfile(reference_audio_path):
                        raise SongForgeMCPError(
                            ErrorCode.INVALID_PARAMETER,
                            f"reference_audio_path does not exist: {reference_audio_path}",
                        )
                    trimmed_reference_path = await asyncio.to_thread(
                        _trim_reference_for_key_coherence, reference_audio_path
                    )
                    ref_section = page.locator("text=Reference Audio").locator(
                        "xpath=ancestor::*[contains(@class,'block')][1]"
                    )
                    await ref_section.locator("input[type='file']").first.set_input_files(
                        trimmed_reference_path
                    )
                    # Confirm the upload actually registered in the UI before
                    # proceeding - a blind fixed wait here previously let
                    # generation start with no reference audio attached at all
                    # (indistinguishable from the base model) whenever the
                    # upload happened to take longer than the wait, with no
                    # error or signal that anything was wrong. Recorded below
                    # (reference_audio_confirmed) and returned to the caller so
                    # it's visible in the job result, not just enforced silently.
                    try:
                        await ref_section.locator("audio").first.wait_for(
                            state="attached", timeout=20000
                        )
                        reference_audio_confirmed = True
                        logger.info(
                            f"Reference audio confirmed attached in UI: {trimmed_reference_path}"
                        )
                    except Exception as e:
                        raise SongForgeMCPError(
                            ErrorCode.SUBPROCESS_FAILED,
                            "Reference audio did not register in the UI within 20s - "
                            "generation would have proceeded without it. Not continuing.",
                        ) from e

                # Batch Size defaults to ACE-Step's own "2" (two full takes
                # generated per request) unless overridden. Relying on the
                # calling LLM to remember to pass "Batch Size": 1 in
                # advanced_settings every time was confirmed unreliable in
                # production (a live generation was caught still running
                # batch_size=2 after the instructions had already been
                # updated and Claude Desktop restarted) - so it's forced
                # here in code instead. An explicit advanced_settings value
                # still wins, for a future multi-take feature.
                effective_settings = {"Batch Size": 1, **(advanced_settings or {})}
                for label, value in effective_settings.items():
                    await self._set_field_by_label(page, label, value)
                await page.wait_for_timeout(500)

                if on_progress:
                    await on_progress(0.0, "Submitting generation request")

                await page.get_by_role("button", name="Generate Music").click()
                # Do not close/navigate for several seconds — closing too
                # soon after this click can abort the request server-side
                # before it's even registered (confirmed in testing).
                await page.wait_for_timeout(5000)

                deadline = time.time() + Timeouts.GENERATION
                status_value = ""
                while time.time() < deadline:
                    status_box = (
                        page.locator("text=Generation Status")
                        .locator("..")
                        .locator("textarea, input")
                        .first
                    )
                    try:
                        status_value = await status_box.input_value(timeout=2000)
                    except Exception:
                        status_value = ""
                    if "Complete" in status_value:
                        if on_progress:
                            await on_progress(1.0, "Generation complete")
                        break
                    if "Error" in status_value or "Failed" in status_value:
                        raise SongForgeMCPError(
                            ErrorCode.SYNTHESIS_FAILED, f"ACE-Step reported: {status_value}"
                        )

                    if on_progress:
                        elapsed = time.time() - start_time
                        progress_text = await self._read_progress_text(page)
                        percent_match = _PERCENT_RE.search(progress_text) if progress_text else None
                        if percent_match:
                            fraction = min(float(percent_match.group(1)) / 100.0, 0.99)
                            eta_match = _ETA_RE.search(progress_text)
                            if eta_match:
                                msg = f"Generating - {percent_match.group(1)}% ({eta_match.group(1)}/{eta_match.group(2)}s)"
                            else:
                                msg = f"Generating - {percent_match.group(1)}%"
                        else:
                            # No live percentage available yet (e.g. still
                            # in the LM/lyric-conditioning phase) - report
                            # elapsed time so the caller isn't left blind.
                            fraction = min(elapsed / Timeouts.GENERATION, 0.99)
                            msg = f"Generating - {elapsed:.0f}s elapsed, no live percentage yet"
                        await on_progress(fraction, msg)

                    # Coarser than the earlier one-shot post-click wait -
                    # this interval repeats for the whole generation, and
                    # a tight 5s poll here would emit up to ~180 progress
                    # notifications over the 900s worst-case ceiling. 15s
                    # keeps the common ~70-100s case to a handful of
                    # updates without leaving long-tail runs silent.
                    await page.wait_for_timeout(15000)
                else:
                    raise SongForgeMCPError(
                        ErrorCode.SUBPROCESS_TIMEOUT,
                        f"Generation did not complete within {Timeouts.GENERATION}s",
                    )
            finally:
                await browser.close()

        output_dir = os.path.join(self.acestep_home, "gradio_outputs")
        ext = _OUTPUT_FORMAT_EXTENSIONS[output_format]
        candidates = [
            f for f in Path(output_dir).rglob(f"*.{ext}") if f.stat().st_mtime >= start_time
        ]
        if not candidates:
            raise SongForgeMCPError(
                ErrorCode.SYNTHESIS_FAILED,
                "ACE-Step reported completion but no new output file was found",
            )
        newest = max(candidates, key=lambda f: f.stat().st_mtime)
        slug = _slugify_title(song_title) if song_title else ""
        filename = f"{slug}_{int(start_time)}{newest.suffix}" if slug else f"{int(start_time)}_{newest.name}"
        final_path = os.path.join(Paths.OUTPUT_DIR, filename)
        shutil.copy(str(newest), final_path)
        return {
            "audio_path": final_path,
            "generation_seconds": round(time.time() - start_time, 1),
            "reference_audio_confirmed": reference_audio_confirmed,
        }
