"""Cosmetic startup console output for songforge-mcp.

Purely decorative - fills the otherwise-blank console window Claude
Desktop pops up when it launches this server (a console-subsystem exe on
Windows always gets one, whether or not anything is printed to it).

MUST write to stderr only, never stdout: this server runs over MCP's
stdio transport (see main.py), where stdout is the JSON-RPC channel
Claude Desktop reads. Anything printed to stdout here would corrupt that
stream and break the connection.
"""
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import colorama
from colorama import Fore, Style

from songforge_mcp.job_registry import JobRegistry
from songforge_mcp_shared.constants import GradioServer, Paths

colorama.just_fix_windows_console()

# 5x7 dot-matrix glyphs for a big block-letter "SONGFORGE" logo - only the
# letters the word actually needs. Rendered at 2 characters per pixel (see
# _render_logo) so it reads as a large, chunky banner rather than the thin
# single-line-per-pixel look a 1:1 render would give in a monospace font.
_FONT = {
    "S": ["01110", "10000", "10000", "01110", "00001", "00001", "11110"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10101", "10011", "10001", "10001"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
}


def _render_logo(word: str) -> str:
    pixel = "██"
    rows = []
    for row in range(7):
        rows.append("  ".join("".join(pixel if b == "1" else "  " for b in _FONT[ch][row]) for ch in word))
    return "\n".join(rows)


_ART = _render_logo("SONGFORGE")


def _get_version() -> str:
    try:
        return version("songforge-mcp")
    except PackageNotFoundError:
        return "dev"


def _detect_gpu() -> str | None:
    """Best-effort GPU name + VRAM via nvidia-smi - never raises. This
    server's own venv has no torch (that only lives in ACE-Step's own
    venv), so shelling out to nvidia-smi directly is the only detection
    available from this process without launching a second interpreter."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        return line.strip()
    except Exception:
        return None


def print_startup_banner(jobs: JobRegistry) -> None:
    """Printed once, synchronously, before mcp.run() takes over stdio.
    Static snapshot only - job counts reflect this instant (always 0 on a
    fresh server), not a live view."""
    running = sum(1 for j in jobs.list_all() if j.status == "running")
    complete = sum(1 for j in jobs.list_all() if j.status != "running")

    lines = [
        Fore.MAGENTA + _ART + Style.RESET_ALL,
        Fore.CYAN + f"  ♪♫  SongForge MCP v{_get_version()}  ♫♪" + Style.RESET_ALL,
        "",
        f"  {Fore.YELLOW}PID{Style.RESET_ALL}            {os.getpid()}",
        f"  {Fore.YELLOW}ACE-Step home{Style.RESET_ALL}  {Paths.ACESTEP_HOME or '(not configured)'}",
        f"  {Fore.YELLOW}ACE-Step port{Style.RESET_ALL}  {GradioServer.PORT}",
        f"  {Fore.YELLOW}Output dir{Style.RESET_ALL}     {Paths.OUTPUT_DIR}",
        f"  {Fore.YELLOW}Jobs{Style.RESET_ALL}           {running} active / {complete} completed (session start)",
        "",
        Fore.GREEN + "  \U0001f3b5 warming up ACE-Step..." + Style.RESET_ALL,
        "",
    ]
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def print_ready_banner(checkpoint: str) -> None:
    """Printed once, from main.py's warm_up() success path, once
    ACE-Step actually confirms up."""
    gpu = _detect_gpu()
    gpu_line = f"  {Fore.YELLOW}GPU{Style.RESET_ALL}            {gpu}\n" if gpu else ""
    sys.stderr.write(
        "\n"
        + Fore.GREEN + Style.BRIGHT + "  ♪♫♪  ACE-Step is READY  ♪♫♪" + Style.RESET_ALL + "\n"
        + gpu_line
        + f"  {Fore.YELLOW}Checkpoint{Style.RESET_ALL}     {checkpoint}\n"
        + "\n"
    )
    sys.stderr.flush()
