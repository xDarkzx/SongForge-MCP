import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# MCP's stdio transport requires UTF-8 JSON-RPC framing; Python's default
# stdio encoding otherwise follows the OS/locale default, which on Windows
# is a legacy codepage, not UTF-8. Must happen before anything touches
# stdio. Same fix as reaper-mcp's main.py, same underlying reason.
#
# Guarded per-stream: under pytest, sys.stdin is replaced with a
# DontReadFromInput object that has no .reconfigure(), which would raise
# AttributeError on import. A no-op there is safe — it only matters for the
# real stdio transport this module drives when actually run as a server.
for _stream in (sys.stdout, sys.stdin, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from songforge_mcp.banner import print_ready_banner, print_startup_banner
from songforge_mcp.shared_state import jobs as _jobs

# Printed here, before importing tool_registry (below), on purpose: some
# tool modules (e.g. midi_tools, via basic-pitch) print noisy
# third-party import-time warnings (missing optional coreml/tflite/onnx
# backends, TensorFlow's own startup logging) straight to stderr. Doing
# this import-time banner print first means those warnings land after the
# banner instead of before it, in the same console window.
print_startup_banner(_jobs)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from songforge_mcp.instructions import load_instructions  # noqa: E402
from songforge_mcp.tool_registry import register_all_tools  # noqa: E402
from songforge_mcp.tools.generate_tools import _client as _acestep_client  # noqa: E402
from songforge_mcp_shared.constants import GradioServer  # noqa: E402


@asynccontextmanager
async def _lifespan(_app: FastMCP) -> AsyncIterator[dict]:
    """Starts warming up ACE-Step's server as soon as this MCP server
    itself starts (i.e. as soon as Claude Desktop launches it) instead of
    waiting for the first generate_vocal_track call to discover it's cold
    and pay the multi-minute checkpoint-load cost right when a user is
    waiting on a result. Uses the exact same ACEStepClient instance (and
    its lock) that generate_vocal_track uses — a separate instance here
    would risk two competing launches racing to bind the same port if a
    real request comes in while this is still starting up.

    Fire-and-forget: this schedules the warm-up and returns immediately,
    so it never delays this server's own startup or its ability to accept
    the initialize handshake. Failures (e.g. SONGFORGE_ACESTEP_HOME not
    configured yet) are swallowed here — a real generate_vocal_track call
    will surface the same problem properly through its own error path.
    """
    async def warm_up() -> None:
        try:
            await _acestep_client.ensure_server_running()
            print_ready_banner(GradioServer.CHECKPOINT)
        except Exception as e:
            sys.stderr.write(f"[songforge-mcp] ACE-Step warm-up did not complete: {e}\n")

    warmup_task = asyncio.create_task(warm_up())
    try:
        yield {}
    finally:
        warmup_task.cancel()


mcp = FastMCP("SongForgeMCP", instructions=load_instructions(), lifespan=_lifespan)
register_all_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
