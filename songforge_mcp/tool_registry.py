"""Auto-discovers and registers every tool module in songforge_mcp/tools/."""
import importlib
import logging
import pkgutil
import sys

from mcp.server.fastmcp import FastMCP

import songforge_mcp.tools as tools_package

logger = logging.getLogger(__name__)

_EXPECTED_MODULES = frozenset(
    {
        "generate_tools",
        "separate_tools",
        "analysis_tools",
        "midi_tools",
        "voice_reference_tools",
        "playback_tools",
        "track_management_tools",
        "audio_edit_tools",
        "extract_tools",
    }
)


def register_all_tools(mcp: FastMCP) -> None:
    """Discover and register every tool module in songforge_mcp/tools/.

    A module counts as a tool provider if it defines register(mcp). If a
    module raises during import or registration, log loudly and keep going
    — one broken file shouldn't take the whole server down.
    """
    registered: list[str] = []
    failures: list[tuple[str, Exception]] = []

    for _finder, name, _ispkg in pkgutil.iter_modules(tools_package.__path__):
        try:
            module = importlib.import_module(f"songforge_mcp.tools.{name}")
        except Exception as e:
            logger.error("IMPORT FAILED for tool module %s: %s", name, e, exc_info=True)
            sys.stderr.write(f"[songforge-mcp] FAILED to import tool module '{name}': {e}\n")
            failures.append((name, e))
            continue

        if not hasattr(module, "register"):
            continue

        try:
            module.register(mcp)
            registered.append(name)
        except Exception as e:
            logger.error("REGISTER FAILED for %s: %s", name, e, exc_info=True)
            sys.stderr.write(f"[songforge-mcp] Tool registration failed for '{name}': {e}\n")
            failures.append((name, e))

    missing = _EXPECTED_MODULES - set(registered) - {n for n, _ in failures}
    if missing:
        sys.stderr.write(f"[songforge-mcp] WARNING: expected module(s) not found on disk: {sorted(missing)}\n")
    sys.stderr.write(f"[songforge-mcp] registered {len(registered)} tool module(s)\n")
    if failures:
        sys.stderr.write(f"[songforge-mcp] {len(failures)} tool module(s) failed to load: {[n for n, _ in failures]}\n")
