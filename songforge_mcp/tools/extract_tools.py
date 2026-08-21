import asyncio

from mcp.server.fastmcp import Context, FastMCP

from songforge_mcp.shared_state import acestep_client as _client, jobs as _jobs
from songforge_mcp_shared.constants import GradioServer
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import validate_audio_file_path


def register(mcp: FastMCP):
    @mcp.tool(structured_output=False)
    async def extract_track_from_mix(source_audio_path: str, track_name: str, ctx: Context) -> dict:
        """Starts isolating a single named track (e.g. "keyboard", vocals,
        drums) out of an EXISTING mix via ACE-Step's Extract generation
        mode. Returns {"job_id": str} immediately; poll
        check_vocal_track_status(job_id) exactly as for
        generate_vocal_track.

        This is the real, purpose-built tool for "give me back just the
        X part of this song" - do not reach for generate_vocal_track's
        remix_source_path for this, it's confirmed to just reproduce or
        garble the whole source rather than isolate any part of it.

        Real, unavoidable cost - warn the user before calling this: this
        server normally runs ACE-Step's SFT checkpoint (confirmed clearly
        better for everyday generation), but Extract mode only exists
        under the Base checkpoint. This tool automatically swaps the
        server to Base for the call and back to SFT afterward - roughly
        a minute or two PER swap on top of the actual extraction, so a
        typical call takes several minutes total, most of it checkpoint
        loading rather than extraction. It will not interrupt an
        in-progress generate_vocal_track call elsewhere (the swap is
        gated behind the same cross-process generation lock every
        generation on this server already respects) - it will simply
        wait its turn.

        Args:
            source_audio_path: Local path to the existing mix to extract
                from - the full song, not a pre-separated stem. Use a
                file already on disk (e.g. from an earlier
                generate_vocal_track call, or split_vocal_stems' own
                instrumental_path) rather than asking the user to
                describe what's in it.
            track_name: One of woodwinds, brass, fx, synth, strings,
                percussion, keyboard, guitar, bass, drums,
                backing_vocals, vocals - ACE-Step's own fixed vocabulary.
                Note "keyboard" covers piano; there is no separate
                "piano" option.

        On completion, check_vocal_track_status returns {"audio_path":
        str, "diagnostics": {"generation_seconds": float}}. The
        extraction quality on real mixed audio has not been extensively
        validated by this project yet - tell the user this is worth a
        real listen, not something to trust as correct by default.
        """
        if track_name not in GradioServer.EXTRACT_TRACK_NAMES:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"track_name must be one of {sorted(GradioServer.EXTRACT_TRACK_NAMES)}, got {track_name!r}",
            )
        source_audio_path = validate_audio_file_path(source_audio_path, param_name="source_audio_path")

        job = _jobs.create()

        async def report(fraction: float, message: str) -> None:
            job.progress = fraction
            job.message = message
            try:
                await ctx.report_progress(progress=fraction, total=1.0, message=message)
            except Exception:
                pass

        async def run_job() -> None:
            try:
                result = await _client.extract_track(
                    source_audio_path=source_audio_path,
                    track_name=track_name,
                    on_progress=report,
                )
                job.result = {
                    "audio_path": result["audio_path"],
                    "diagnostics": {"generation_seconds": result["generation_seconds"]},
                }
                job.progress = 1.0
                job.message = "Extraction complete"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}
