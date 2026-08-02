import asyncio
import os

from mcp.server.fastmcp import FastMCP

from songforge_mcp.separator_client import SeparatorClient
from songforge_mcp.shared_state import jobs as _jobs
from songforge_mcp.voice_reference_library import get_voice_library_status, save_voice_clip
from songforge_mcp.youtube_reference import YouTubeReferenceClient
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import validate_audio_file_path

_youtube_client = YouTubeReferenceClient()
_separator_client = SeparatorClient()


def register(mcp: FastMCP):
    @mcp.tool()
    async def prepare_voice_reference(
        voice_name: str,
        youtube_url: str | None = None,
        local_audio_path: str | None = None,
    ) -> dict:
        """Checks whether reference vocal clips already exist for a named
        voice, or prepares a new one from a YouTube link or a local audio
        file already on disk.

        Call with ONLY voice_name first (no youtube_url/local_audio_path)
        whenever the user wants to use a particular voice. Two possible
        results:
        - {"status": "found", ...} — includes total_duration_seconds and
          meets_recommended_minimum. One song typically yields only 1-2
          minutes of actual vocal once instrumental sections are excluded,
          so a single clip essentially never meets the recommended
          minimum (currently 600s) — if meets_recommended_minimum is
          False, tell the user how much they have and that more clips
          (ideally from different songs) are needed, don't treat one clip
          as done.
        - {"status": "not_found", "message": ...} — none exist yet. Ask
          the user for either a YouTube link or a local audio file path
          for a song featuring this voice (recommend an acoustic version
          if using YouTube, since sparser instrumentation separates into
          a cleaner vocal). Set the expectation up front that multiple
          clips will likely be needed, not just one. Do not search for or
          guess a link/path yourself.

        Once the user gives a youtube_url OR a local_audio_path (exactly
        one, not both), call this again with voice_name plus that one
        source to actually prepare it — this returns {"job_id": str}
        immediately (downloads if from YouTube, separates the vocal
        either way, and saves it into this voice's own library folder,
        clearly labeled with both the voice name and source). Poll with
        check_vocal_track_status(job_id) exactly as for
        generate_vocal_track (same tool, same registry); on completion it
        returns the same found-style status (clips, total_duration_seconds,
        meets_recommended_minimum) so you can tell the user whether to
        keep going or stop.

        Args:
            voice_name: The voice to check/prepare, e.g. "Annika Wells".
            youtube_url: A YouTube link featuring this voice. Only pass
                once the user has supplied a real link — omit for the
                initial check. Mutually exclusive with local_audio_path.
            local_audio_path: Path to an audio file already on disk
                (any format/location — a personal sample library, a
                downloaded clip, etc.) featuring this voice. Mutually
                exclusive with youtube_url.
        """
        if youtube_url is None and local_audio_path is None:
            status = get_voice_library_status(voice_name)
            if status["clips"]:
                return {"status": "found", **status}
            return {
                "status": "not_found",
                "message": (
                    f"No reference clips exist yet for '{voice_name}'. Ask the user for "
                    "either a YouTube link or a local audio file path for a song featuring "
                    "this voice — for YouTube, recommend an acoustic version if one exists, "
                    "since sparser instrumentation separates into a cleaner vocal. One clip "
                    "will almost certainly not be enough on its own (typically only 1-2 "
                    "minutes of usable vocal per song) — mention that several more will "
                    "likely be needed too. Call this tool again with that youtube_url or "
                    "local_audio_path once you have one."
                ),
            }

        if youtube_url is not None and local_audio_path is not None:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                "provide only one of youtube_url or local_audio_path, not both",
            )

        if local_audio_path is not None:
            local_audio_path = validate_audio_file_path(local_audio_path, param_name="local_audio_path")

        job = _jobs.create()

        async def run_job() -> None:
            try:
                if youtube_url is not None:
                    job.message = f"Downloading {youtube_url}"
                    source_path = await asyncio.to_thread(_youtube_client.download, youtube_url)
                else:
                    source_path = local_audio_path
                job.message = "Separating vocals"
                stems = await asyncio.to_thread(_separator_client.separate, source_path)
                source_label = os.path.splitext(os.path.basename(source_path))[0]
                await asyncio.to_thread(save_voice_clip, voice_name, stems["vocals_path"], source_label)
                job.result = {"status": "prepared", **get_voice_library_status(voice_name)}
                job.progress = 1.0
                job.message = "Voice reference prepared"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}
