import asyncio
import random

from mcp.server.fastmcp import Context, FastMCP

from songforge_mcp.acestep_client import ACEStepClient
from songforge_mcp.audio_analysis import analyze_audio
from songforge_mcp.media_player import play_audio_now
from songforge_mcp.shared_state import jobs as _jobs, separator_client as _separator_client
from songforge_mcp.youtube_reference import YouTubeReferenceClient
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import (
    measure_wav_duration_seconds,
    validate_audio_file_path,
    validate_caption,
    validate_lyrics,
    validate_output_format,
)

_MIN_TAKES = 2
_MAX_TAKES = 5

_client = ACEStepClient()
_youtube_client = YouTubeReferenceClient()


def register(mcp: FastMCP):
    @mcp.tool(structured_output=False)
    async def generate_vocal_track(
        caption: str,
        lyrics: str,
        ctx: Context,
        reference_audio_path: str | None = None,
        reference_youtube_url: str | None = None,
        advanced_settings: dict | None = None,
        output_format: str = "wav",
        remix_source_path: str | None = None,
        remix_source_youtube_url: str | None = None,
        remix_strength: float = 0.5,
        remix_melody_retention: float | None = None,
        remix_no_fsq: bool = False,
        song_title: str | None = None,
        split_stems: bool = False,
        lora_path: str | None = None,
    ) -> dict:
        """Start generating a full EDM/vocal track (music + vocals
        together) via ACE-Step 1.5. Returns {"job_id": str} immediately —
        does not wait for generation to finish. Poll
        check_vocal_track_status(job_id) to follow it through; see that
        tool's own docs for how sparse updates should be.

        Renders a complete song, not an isolated vocal — the default,
        complete deliverable. Don't call split_vocal_stems unless the
        user explicitly asks for the stems.

        Args:
            caption: Genre/mood/instrumentation/vocal-style tags, e.g.
                "melodic dubstep, female vocals, dreamy, atmospheric, 150
                BPM". Never name a real artist — describe their sound
                instead. This server does not clone real people's voices.
            lyrics: Full original lyrics with structure tags — [verse],
                [pre-chorus], [chorus], [bridge], [outro], or
                [instrumental]/[inst]. Must be original, not reproduced
                from a real song.
            reference_audio_path: Optional local audio file for style
                reference. Confirmed to risk detuning the generated vocal
                — see this server's instructions for the full finding.
                Warn the user before using it; don't offer it unprompted.
            reference_youtube_url: Same as reference_audio_path, downloaded
                from YouTube first. Mutually exclusive with
                reference_audio_path.
            advanced_settings: Optional {exact UI label: value} overrides
                for any ACE-Step setting. Leave unset fields alone — they
                stay at ACE-Step's own default.
            output_format: "wav" (default), "flac", "mp3", "opus", "aac",
                or "wav32". Defaults to uncompressed wav for DAW editing;
                use mp3 when file size matters more than edit quality.
            remix_source_path: Optional local audio file for ACE-Step's
                Remix mode. Discouraged by default — confirmed to risk
                both vocal detuning and reproducing the source's own
                melody/lyrics (a copyright concern, not just a quality
                one). Only use if the user explicitly asks after being
                warned of both risks — see this server's instructions.
            remix_source_youtube_url: Same as remix_source_path,
                downloaded from YouTube. Mutually exclusive with
                remix_source_path and with the reference_audio_path/
                reference_youtube_url pair.
            remix_strength: 0.0-1.0, default 0.5. How closely Remix mode
                follows the source's structure — confirmed not to
                reliably stop source content bleeding through even at low
                values.
            remix_melody_retention: Optional 0.0-1.0 override for Remix
                mode's "Cover Strength (Melody Retention)". Confirmed not
                to fix garbled vocals — leave at default (None) absent a
                specific, confirmed reason to change it.
            remix_no_fsq: Optional override for Remix mode's "no_fsq".
                Confirmed not to be a fix — trades garbled vocals for a
                near-exact copy of the source instead. Don't present
                either as an improvement.
            song_title: A real title for this song — always provide one
                (write your own if the user hasn't given one). Used to
                name the output file instead of a bare timestamp/UUID.
            lora_path: Optional path to a trained LoRA/LoKr adapter
                directory (e.g. from this server's own training pipeline
                or ACE-Step's Training tab) to load and enable before
                generating. Leave unset for normal generation.
            split_stems: If True, also separate the finished mix into
                vocal-only and instrumental-only stems in this same job,
                returned as vocals_path/instrumental_path from
                check_vocal_track_status. Prefer this over a separate
                split_vocal_stems call whenever you already know you'll
                want the stems.
        """
        validate_caption(caption)
        validate_lyrics(lyrics)
        output_format = validate_output_format(output_format)

        # Mutual-exclusivity checks first, before validating any individual
        # path actually exists - a caller mixing modes should get a clear
        # "these don't combine" error rather than an unrelated
        # FILE_NOT_FOUND for whichever path happened to be checked first.
        if reference_audio_path and reference_youtube_url:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                "provide only one of reference_audio_path or reference_youtube_url, not both",
            )
        if remix_source_path and remix_source_youtube_url:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                "provide only one of remix_source_path or remix_source_youtube_url, not both",
            )
        if (remix_source_path or remix_source_youtube_url) and (reference_audio_path or reference_youtube_url):
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                "remix_source_path/remix_source_youtube_url (Remix mode) cannot be combined with "
                "reference_audio_path/reference_youtube_url (Custom mode) — these are different "
                "ACE-Step generation modes",
            )
        if (remix_source_path or remix_source_youtube_url) and not (0.0 <= remix_strength <= 1.0):
            raise SongForgeMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE, f"remix_strength must be between 0.0 and 1.0, got {remix_strength}"
            )
        if remix_melody_retention is not None and not (0.0 <= remix_melody_retention <= 1.0):
            raise SongForgeMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE,
                f"remix_melody_retention must be between 0.0 and 1.0, got {remix_melody_retention}",
            )

        if reference_audio_path:
            reference_audio_path = validate_audio_file_path(
                reference_audio_path, param_name="reference_audio_path"
            )
        if remix_source_path:
            remix_source_path = validate_audio_file_path(remix_source_path, param_name="remix_source_path")

        job = _jobs.create()

        async def report(fraction: float, message: str) -> None:
            job.progress = fraction
            job.message = message
            # Best-effort: reaches clients that do supply a progressToken.
            # Not relied on for correctness — see check_vocal_track_status.
            try:
                await ctx.report_progress(progress=fraction, total=1.0, message=message)
            except Exception:
                pass

        async def run_job() -> None:
            try:
                resolved_reference_path = reference_audio_path
                if reference_youtube_url:
                    await report(0.0, f"Downloading reference audio from {reference_youtube_url}")
                    resolved_reference_path = await asyncio.to_thread(
                        _youtube_client.download, reference_youtube_url
                    )

                resolved_remix_path = remix_source_path
                if remix_source_youtube_url:
                    await report(0.0, f"Downloading remix source audio from {remix_source_youtube_url}")
                    resolved_remix_path = await asyncio.to_thread(
                        _youtube_client.download, remix_source_youtube_url
                    )

                result = await _client.generate(
                    caption=caption,
                    lyrics=lyrics,
                    reference_audio_path=resolved_reference_path,
                    advanced_settings=advanced_settings,
                    output_format=output_format,
                    remix_source_path=resolved_remix_path,
                    remix_strength=remix_strength,
                    remix_melody_retention=remix_melody_retention,
                    remix_no_fsq=remix_no_fsq,
                    song_title=song_title,
                    lora_path=lora_path,
                    on_progress=report,
                )
                duration = (
                    measure_wav_duration_seconds(result["audio_path"])
                    if result["audio_path"].endswith(".wav")
                    else None
                )
                job.result = {
                    "audio_path": result["audio_path"],
                    "diagnostics": {
                        "generation_seconds": result["generation_seconds"],
                        "duration_seconds": duration,
                        # Confirmed the file actually registered in ACE-Step's
                        # UI before generation started - not just that a path
                        # was passed in. False here (when a reference WAS
                        # requested) would previously have meant the file
                        # silently never loaded and generation ran on the
                        # base model instead, indistinguishable from a real
                        # voice match without this field.
                        "used_reference_audio": result.get("reference_audio_confirmed", False),
                        "used_remix_source": resolved_remix_path is not None,
                        "output_format": output_format,
                    },
                }

                if split_stems:
                    await report(0.99, "Splitting vocal/instrumental stems")
                    stems = await asyncio.to_thread(_separator_client.separate, result["audio_path"])
                    job.result["vocals_path"] = stems["vocals_path"]
                    job.result["instrumental_path"] = stems["instrumental_path"]

                try:
                    play_audio_now(result["audio_path"])
                except Exception:
                    # Best-effort convenience, not the actual deliverable -
                    # the real file path is already in job.result regardless
                    # of whether this succeeds, and inline MCP AudioContent
                    # playback was already tried and removed for causing
                    # worse failures than this (see docs/ARCHITECTURE.md) -
                    # a failure to auto-launch a player must not be allowed
                    # to turn a successful generation into a reported error.
                    pass

                job.progress = 1.0
                job.message = "Generation complete"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}

    @mcp.tool(structured_output=False)
    async def check_vocal_track_status(job_id: str, wait_seconds: float = 25.0) -> list:
        """Poll a job started by any job-based tool on this server
        (generate_vocal_track, generate_vocal_track_takes,
        split_vocal_stems, analyze_reference_audio,
        transcribe_instrumental_to_midi, edit_audio_track,
        prepare_voice_reference) — they all share this one status tool
        and one registry. Waits up to wait_seconds server-side if still
        running (capped at 25s) — call again immediately on "running",
        no extra delay needed.

        Say almost nothing while polling: one message when it finishes or
        errors, optionally one if it's taking a couple of minutes-plus.
        Never narrate individual "still running" polls.

        Args:
            job_id: From whichever of the four tools started the job.
            wait_seconds: Max block time if still running (default 25s).

        Returns a list; first item is a dict with "status" — "running"
        (progress, message), "error" (error), or "complete" (fields
        depend on which tool started the job — see that tool's own
        docs). Every path returned is real and absolute — pass it
        directly to other tools. No inline audio is ever attached.
        """
        job = _jobs.get(job_id)
        if job is None:
            raise SongForgeMCPError(ErrorCode.FILE_NOT_FOUND, f"no such job_id: {job_id}")

        deadline = asyncio.get_event_loop().time() + max(0.0, min(wait_seconds, 25.0))
        while job.status == "running" and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.5)

        if job.status == "running":
            return [{"status": "running", "progress": round(job.progress, 2), "message": job.message}]

        if job.status == "error":
            return [{"status": "error", "error": job.error}]

        return [{"status": "complete", **job.result}]

    @mcp.tool(structured_output=False)
    async def generate_vocal_track_takes(
        caption: str,
        lyrics: str,
        ctx: Context,
        num_takes: int = 3,
        reference_audio_path: str | None = None,
        reference_youtube_url: str | None = None,
        advanced_settings: dict | None = None,
        output_format: str = "wav",
        song_title: str | None = None,
    ) -> dict:
        """Starts generating multiple independent takes of the same song
        (same caption/lyrics, a different random seed each time) so a
        result can be picked from several rather than accepting whatever
        the first one produced. Runs sequentially, not in parallel —
        this server's GPU is a single shared resource, and running takes
        concurrently would recreate the exact contention problem that
        made ACE-Step's own default Batch Size (2 simultaneous takes) a
        real, measured cause of generations silently taking far longer
        (see this server's CHANGELOG). Takes num_takes times as long as
        a single generate_vocal_track call — say so before calling this
        unless the user has already explicitly asked for multiple takes.

        Returns {"job_id": str} immediately; poll
        check_vocal_track_status(job_id) exactly as for
        generate_vocal_track (same tool, same registry).

        Args:
            caption: Same as generate_vocal_track.
            lyrics: Same as generate_vocal_track.
            num_takes: How many independent takes to generate (2-5,
                capped to prevent an accidentally very long request).
            reference_audio_path: Same as generate_vocal_track, applied
                identically to every take.
            reference_youtube_url: Same as generate_vocal_track.
            advanced_settings: Same as generate_vocal_track, applied to
                every take — except "Seed", which this tool always
                overrides per take regardless of what's passed here,
                since varied results are the entire point.
            output_format: Same as generate_vocal_track, applied to
                every take.
            song_title: Base title for every take — each take's actual
                filename gets a take number appended.

        On completion, check_vocal_track_status returns {"takes": [...]}
        — one entry per take, in order: {"audio_path", "seed",
        "diagnostics" (same shape as generate_vocal_track's), "measured":
        {"bpm", "key", "mode", "key_confidence"} from actually analyzing
        that take's output}. A take that individually fails is recorded
        as {"seed", "error"} instead — one bad take does not fail the
        whole job.
        """
        validate_caption(caption)
        validate_lyrics(lyrics)
        output_format = validate_output_format(output_format)
        if not (_MIN_TAKES <= num_takes <= _MAX_TAKES):
            raise SongForgeMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE,
                f"num_takes must be between {_MIN_TAKES} and {_MAX_TAKES}, got {num_takes}",
            )
        if reference_audio_path and reference_youtube_url:
            raise SongForgeMCPError(
                ErrorCode.INVALID_PARAMETER,
                "provide only one of reference_audio_path or reference_youtube_url, not both",
            )
        if reference_audio_path:
            reference_audio_path = validate_audio_file_path(
                reference_audio_path, param_name="reference_audio_path"
            )

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
                resolved_reference_path = reference_audio_path
                if reference_youtube_url:
                    await report(0.0, f"Downloading reference audio from {reference_youtube_url}")
                    resolved_reference_path = await asyncio.to_thread(
                        _youtube_client.download, reference_youtube_url
                    )

                takes = []
                for i in range(num_takes):
                    await report(i / num_takes, f"Generating take {i + 1} of {num_takes}")
                    seed = random.randint(1, 2_147_483_647)
                    per_take_settings = {**(advanced_settings or {}), "Seed": str(seed)}
                    take_title = f"{song_title} (take {i + 1})" if song_title else None
                    try:
                        result = await _client.generate(
                            caption=caption,
                            lyrics=lyrics,
                            reference_audio_path=resolved_reference_path,
                            advanced_settings=per_take_settings,
                            output_format=output_format,
                            remix_source_path=None,
                            remix_strength=0.5,
                            remix_melody_retention=None,
                            remix_no_fsq=False,
                            song_title=take_title,
                            lora_path=None,
                            on_progress=None,
                        )
                        duration = (
                            measure_wav_duration_seconds(result["audio_path"])
                            if result["audio_path"].endswith(".wav")
                            else None
                        )
                        measured = await asyncio.to_thread(analyze_audio, result["audio_path"])
                        takes.append(
                            {
                                "audio_path": result["audio_path"],
                                "seed": seed,
                                "diagnostics": {
                                    "generation_seconds": result["generation_seconds"],
                                    "duration_seconds": duration,
                                    "used_reference_audio": result.get("reference_audio_confirmed", False),
                                    "output_format": output_format,
                                },
                                "measured": measured,
                            }
                        )
                    except SongForgeMCPError as e:
                        takes.append({"seed": seed, "error": f"[{e.code.name}] {e.message}"})
                    except Exception as e:
                        takes.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})

                job.result = {"takes": takes}
                job.progress = 1.0
                job.message = "All takes complete"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}
