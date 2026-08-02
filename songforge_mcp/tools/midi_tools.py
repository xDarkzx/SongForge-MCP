import asyncio
import os

from mcp.server.fastmcp import FastMCP

from songforge_mcp.midi_transcription import get_midi_notes as _get_midi_notes
from songforge_mcp.midi_transcription import transcribe_to_midi
from songforge_mcp.shared_state import jobs as _jobs
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import (
    validate_output_dir_audio_path,
    validate_output_dir_midi_path,
)


def register(mcp: FastMCP):
    @mcp.tool()
    async def transcribe_instrumental_to_midi(audio_path: str) -> dict:
        """Starts converting an instrumental audio file into MIDI via
        basic-pitch, then splitting it into separate bass/melody/chords
        tracks by pitch register and note-overlap density — NOT real
        per-instrument-class separation (can't tell a pad from a
        saw-stack), but genuinely separate, DAW-assignable parts instead
        of one flat blob of every note layered together. No drums/
        percussion captured, pitched content only. Import the three split
        tracks as separate Reaper tracks, not the flat one — that's the
        actual usable starting point for a human to re-orchestrate.

        Returns {"job_id": str} immediately; poll
        check_vocal_track_status(job_id) exactly as for
        generate_vocal_track (same tool, same registry). Usually finishes
        in a few seconds, but duration scales with the input file's
        length with no fixed cap, so this still goes through the
        job/poll pattern rather than blocking. Idempotent — calling it
        again on a file already transcribed returns the existing MIDI
        files instantly instead of re-running the model, so if you've
        lost track of an earlier result it is always cheap and safe to
        call this again rather than assuming nothing exists yet.

        Typical flow: generate_vocal_track → split_vocal_stems for the
        instrumental → this tool.

        Args:
            audio_path: A file this server previously produced (typically
                an instrumental stem from split_vocal_stems). Must be
                inside this server's own output folder.

        On completion, check_vocal_track_status returns midi_path/
        note_count for the flat transcription, plus bass_midi_path,
        melody_midi_path, chords_midi_path (each with its own
        _note_count) for the split tracks.
        """
        resolved = validate_output_dir_audio_path(audio_path, param_name="audio_path")
        midi_path = os.path.splitext(resolved)[0] + ".mid"
        job = _jobs.create()

        async def run_job() -> None:
            try:
                job.result = await asyncio.to_thread(transcribe_to_midi, resolved, midi_path)
                job.progress = 1.0
                job.message = "Transcription complete"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}

    @mcp.tool()
    async def get_midi_notes(midi_path: str, offset: int = 0, max_results: int = 500) -> dict:
        """Returns the actual note data (pitch, start, end, velocity) from
        a MIDI file this server produced — real ground truth, not a
        summary. Use this before recreating, describing, or importing a
        transcribed MIDI file's content anywhere: neither
        transcribe_instrumental_to_midi's own result nor Reaper's own MIDI
        tools expose real note content from an external file, so without
        calling this there is no actual data to work from, only the file
        path and a note count — attempting to "recreate" the notes without
        this produces fabricated content, not the real transcription.

        Paginated (offset/max_results) since a real transcription can have
        hundreds of notes — call again with a higher offset for more.

        Args:
            midi_path: A .mid file this server previously produced
                (flat/bass/melody/chords from transcribe_instrumental_to_midi).
                Must be inside this server's own output folder.
            offset: Index of the first note to return, sorted by start time.
            max_results: Maximum notes to return in one call (default 500,
                hard-capped at 500 server-side regardless of a higher
                value — page with offset for more instead).

        Returns total_note_count, returned_count, offset, and notes (a
        list of {pitch, start, end, velocity}).
        """
        resolved = validate_output_dir_midi_path(midi_path, param_name="midi_path")
        return await asyncio.to_thread(_get_midi_notes, resolved, offset, max_results)
