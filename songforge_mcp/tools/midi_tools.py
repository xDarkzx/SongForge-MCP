import asyncio
import os

from mcp.server.fastmcp import FastMCP

from songforge_mcp.midi_transcription import _LEGATO_GAP_SECONDS as _DEFAULT_LEGATO_GAP_SECONDS
from songforge_mcp.midi_transcription import _MERGE_GAP_SECONDS as _DEFAULT_MERGE_GAP_SECONDS
from songforge_mcp.midi_transcription import _MINIMUM_NOTE_LENGTH_MS as _DEFAULT_MIN_NOTE_LENGTH_MS
from songforge_mcp.midi_transcription import _ONSET_THRESHOLD as _DEFAULT_ONSET_THRESHOLD
from songforge_mcp.midi_transcription import _FRAME_THRESHOLD as _DEFAULT_FRAME_THRESHOLD
from songforge_mcp.midi_transcription import get_midi_notes as _get_midi_notes
from songforge_mcp.midi_transcription import transcribe_to_midi
from songforge_mcp.omnizart_client import transcribe_with_omnizart
from songforge_mcp.shared_state import jobs as _jobs
from songforge_mcp_shared.constants import OmnizartTranscription
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import (
    validate_output_dir_audio_path,
    validate_output_dir_midi_path,
)


def register(mcp: FastMCP):
    @mcp.tool()
    async def transcribe_instrumental_to_midi(
        audio_path: str,
        onset_threshold: float = _DEFAULT_ONSET_THRESHOLD,
        frame_threshold: float = _DEFAULT_FRAME_THRESHOLD,
        minimum_note_length_ms: float = _DEFAULT_MIN_NOTE_LENGTH_MS,
        minimum_frequency: float | None = None,
        maximum_frequency: float | None = None,
        melodia_trick: bool = True,
        merge_gap_seconds: float = _DEFAULT_MERGE_GAP_SECONDS,
        legato_gap_seconds: float = _DEFAULT_LEGATO_GAP_SECONDS,
        bass_pitch_threshold: int = 48,
        chord_polyphony_threshold: int = 3,
        midi_tempo: float | None = None,
    ) -> dict:
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
        job/poll pattern rather than blocking. Idempotent per (audio_path,
        exact settings) combination — calling it again with the same
        audio_path and the same argument values (all defaults, or the
        same overrides) returns the existing MIDI files instantly instead
        of re-running the model, so if you've lost track of an earlier
        result it is always cheap and safe to call this again rather than
        assuming nothing exists yet. Changing ANY argument always re-runs
        and produces a NEW, separately-cached output file rather than
        overwriting or reusing the previous one — this also means a past
        result from before this tool's own defaults were last tuned is
        never mistaken for a fresh one; each distinct settings
        combination gets its own file forever.

        Typical flow: generate_vocal_track → split_vocal_stems for the
        instrumental → this tool. If the first pass comes back with too
        many scattered/short notes or off-grid timing, re-call with
        adjusted settings rather than accepting it — see the args below.

        IMPORTANT — onset_threshold/frame_threshold/minimum_note_length_ms
        are enforced INSIDE basic-pitch itself: a note filtered out there
        is gone permanently, before this tool's own cleanup logic
        (merge_gap_seconds/legato_gap_seconds) ever sees it. Raising these
        to fight scattered/fragmented output is a real, confirmed trap —
        it silently deletes real notes rather than cleaning them up (a
        prior version of these defaults did exactly this: raising
        minimum_note_length_ms to 180 was confirmed, via direct inspection
        of real output plus a real listening test, to have quietly
        discarded roughly half the actual musical content, not merely
        fragmented it). If output still looks too scattered after a
        default-settings call, prefer raising merge_gap_seconds/
        legato_gap_seconds first — they only ever consolidate/extend notes
        that already exist, they can't lose real content the way these
        three can.

        Args:
            audio_path: A file this server previously produced (typically
                an instrumental stem from split_vocal_stems). Must be
                inside this server's own output folder.
            onset_threshold: 0-1, higher = stricter about what counts as
                a new note. Only raise a little (e.g. 0.55-0.6) even on a
                noisy/dense mix — see the warning above; lower (e.g.
                0.4-0.5, basic-pitch's own default) if real notes are
                being dropped from a quiet or sparse recording, or from a
                smooth/legato part with weak note-attack transients
                (common in synth/EDM bass — a stricter onset threshold
                specifically hurts detecting pitch changes within a
                sustained glide).
            frame_threshold: 0-1, higher = stricter about sustaining a
                note frame-to-frame. Same raise/lower guidance as
                onset_threshold — small moves only.
            minimum_note_length_ms: shortest note basic-pitch will keep.
                Default (90ms) is already BELOW basic-pitch's own default
                (127.7ms), deliberately, so a fast real note (16th-note
                bass runs, tremolo, ornamental piano notes) is never
                discarded at the model level — let merge_gap_seconds
                consolidate any resulting fragments instead. Raise this
                only for a part that's unambiguously all long, sustained
                notes (slow pads, whole-note chords) and still coming out
                fragmented after trying merge_gap_seconds first.
            minimum_frequency / maximum_frequency: Hz, None = full range.
                Constrain to a known instrument's real range (e.g.
                ~41-400 Hz for a bass guitar, ~65-1050 Hz for a cello) to
                cut octave-jump errors and stray content bleeding in from
                other frequency ranges.
            melodia_trick: basic-pitch's own monophonic-melody continuity
                aid. Leave True for a single-line instrument; try False
                if a dense polyphonic passage is coming out over-connected.
            merge_gap_seconds: max gap between same-pitch fragments still
                merged into one sustained note. Default 0.06 (60ms)
                targets frame-flicker, not real repeated notes.
            legato_gap_seconds: max gap between different-pitch bass/
                melody notes still closed into legato (not applied to
                chords). Default 0.35 (350ms). Raise if a part with real
                short rests is being wrongly glued together; lower
                (or 0) if a part should stay staccato/detached instead of
                legato.
            bass_pitch_threshold: MIDI note number — pitches below this
                go to the bass track. Default 48 (C3).
            chord_polyphony_threshold: notes sounding at once at/above
                this count go to the chords track instead of melody.
                Default 3.
            midi_tempo: explicit BPM override. By default the audio's
                real tempo is measured (librosa beat tracking) instead of
                trusting basic-pitch's own hardcoded 120 — but beat
                trackers can report a tempo's octave wrong (half/double
                the real BPM). Pass the song's actual BPM here directly
                if you already know it or the measured value looks wrong
                (e.g. notes visibly off the beat grid after import).

        On completion, check_vocal_track_status returns midi_path/
        note_count/measured_bpm for the flat transcription, plus
        bass_midi_path, melody_midi_path, chords_midi_path (each with its
        own _note_count) for the split tracks. measured_bpm is the
        audio's real, detected tempo (or your midi_tempo override) — the
        MIDI file's embedded tempo is set to it, so importing at the
        song's actual BPM keeps notes aligned to the beat grid.
        """
        resolved = validate_output_dir_audio_path(audio_path, param_name="audio_path")
        midi_path = os.path.splitext(resolved)[0] + ".mid"
        job = _jobs.create()

        async def run_job() -> None:
            try:
                job.result = await asyncio.to_thread(
                    transcribe_to_midi,
                    resolved,
                    midi_path,
                    onset_threshold,
                    frame_threshold,
                    minimum_note_length_ms,
                    minimum_frequency,
                    maximum_frequency,
                    melodia_trick,
                    merge_gap_seconds,
                    legato_gap_seconds,
                    bass_pitch_threshold,
                    chord_polyphony_threshold,
                    midi_tempo,
                )
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
    async def transcribe_instrumental_to_midi_omnizart(
        audio_path: str,
        model_mode: str = OmnizartTranscription.DEFAULT_MODEL_MODE,
        chunk_seconds: float = OmnizartTranscription.CHUNK_SECONDS,
        merge_gap_seconds: float = _DEFAULT_MERGE_GAP_SECONDS,
        legato_gap_seconds: float = _DEFAULT_LEGATO_GAP_SECONDS,
        midi_tempo: float | None = None,
    ) -> dict:
        """Starts converting an instrumental audio file into MIDI via
        Omnizart instead of basic-pitch (see transcribe_instrumental_to_midi)
        — a genuinely different transcription engine, not a re-tuned
        version of the same one. Confirmed via direct side-by-side
        listening on real material in this project's own testing to be
        dramatically more accurate than basic-pitch on at least one real
        song's piano part. Requires SONGFORGE_OMNIZART_PYTHON to be
        configured (run scripts/install_omnizart.py, or see
        docs/INSTALLATION.md) — this is an optional, separately-installed
        engine, not bundled by default.

        Runs Omnizart on short chunks of the audio rather than the whole
        file at once. This is NOT a performance optimization — it works
        around a real, confirmed bug: Omnizart's own note-detection
        density collapses over the course of a long audio file (measured
        directly: 5.4 notes/sec in the first 30s of a real song, 0.03
        notes/sec in the last 30s of that SAME file; re-running that
        section standalone recovered the missing notes). No post-processing
        can fix that after the fact — chunking before transcription is the
        actual fix.

        Applies the same non-destructive cleanup as
        transcribe_instrumental_to_midi (merges same-pitch fragments,
        closes small gaps) — no per-instrument bass/melody/chords split
        (Omnizart's own output doesn't have the register categories that
        split relies on). Also detects the song's key from the transcribed
        notes and reports (does NOT remove) notes outside it, since
        deciding whether an unusual note is a real accidental or a
        transcription error requires actually listening to it — that
        judgment belongs to whoever's reviewing the result, not this tool.

        Returns {"job_id": str} immediately; poll
        check_vocal_track_status(job_id) exactly as for
        generate_vocal_track. Idempotent per (audio_path, exact settings)
        combination, same reasoning and same guarantee as
        transcribe_instrumental_to_midi — changing any argument always
        produces a fresh, separately-cached file.

        Args:
            audio_path: A file this server previously produced (typically
                an instrumental stem from split_vocal_stems). Must be
                inside this server's own output folder.
            model_mode: Omnizart's own transcription-mode name - "Piano"
                (dedicated piano checkpoint, sharper for that one
                instrument specifically), "Pop" (general/mixed pop
                instrumentation), or "Stream" (note-stream variant).
            chunk_seconds: Length of each independently-transcribed chunk.
                Default 30 is where the degradation-recovery was actually
                confirmed, not exhaustively tuned - try a different value
                if this default doesn't work well for other material.
            merge_gap_seconds: max gap between same-pitch fragments still
                merged into one sustained note. Same meaning as
                transcribe_instrumental_to_midi's identically-named arg.
            legato_gap_seconds: max gap between consecutive notes still
                closed into legato. Same meaning as
                transcribe_instrumental_to_midi's identically-named arg.
            midi_tempo: explicit BPM override. By default the audio's
                tempo is measured the same way as
                transcribe_instrumental_to_midi (see that tool's
                docstring for the known fast-tempo measurement risk) -
                pass the real BPM directly if you already know it.

        On completion, check_vocal_track_status returns midi_path,
        note_count, measured_bpm, chunk_count, detected_key (e.g.
        "C# minor", or null if no notes), off_scale_note_count, and
        off_scale_notes (a capped preview list of {pitch, start, end} for
        notes outside the detected key - not a claim they're wrong, just
        worth listening to).
        """
        resolved = validate_output_dir_audio_path(audio_path, param_name="audio_path")
        midi_path = os.path.splitext(resolved)[0] + "_omnizart.mid"
        job = _jobs.create()

        async def run_job() -> None:
            try:
                job.result = await asyncio.to_thread(
                    transcribe_with_omnizart,
                    resolved,
                    midi_path,
                    model_mode,
                    chunk_seconds,
                    merge_gap_seconds,
                    legato_gap_seconds,
                    midi_tempo,
                )
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
