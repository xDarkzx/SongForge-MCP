"""Multi-instrument MIDI transcription via Omnizart, an alternative to
basic-pitch (see midi_transcription.py). Exists because a real, direct
side-by-side listening comparison on this project's own material found
Omnizart's piano model dramatically more accurate than basic-pitch - not a
theoretical preference, a confirmed result on a real song.

Runs Omnizart in its own isolated venv (SONGFORGE_OMNIZART_PYTHON - see
scripts/install_omnizart.py) via subprocess, matching separator_client.py's
established pattern for exactly the same reason: Omnizart's dependencies
(madmom, numpy<2) conflict with this project's own environment.

CHUNKING IS NOT OPTIONAL POLISH - IT FIXES A CONFIRMED CORRECTNESS BUG:
Omnizart's own note-detection density collapses over the course of a long
audio file. Measured directly on a real 187s song: 5.4 notes/sec in the
first 30 seconds, 0.03 notes/sec (essentially nothing) in the last 30
seconds of that SAME file. Re-running the exact same audio as a short
standalone clip recovered real notes that were missing when it was
processed embedded deep in the long file (48 notes -> 67 notes for one
90-120s window, confirmed by direct A/B test). This is a detection failure,
not a fragmentation/gap problem - midi_transcription.py's merge/close-gap
cleanup cannot recover a note that was never detected in the first place.
Splitting into short chunks before Omnizart ever sees the audio sidesteps
whatever accumulates over a long single inference pass.
"""
import os
import platform
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import pretty_midi
import soundfile as sf

from songforge_mcp.audio_analysis import _PITCH_CLASSES, analyze_audio, detect_key
from songforge_mcp.midi_transcription import (
    _LEGATO_GAP_SECONDS,
    _MERGE_GAP_SECONDS,
    _close_gaps,
    _merge_close_notes,
    _settings_tag,
)
from songforge_mcp_shared.constants import (
    OmnizartTranscription,
    Timeouts,
    ensure_private_dir,
    no_window_popen_kwargs,
)
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_OMNIZART_EXE_NAME = "omnizart.exe" if platform.system() == "Windows" else "omnizart"

# Omnizart loads TensorFlow models onto the same GPU ACE-Step/the separator
# run on - a plain threading.Lock (not asyncio.Lock, since this module is
# driven via asyncio.to_thread rather than natively async) keeps concurrent
# transcription calls from competing for GPU memory, matching
# SeparatorClient._lock's identical reasoning.
_lock = threading.Lock()

# Krumhansl-Schmuckler major/minor diatonic scale degrees, as semitone
# offsets from the root - used to flag notes outside the song's own
# detected key. Standard music theory, not invented for this project.
_MAJOR_SCALE_DEGREES = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE_DEGREES = (0, 2, 3, 5, 7, 8, 10)


def _require_configured() -> tuple[str, str]:
    """Returns (python_exe, omnizart_exe), raising if not configured -
    same shape as SeparatorClient._require_configured."""
    python_exe = os.environ.get("SONGFORGE_OMNIZART_PYTHON", "")
    if not python_exe or not os.path.isfile(python_exe):
        raise SongForgeMCPError(
            ErrorCode.OMNIZART_NOT_CONFIGURED,
            "SONGFORGE_OMNIZART_PYTHON is not set or does not point to a valid "
            "Python interpreter with omnizart installed. Run "
            "scripts/install_omnizart.py, or see docs/INSTALLATION.md.",
        )
    omnizart_exe = os.path.join(os.path.dirname(python_exe), _OMNIZART_EXE_NAME)
    if not os.path.isfile(omnizart_exe):
        raise SongForgeMCPError(
            ErrorCode.OMNIZART_NOT_CONFIGURED,
            f"{_OMNIZART_EXE_NAME} not found alongside {python_exe} - is omnizart "
            "actually installed in that environment?",
        )
    return python_exe, omnizart_exe


def _resolve_model_path(python_exe: str, model_mode: str) -> str:
    """For "Piano" mode, resolves to the tuned checkpoint's full path
    (see OmnizartTranscription.TUNED_PIANO_CHECKPOINT_NAME's comment for
    why it exists and the real measurement behind it) when
    scripts/install_omnizart.py has created one, falling back to the bare
    "Piano" mode name (Omnizart's own stock checkpoint) otherwise - e.g.
    an Omnizart env installed before this tuned checkpoint existed. Any
    other model_mode ("Pop", "Stream", or an already-explicit path) is
    passed straight through unchanged - the tuning was only measured and
    validated for Piano."""
    if model_mode != "Piano":
        return model_mode
    venv_root = os.path.dirname(os.path.dirname(python_exe))
    lib_dir = "Lib" if platform.system() == "Windows" else "lib"
    for root, dirs, _files in os.walk(os.path.join(venv_root, lib_dir)):
        # Requiring the stock "music_piano" checkpoint to also be present
        # (not just any "checkpoints/music" directory) is a much more
        # specific signal that this is actually Omnizart's own checkpoint
        # tree, not an unrelated package that happens to share that
        # directory naming - os.walk's traversal order isn't guaranteed,
        # so a loose match could otherwise silently resolve to the wrong
        # directory and fall back to the untuned checkpoint with no
        # warning. Keep walking (don't break) past a "checkpoints/music"
        # that fails this check, in case it was a false positive and
        # Omnizart's real one is elsewhere in the tree.
        if (
            os.path.basename(root) == "checkpoints"
            and "music" in dirs
            and os.path.isdir(os.path.join(root, "music", "music_piano"))
        ):
            candidate = os.path.join(
                root, "music", OmnizartTranscription.TUNED_PIANO_CHECKPOINT_NAME
            )
            if os.path.isdir(candidate):
                return candidate
            break
    return model_mode


def _group_by_pitch(notes: list) -> dict:
    """Groups notes by MIDI pitch - the same grouping _merge_close_notes
    uses internally, exposed here so _close_gaps (which assumes a single
    pitch/line - see its docstring) can be applied per-pitch instead of
    across a flat, polyphonic note list."""
    by_pitch: dict[int, list] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    return by_pitch


def _chunk_audio(audio_path: str, chunk_seconds: float, out_dir: str) -> list[dict]:
    """Splits audio_path into non-overlapping chunk_seconds-long WAV files
    under out_dir. Returns [{"path": str, "start": float}, ...] in order.

    Non-overlapping by design, not overlap-and-deduplicate: the bug this
    exists to work around (see module docstring) is detection quality
    degrading over a long single inference pass, not content specifically
    getting lost at chunk boundaries - a plain sequential split already
    fixes the confirmed problem without the added complexity (and its own
    failure modes, like double-counting a note that straddles an overlap
    region) of stitching overlapping regions back together."""
    y, sr = sf.read(audio_path)
    duration = len(y) / sr
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_path = os.path.join(out_dir, f"chunk_{idx:03d}.wav")
        sf.write(chunk_path, y[int(start * sr):int(end * sr)], sr)
        chunks.append({"path": chunk_path, "start": start})
        start = end
        idx += 1
    return chunks


def _detect_key_from_notes(notes: list) -> tuple:
    """Builds a duration-weighted, chroma-like pitch-class histogram from
    MIDI notes and runs it through the SAME Krumhansl-Schmuckler
    correlation audio_analysis.detect_key already uses for real audio -
    the identical, established key-finding approach, just applied to
    symbolic note data instead of audio chroma. Returns
    (root, mode, confidence), or (None, None, 0.0) if there are no notes
    to analyze."""
    weights = np.zeros(12)
    for n in notes:
        weights[n.pitch % 12] += n.end - n.start
    if weights.sum() == 0:
        return None, None, 0.0
    root, mode, confidence = detect_key(weights)
    return root, mode, confidence


def _find_off_scale_notes(notes: list, root: str, mode: str) -> list:
    """Returns notes whose pitch class falls outside the detected key's
    diatonic scale - REPORTED ONLY, never auto-removed. Distinguishing a
    real accidental/passing tone from a genuine transcription error
    requires actually listening to it; this project's own testing
    confirmed that judging note correctness from data/statistics alone,
    without listening, is not reliable enough to act on automatically -
    the caller decides what (if anything) to do with this list."""
    if root is None:
        return []
    root_idx = _PITCH_CLASSES.index(root)
    degrees = _MAJOR_SCALE_DEGREES if mode == "major" else _MINOR_SCALE_DEGREES
    in_scale = {(root_idx + d) % 12 for d in degrees}
    return [n for n in notes if n.pitch % 12 not in in_scale]


def transcribe_with_omnizart(
    audio_path: str,
    output_path: str,
    model_mode: str = OmnizartTranscription.DEFAULT_MODEL_MODE,
    chunk_seconds: float = OmnizartTranscription.CHUNK_SECONDS,
    merge_gap_seconds: float = _MERGE_GAP_SECONDS,
    legato_gap_seconds: float = _LEGATO_GAP_SECONDS,
    midi_tempo: float | None = None,
) -> dict:
    """Transcribes audio_path via Omnizart, chunked (see module docstring
    for why this isn't optional), then applies the same non-destructive
    cleanup as basic-pitch's pipeline (merge same-pitch fragments, close
    small gaps) reused directly from midi_transcription.py. Also detects
    the song's key from the resulting notes and reports (does not remove)
    notes outside it - see _find_off_scale_notes.

    Idempotent per (audio file, exact settings) combination, same pattern
    and same reasoning as midi_transcription.transcribe_to_midi: the
    settings themselves (not just "are these the defaults") are baked into
    the output filename via _settings_tag, so changing any argument - or
    a future change to this project's own defaults - always produces a
    fresh, separately-cached file rather than silently reusing output
    generated under different settings.

    Returns: midi_path, note_count, measured_bpm, chunk_count,
    detected_key ("C# minor" style string or None), off_scale_note_count,
    off_scale_notes (list of {pitch, start, end} - capped for response
    size, see the note count for the real total).
    """
    python_exe, omnizart_exe = _require_configured()
    if not os.path.isfile(audio_path):
        raise SongForgeMCPError(
            ErrorCode.FILE_NOT_FOUND, f"audio_path does not exist: {audio_path}"
        )

    # The RESOLVED path, not the bare mode name, goes into the cache key -
    # if a "Piano" run's resolution ever changes (e.g. the tuned checkpoint
    # gets created by a later install_omnizart.py run, or removed), that
    # must count as a different setting, not silently reuse output
    # generated under the other checkpoint. Same reasoning, same fix
    # pattern as _settings_tag's own docstring / the separator's
    # _model_tag - both already fixed the identical class of bug once.
    resolved_model_path = _resolve_model_path(python_exe, model_mode)
    tag = _settings_tag(
        "omnizart", resolved_model_path, chunk_seconds, merge_gap_seconds, legato_gap_seconds, midi_tempo,
    )
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}__{tag}{ext}"

    if os.path.isfile(output_path):
        pm = pretty_midi.PrettyMIDI(output_path)
        notes = [n for inst in pm.instruments for n in inst.notes]
        cached_root, cached_mode, _confidence = _detect_key_from_notes(notes)
        cached_off_scale = _find_off_scale_notes(notes, cached_root, cached_mode)
        tempi = pm.get_tempo_changes()[1]
        return {
            "midi_path": output_path,
            "note_count": len(notes),
            "measured_bpm": round(tempi[0], 1) if len(tempi) else None,
            # Not persisted in the cached file, so unknown on a cache hit -
            # None rather than a misleading guess. Every other field below
            # is cheap to recompute from the notes already on disk, so it
            # stays consistent with a fresh run instead of returning a
            # different, smaller shape just because this was a cache hit.
            "chunk_count": None,
            "detected_key": f"{cached_root} {cached_mode}" if cached_root else None,
            "off_scale_note_count": len(cached_off_scale),
            "off_scale_notes": [
                {"pitch": n.pitch, "start": round(n.start, 3), "end": round(n.end, 3)}
                for n in sorted(cached_off_scale, key=lambda n: n.start)[:50]
            ],
        }

    if midi_tempo is not None:
        measured_bpm = midi_tempo
    else:
        try:
            measured_bpm = analyze_audio(audio_path)["bpm"]
        except SongForgeMCPError:
            measured_bpm = 120.0

    work_dir = tempfile.mkdtemp(prefix="omnizart_chunks_")
    try:
        chunks = _chunk_audio(audio_path, chunk_seconds, work_dir)

        # Keyed by instrument index rather than flattened into one list -
        # non-Piano modes (Pop/Stream) can produce multiple pretty_midi
        # instruments per chunk (e.g. separate bass/melody parts), and
        # Omnizart emits them in a consistent order per chunk, so index
        # is a reasonable cross-chunk identity. Keeping them separate
        # stops _merge_close_notes/_close_gaps below from gluing an
        # unrelated instrument's note onto this one just because they
        # share a pitch and are close in time.
        notes_by_instrument: dict[int, list] = {}
        for chunk in chunks:
            chunk_midi_path = os.path.splitext(chunk["path"])[0] + ".mid"
            with _lock:
                try:
                    result = subprocess.run(
                        [
                            omnizart_exe, "music", "transcribe", chunk["path"],
                            "--model-path", resolved_model_path,
                            "--output", chunk_midi_path,
                        ],
                        capture_output=True, text=True, timeout=Timeouts.OMNIZART_CHUNK, check=False,
                        **no_window_popen_kwargs(),
                    )
                except subprocess.TimeoutExpired as e:
                    raise SongForgeMCPError(
                        ErrorCode.SUBPROCESS_TIMEOUT,
                        f"omnizart exceeded {Timeouts.OMNIZART_CHUNK}s on chunk starting "
                        f"at {chunk['start']:.1f}s",
                    ) from e
            if result.returncode != 0:
                raise SongForgeMCPError(
                    ErrorCode.OMNIZART_TRANSCRIPTION_FAILED,
                    f"omnizart exited {result.returncode} on chunk starting at "
                    f"{chunk['start']:.1f}s: {result.stderr.strip()[-2000:]}",
                )
            if not os.path.isfile(chunk_midi_path):
                raise SongForgeMCPError(
                    ErrorCode.OMNIZART_TRANSCRIPTION_FAILED,
                    f"omnizart reported success but produced no output for chunk "
                    f"starting at {chunk['start']:.1f}s",
                )

            chunk_pm = pretty_midi.PrettyMIDI(chunk_midi_path)
            for inst_idx, inst in enumerate(chunk_pm.instruments):
                for note in inst.notes:
                    note.start += chunk["start"]
                    note.end += chunk["start"]
                    notes_by_instrument.setdefault(inst_idx, []).append(note)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Merge/close-gap independently per instrument, then flatten to the
    # single output track this tool has always produced (matching
    # basic-pitch's own flat-track output - see midi_transcription.py's
    # module docstring) - the per-instrument separation above only exists
    # to keep the two cleanup passes from crossing instrument boundaries.
    merged_notes = []
    for inst_notes in notes_by_instrument.values():
        inst_merged = _merge_close_notes(inst_notes, max_gap_seconds=merge_gap_seconds)
        # _close_gaps assumes a single pitch/line (see its docstring) -
        # Piano (the default model_mode) is inherently polyphonic, so
        # applying it across all pitches at once would extend a note's
        # end into an unrelated note's onset just because they're
        # chronologically adjacent, regardless of pitch. Grouping by
        # pitch first (same grouping _merge_close_notes already uses
        # internally) keeps each call within a single line, closing only
        # genuine same-pitch transcription gaps.
        inst_merged = [
            closed
            for pitch_notes in _group_by_pitch(inst_merged).values()
            for closed in _close_gaps(pitch_notes, max_gap_seconds=legato_gap_seconds)
        ]
        merged_notes.extend(inst_merged)
    merged_notes.sort(key=lambda n: n.start)

    root, mode, _confidence = _detect_key_from_notes(merged_notes)
    off_scale = _find_off_scale_notes(merged_notes, root, mode)

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=measured_bpm)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = merged_notes
    out_pm.instruments.append(instrument)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_private_dir(output_dir)
    out_pm.write(output_path)

    return {
        "midi_path": output_path,
        "note_count": len(merged_notes),
        "measured_bpm": round(measured_bpm, 1),
        "chunk_count": len(chunks),
        "detected_key": f"{root} {mode}" if root else None,
        "off_scale_note_count": len(off_scale),
        "off_scale_notes": [
            {"pitch": n.pitch, "start": round(n.start, 3), "end": round(n.end, 3)}
            for n in sorted(off_scale, key=lambda n: n.start)[:50]
        ],
    }
