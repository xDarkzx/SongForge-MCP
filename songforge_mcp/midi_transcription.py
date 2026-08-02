"""Audio-to-MIDI transcription for a generated instrumental, via Spotify's
basic-pitch (a real, established polyphonic transcription model - not
something built from scratch here), followed by a heuristic split into
separate bass/melody/chords tracks.

basic-pitch itself produces one flat, single-track transcription - it is
NOT per-instrument-class separation (pads vs bass vs saw-stack, etc -
reliably telling those apart from mixed audio is a genuinely hard,
unsolved problem). Importing that flat track as-is gives a human nothing
to actually re-orchestrate around - a single track of every note in the
mix layered together is not a usable starting point. The split below is
a real, different technique: pitch register (bass = low notes) and
onset polyphony (chords = several notes sounding at once) heuristics,
the same general approach used by several real auto-arrangement tools
for exactly this problem. It is still not real instrument
classification and will misclassify real basslines that go high or
leads with doubled notes - but three separately assignable tracks are a
meaningfully more useful DAW starting point than one flat blob.
"""
import heapq
import os

import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from songforge_mcp_shared.constants import MAX_MIDI_NOTES_PER_PAGE
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError


def _onset_polyphony(notes: list) -> dict:
    """Maps id(note) -> how many notes (including itself) are sounding at
    the moment this note begins, via a sweep-line over sorted onsets.

    Notes sharing the exact same start time are processed as one batch -
    pushing them onto the active-notes heap one at a time before reading
    any of their counts would make the first note in the batch miss the
    others that start alongside it (confirmed by a real failing test:
    two notes starting together each need to see the other)."""
    polyphony = {}
    active_ends: list[float] = []
    sorted_notes = sorted(notes, key=lambda n: n.start)
    i = 0
    total = len(sorted_notes)
    while i < total:
        current_start = sorted_notes[i].start
        while active_ends and active_ends[0] <= current_start:
            heapq.heappop(active_ends)

        batch = []
        while i < total and sorted_notes[i].start == current_start:
            batch.append(sorted_notes[i])
            i += 1
        for note in batch:
            heapq.heappush(active_ends, note.end)

        count = len(active_ends)
        for note in batch:
            polyphony[id(note)] = count
    return polyphony


def split_midi_by_register(
    midi_path: str,
    bass_pitch_threshold: int = 48,
    chord_polyphony_threshold: int = 3,
) -> dict:
    """Heuristically splits a flat, single-track MIDI transcription into
    three separate tracks: bass (pitch below bass_pitch_threshold - MIDI
    note 48 is C3), chords (chord_polyphony_threshold+ notes sounding at
    once), and melody (everything else). See module docstring for why
    this is a heuristic, not real instrument classification.

    Returns bass_midi_path, melody_midi_path, chords_midi_path (written
    next to `midi_path`) and their respective note counts.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = [note for inst in pm.instruments for note in inst.notes]

    bass_notes = [n for n in all_notes if n.pitch < bass_pitch_threshold]
    upper_notes = [n for n in all_notes if n.pitch >= bass_pitch_threshold]

    polyphony = _onset_polyphony(upper_notes)
    chord_notes = [n for n in upper_notes if polyphony[id(n)] >= chord_polyphony_threshold]
    melody_notes = [n for n in upper_notes if polyphony[id(n)] < chord_polyphony_threshold]

    base = os.path.splitext(midi_path)[0]
    result: dict = {}
    for name, notes, suffix in (
        ("bass", bass_notes, "_bass.mid"),
        ("melody", melody_notes, "_melody.mid"),
        ("chords", chord_notes, "_chords.mid"),
    ):
        out_pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, name=name.title())
        instrument.notes = notes
        out_pm.instruments.append(instrument)
        out_path = f"{base}{suffix}"
        out_pm.write(out_path)
        result[f"{name}_midi_path"] = out_path
        result[f"{name}_note_count"] = len(notes)

    return result


def _count_midi_notes(path: str) -> int:
    pm = pretty_midi.PrettyMIDI(path)
    return sum(len(instrument.notes) for instrument in pm.instruments)


def get_midi_notes(path: str, offset: int = 0, max_results: int = 500) -> dict:
    """Returns the actual note data (pitch/start/end/velocity) from a MIDI
    file this server produced, sorted by start time. Exists because
    nothing else on this server (or in reaper-mcp) exposes real note
    content to the calling model - transcribe_instrumental_to_midi only
    ever returned a file path and a note count, never the notes
    themselves, which meant a calling model asked to "use" or "recreate"
    a transcribed MIDI file had no real data to work from and had no
    option but to fabricate plausible-sounding notes instead. This is
    that missing ground truth.

    Paginated via offset/max_results since a real transcription can have
    hundreds of notes (e.g. this project's own testing measured 510-821
    notes in a single split track) - returning all of them unconditionally
    in one call could be a very large response for no benefit if the
    caller only needs to inspect part of it. max_results is hard-capped
    at MAX_MIDI_NOTES_PER_PAGE server-side regardless of what's passed in
    - a caller-supplied value alone was not actually enforcing the cap
    this docstring claims to have."""
    pm = pretty_midi.PrettyMIDI(path)
    all_notes = sorted(
        (note for instrument in pm.instruments for note in instrument.notes),
        key=lambda n: n.start,
    )
    max_results = min(max_results, MAX_MIDI_NOTES_PER_PAGE)
    page = all_notes[offset : offset + max_results]
    return {
        "total_note_count": len(all_notes),
        "returned_count": len(page),
        "offset": offset,
        "notes": [
            {
                "pitch": note.pitch,
                "start": round(note.start, 3),
                "end": round(note.end, 3),
                "velocity": note.velocity,
            }
            for note in page
        ],
    }


def _split_paths_for(output_path: str) -> dict:
    base = os.path.splitext(output_path)[0]
    return {
        "bass": f"{base}_bass.mid",
        "melody": f"{base}_melody.mid",
        "chords": f"{base}_chords.mid",
    }


def transcribe_to_midi(audio_path: str, output_path: str) -> dict:
    """Runs basic-pitch on `audio_path`, writes the flat transcription to
    `output_path`, then splits it via split_midi_by_register. Returns
    midi_path, note_count, plus bass/melody/chords paths and counts.

    Idempotent: if `output_path` and its bass/melody/chords siblings
    already exist, returns their info directly (reading note counts back
    from the existing files - fast, no model inference) instead of
    re-running basic-pitch. A real, confirmed failure mode is the calling
    model losing track of an earlier result over a long conversation and
    asking to transcribe the same file again - there is no reason to
    re-run the model and regenerate different, possibly diverging output
    for a file already transcribed."""
    split_paths = _split_paths_for(output_path)
    if os.path.isfile(output_path) and all(os.path.isfile(p) for p in split_paths.values()):
        result = {"midi_path": output_path, "note_count": _count_midi_notes(output_path)}
        for name, path in split_paths.items():
            result[f"{name}_midi_path"] = path
            result[f"{name}_note_count"] = _count_midi_notes(path)
        return result

    try:
        _model_output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.SYNTHESIS_FAILED, f"MIDI transcription failed for {audio_path!r}: {e}"
        ) from e

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    midi_data.write(output_path)

    split_result = split_midi_by_register(output_path)

    return {
        "midi_path": output_path,
        "note_count": len(note_events),
        **split_result,
    }
