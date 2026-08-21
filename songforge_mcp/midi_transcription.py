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
import hashlib
import heapq
import os

import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from songforge_mcp.audio_analysis import analyze_audio
from songforge_mcp_shared.constants import MAX_MIDI_NOTES_PER_PAGE
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

# basic-pitch's own defaults (onset_threshold=0.5, frame_threshold=0.3,
# minimum_note_length=127.7ms) are tuned for clean, isolated audio, and
# on real transcriptions with heavy fragmentation, raising them all
# significantly (0.6/0.4/180ms) was tried first. Real, confirmed
# regression from that: onset_threshold/frame_threshold/
# minimum_note_length are enforced INSIDE basic-pitch itself, before any
# of this module's own code ever sees the output - a note filtered out
# there is gone permanently, not recoverable by _merge_close_notes or
# anything downstream. Direct evidence on real transcriptions: minimum
# note duration was clustering right at the 180ms floor across every
# file checked, and a real listening test confirmed roughly half the
# actual musical content was simply missing, not just fragmented.
# Trading "too many spurious notes" for "half the real notes silently
# deleted" is not an improvement - it's a worse failure mode with less
# visible symptoms (a fragmented note is still audible/present to fix;
# a deleted note just isn't there).
#
# Now only mildly above basic-pitch's own defaults (kept the direction,
# cut the magnitude) - minimum_note_length in particular is set BELOW
# basic-pitch's own default, not above it, specifically so a fast, real,
# short note is never discarded at the model level. Fragmentation
# cleanup is now the job of the non-destructive functions below
# (_merge_close_notes, _close_gaps), which only ever consolidate or
# extend notes that already exist - they can't lose real content the
# way a stricter model-level filter can.
_ONSET_THRESHOLD = 0.55
_FRAME_THRESHOLD = 0.32
_MINIMUM_NOTE_LENGTH_MS = 90.0
_MERGE_GAP_SECONDS = 0.06

# For the bass/melody split (not chords - see _close_gaps docstring): how
# large a silence between two consecutive notes is still "transcription
# noise" versus a real, intentional rest. 350ms is short next to a beat
# at most song tempos but comfortably longer than the frame-flicker gaps
# _merge_close_notes targets - it closes the leftover slivers a bassline
# or piano line shouldn't have without eating real phrasing.
_LEGATO_GAP_SECONDS = 0.35


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


def _merge_close_notes(notes: list, max_gap_seconds: float = _MERGE_GAP_SECONDS) -> list:
    """Merges consecutive same-pitch notes separated by a gap no longer
    than max_gap_seconds into one sustained note.

    This is the direct fix for "one sustained note came back as 5-8 tiny
    fragments a few hundred ms apart" - confirmed on real transcriptions,
    including a solo isolated piano stem (not a mixing/bleed problem,
    basic-pitch's frame confidence itself flickers mid-note). Grouped by
    pitch first so it never merges two genuinely different notes that
    happen to be close together - only true repeats/fragments of the
    same pitch. The gap is kept short (60ms default) on purpose so a
    real fast staccato repeat of the same note isn't wrongly glued into
    one long note."""
    by_pitch: dict[int, list] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    merged: list = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: n.start)
        current = pitch_notes[0]
        for nxt in pitch_notes[1:]:
            if nxt.start - current.end <= max_gap_seconds:
                current.end = max(current.end, nxt.end)
                current.velocity = max(current.velocity, nxt.velocity)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    merged.sort(key=lambda n: n.start)
    return merged


def _close_gaps(notes: list, max_gap_seconds: float = _LEGATO_GAP_SECONDS) -> list:
    """Extends each note's end to the following note's start when the gap
    between them is small - for monophonic-ish parts (bass, melody)
    where a brief silence is transcription noise rather than a real
    rest. A bassline or piano line with a genuine held-off pickup note
    or a breath between phrases still needs real rests, so this only
    closes gaps up to max_gap_seconds; anything longer is left alone
    rather than gluing the whole part into one unbroken drone.
    Deliberately not applied to chords - rhythmic chord stabs commonly
    have real, intentional space between them.

    Notes are assumed already non-overlapping for a single pitch/line
    (true after _merge_close_notes); overlapping or touching notes are
    left as-is since there's no gap to close."""
    ordered = sorted(notes, key=lambda n: n.start)
    for current, nxt in zip(ordered, ordered[1:]):
        gap = nxt.start - current.end
        if 0 < gap <= max_gap_seconds:
            current.end = nxt.start
    return ordered


def split_midi_by_register(
    midi_path: str,
    bass_pitch_threshold: int = 48,
    chord_polyphony_threshold: int = 3,
    legato_gap_seconds: float = _LEGATO_GAP_SECONDS,
) -> dict:
    """Heuristically splits a flat, single-track MIDI transcription into
    three separate tracks: bass (pitch below bass_pitch_threshold - MIDI
    note 48 is C3), chords (chord_polyphony_threshold+ notes sounding at
    once), and melody (everything else). See module docstring for why
    this is a heuristic, not real instrument classification.

    legato_gap_seconds is how large a silence between two consecutive
    bass/melody notes still counts as transcription noise to be closed
    rather than a real rest - see _close_gaps.

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

    # Bass and melody are the parts a bassline/piano-lead workflow actually
    # plays legato - close their small gaps. Chords are left untouched,
    # see _close_gaps docstring.
    bass_notes = _close_gaps(bass_notes, max_gap_seconds=legato_gap_seconds)
    melody_notes = _close_gaps(melody_notes, max_gap_seconds=legato_gap_seconds)

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


def _settings_tag(*settings) -> str:
    """Short, deterministic tag identifying exactly which transcription
    settings produced a given output file - folded into the output
    filename in transcribe_to_midi so a cached file from a different
    settings combination is never mistaken for the current
    configuration's output.

    Critically, this also covers a real, confirmed failure mode that a
    simple "are these still the defaults" check does not: a .mid file
    transcribed under this module's OWN OLDER defaults (before a tuning
    pass changed _ONSET_THRESHOLD/_MINIMUM_NOTE_LENGTH_MS/etc., or before
    BPM measurement/fragment-merging existed at all) looks byte-for-byte
    identical to a plain default-settings call today - same audio_path,
    same "just use the defaults" request. Without the actual default
    VALUES baked into the cache key, every future improvement to this
    module's defaults would keep silently serving pre-improvement output
    to any song already transcribed once, no matter how much better the
    code gets - exactly the same class of bug confirmed and fixed in
    separator_client.py's _model_tag around the same time."""
    raw = "|".join(str(s) for s in settings)
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def _split_paths_for(output_path: str) -> dict:
    base = os.path.splitext(output_path)[0]
    return {
        "bass": f"{base}_bass.mid",
        "melody": f"{base}_melody.mid",
        "chords": f"{base}_chords.mid",
    }


def transcribe_to_midi(
    audio_path: str,
    output_path: str,
    onset_threshold: float = _ONSET_THRESHOLD,
    frame_threshold: float = _FRAME_THRESHOLD,
    minimum_note_length_ms: float = _MINIMUM_NOTE_LENGTH_MS,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    melodia_trick: bool = True,
    merge_gap_seconds: float = _MERGE_GAP_SECONDS,
    legato_gap_seconds: float = _LEGATO_GAP_SECONDS,
    bass_pitch_threshold: int = 48,
    chord_polyphony_threshold: int = 3,
    midi_tempo: float | None = None,
) -> dict:
    """Runs basic-pitch on `audio_path`, merges frame-flicker fragments,
    writes the cleaned transcription to `output_path`, then splits it via
    split_midi_by_register (which itself legato-closes small gaps in the
    bass/melody parts). Returns midi_path, note_count, measured_bpm, plus
    bass/melody/chords paths and counts.

    All keyword args are basic-pitch/cleanup knobs with defaults already
    tuned for "clean, neat MIDI, not thousands of scattered notes" - most
    callers should never need to touch them. They exist for the cases
    the defaults get wrong for a specific instrument/song:
      onset_threshold / frame_threshold (0-1, higher = stricter): raise
        further on a noisy/dense mix that's still producing spurious
        notes; lower on quiet or very sparse audio that's dropping real
        notes entirely.
      minimum_note_length_ms: raise for a mostly-sustained part (pads,
        held chords) to suppress more flicker; lower for a genuinely fast
        part (16th-note bass runs, tremolo) that's having short real
        notes discarded.
      minimum_frequency / maximum_frequency (Hz, None = full range):
        constrain to a known instrument's real range (e.g. ~41-400Hz for
        a bass guitar) to cut octave-jump errors and stray content from
        other frequency ranges bleeding into the transcription.
      melodia_trick: basic-pitch's own monophonic-melody continuity aid.
        Leave on for a single-line instrument; turning it off can help a
        dense polyphonic passage that's coming out over-connected.
      merge_gap_seconds: max gap between same-pitch fragments still
        merged into one note - see _merge_close_notes.
      legato_gap_seconds: max gap between different-pitch bass/melody
        notes still closed into legato - see _close_gaps. Raise for a
        part with genuine short rests being wrongly glued together;
        lower (or 0) for a part that should stay staccato.
      bass_pitch_threshold / chord_polyphony_threshold: passed straight
        to split_midi_by_register - see its docstring.
      midi_tempo: explicit BPM override. By default the audio's real
        tempo is measured (audio_analysis.analyze_audio, librosa beat
        tracking) rather than trusting basic-pitch's own hardcoded 120
        default - but beat trackers are known to sometimes report a
        tempo's octave wrong (half/double the real BPM, confirmed
        elsewhere in this codebase - see beat_alignment.py), so pass the
        song's actual BPM directly here if you already know it or the
        measured value looks wrong.

    Idempotent per (audio file, exact settings) combination: if the
    settings-tagged output_path (see _settings_tag) and its bass/melody/
    chords siblings already exist, returns their info directly (reading
    note counts back from the existing files - fast, no model inference)
    instead of re-running basic-pitch. A real, confirmed failure mode is
    the calling model losing track of an earlier result over a long
    conversation and asking to transcribe the same file again - there is
    no reason to re-run the model and regenerate different, possibly
    diverging output for a file already transcribed with the same
    settings. Because the settings themselves (not just "are these the
    defaults") are baked into the cache key, changing ANY argument - or
    this module's own default values changing in a future update -
    always produces a different, fresh output file rather than silently
    reusing something transcribed under different settings."""
    tag = _settings_tag(
        onset_threshold, frame_threshold, minimum_note_length_ms,
        minimum_frequency, maximum_frequency, melodia_trick,
        merge_gap_seconds, legato_gap_seconds,
        bass_pitch_threshold, chord_polyphony_threshold, midi_tempo,
    )
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}__{tag}{ext}"

    split_paths = _split_paths_for(output_path)
    if os.path.isfile(output_path) and all(os.path.isfile(p) for p in split_paths.values()):
        cached_tempi = pretty_midi.PrettyMIDI(output_path).get_tempo_changes()[1]
        result = {
            "midi_path": output_path,
            "note_count": _count_midi_notes(output_path),
            "measured_bpm": round(cached_tempi[0], 1) if len(cached_tempi) else None,
        }
        for name, path in split_paths.items():
            result[f"{name}_midi_path"] = path
            result[f"{name}_note_count"] = _count_midi_notes(path)
        return result

    if midi_tempo is not None:
        measured_bpm = midi_tempo
    else:
        try:
            measured_bpm = analyze_audio(audio_path)["bpm"]
        except SongForgeMCPError:
            measured_bpm = 120.0

    try:
        _model_output, midi_data, _note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length_ms,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=melodia_trick,
            midi_tempo=measured_bpm,
        )
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.SYNTHESIS_FAILED, f"MIDI transcription failed for {audio_path!r}: {e}"
        ) from e

    note_count = 0
    for instrument in midi_data.instruments:
        instrument.notes = _merge_close_notes(instrument.notes, max_gap_seconds=merge_gap_seconds)
        note_count += len(instrument.notes)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    midi_data.write(output_path)

    split_result = split_midi_by_register(
        output_path,
        bass_pitch_threshold=bass_pitch_threshold,
        chord_polyphony_threshold=chord_polyphony_threshold,
        legato_gap_seconds=legato_gap_seconds,
    )

    return {
        "midi_path": output_path,
        "note_count": note_count,
        "measured_bpm": round(measured_bpm, 1),
        **split_result,
    }
