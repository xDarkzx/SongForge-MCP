import os
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from songforge_mcp.midi_transcription import (
    _FRAME_THRESHOLD,
    _LEGATO_GAP_SECONDS,
    _MERGE_GAP_SECONDS,
    _MINIMUM_NOTE_LENGTH_MS,
    _ONSET_THRESHOLD,
    _close_gaps,
    _merge_close_notes,
    _onset_polyphony,
    _settings_tag,
    get_midi_notes,
    split_midi_by_register,
    transcribe_to_midi,
)
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_FAKE_BPM = {"bpm": 140.0, "key": "C", "mode": "major", "key_confidence": 0.9, "duration_seconds": 30.0}


def _tagged(output_path, onset_threshold=_ONSET_THRESHOLD, frame_threshold=_FRAME_THRESHOLD,
            minimum_note_length_ms=_MINIMUM_NOTE_LENGTH_MS,
            minimum_frequency=None, maximum_frequency=None, melodia_trick=True,
            merge_gap_seconds=_MERGE_GAP_SECONDS, legato_gap_seconds=_LEGATO_GAP_SECONDS,
            bass_pitch_threshold=48, chord_polyphony_threshold=3, midi_tempo=None):
    """Computes the same settings-tagged path transcribe_to_midi derives
    internally, so tests assert against real behavior instead of
    duplicating/hardcoding a tag format that could silently drift out of
    sync with the actual implementation."""
    tag = _settings_tag(
        onset_threshold, frame_threshold, minimum_note_length_ms,
        minimum_frequency, maximum_frequency, melodia_trick,
        merge_gap_seconds, legato_gap_seconds,
        bass_pitch_threshold, chord_polyphony_threshold, midi_tempo,
    )
    base, ext = os.path.splitext(str(output_path))
    return f"{base}__{tag}{ext}"


def _fake_midi_data(notes=None):
    midi = MagicMock()
    midi.write = MagicMock()
    instrument = MagicMock()
    instrument.notes = notes if notes is not None else []
    midi.instruments = [instrument]
    return midi


_FAKE_SPLIT_RESULT = {
    "bass_midi_path": "/fake/out_bass.mid", "bass_note_count": 0,
    "melody_midi_path": "/fake/out_melody.mid", "melody_note_count": 0,
    "chords_midi_path": "/fake/out_chords.mid", "chords_note_count": 0,
}


def test_transcribe_to_midi_writes_file_and_returns_note_count(tmp_path):
    fake_notes = [_note(60, 0.0, 1.0), _note(64, 1.0, 2.0), _note(67, 2.0, 3.0)]
    fake_midi = _fake_midi_data(fake_notes)

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, fake_notes)), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        output_path = str(tmp_path / "out.mid")
        result = transcribe_to_midi("input.wav", output_path)

    expected_path = _tagged(output_path)
    fake_midi.write.assert_called_once_with(expected_path)
    assert result["midi_path"] == expected_path
    assert result["note_count"] == 3
    assert result["measured_bpm"] == 140.0
    assert result["bass_midi_path"] == "/fake/out_bass.mid"


def test_transcribe_to_midi_passes_measured_bpm_and_tuned_thresholds_to_predict(tmp_path):
    fake_midi = _fake_midi_data([])

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])) as mock_predict, \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        transcribe_to_midi("input.wav", str(tmp_path / "out.mid"))

    _args, kwargs = mock_predict.call_args
    assert kwargs["midi_tempo"] == 140.0
    # Onset/frame thresholds only mildly above basic-pitch's own defaults
    # (0.5/0.3) - not the earlier, over-tightened 0.6/0.4 that was
    # confirmed to silently discard real notes. minimum_note_length is
    # deliberately BELOW basic-pitch's own default (127.7ms), not above
    # it - see the module-level comment on _MINIMUM_NOTE_LENGTH_MS for
    # why a stricter floor there is a real data-loss regression, not a
    # cleanup win.
    assert 0.5 < kwargs["onset_threshold"] < 0.6
    assert 0.3 < kwargs["frame_threshold"] < 0.4
    assert kwargs["minimum_note_length"] < 127.7


def test_transcribe_to_midi_falls_back_to_120bpm_when_measurement_fails(tmp_path):
    fake_midi = _fake_midi_data([])

    with patch("songforge_mcp.midi_transcription.analyze_audio",
               side_effect=SongForgeMCPError(ErrorCode.INVALID_PARAMETER, "bad audio")), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])) as mock_predict, \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi("input.wav", str(tmp_path / "out.mid"))

    assert mock_predict.call_args.kwargs["midi_tempo"] == 120.0
    assert result["measured_bpm"] == 120.0


def test_transcribe_to_midi_explicit_midi_tempo_override_skips_measurement(tmp_path):
    fake_midi = _fake_midi_data([])

    with patch("songforge_mcp.midi_transcription.analyze_audio") as mock_analyze, \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])) as mock_predict, \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi("input.wav", str(tmp_path / "out.mid"), midi_tempo=93.0)

    mock_analyze.assert_not_called()
    assert mock_predict.call_args.kwargs["midi_tempo"] == 93.0
    assert result["measured_bpm"] == 93.0


def test_transcribe_to_midi_passes_frequency_bounds_and_melodia_trick_to_predict(tmp_path):
    fake_midi = _fake_midi_data([])

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])) as mock_predict, \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        transcribe_to_midi(
            "input.wav", str(tmp_path / "out.mid"),
            minimum_frequency=41.0, maximum_frequency=400.0, melodia_trick=False,
        )

    kwargs = mock_predict.call_args.kwargs
    assert kwargs["minimum_frequency"] == 41.0
    assert kwargs["maximum_frequency"] == 400.0
    assert kwargs["melodia_trick"] is False


def test_transcribe_to_midi_custom_settings_use_a_separate_cache_from_defaults(tmp_path):
    output_path = tmp_path / "out.mid"
    default_path = _tagged(output_path)
    _write_simple_midi(default_path, 5)
    for suffix, count in (("_bass", 2), ("_melody", 1), ("_chords", 2)):
        base, ext = os.path.splitext(default_path)
        _write_simple_midi(f"{base}{suffix}{ext}", count)

    fresh_notes = [_note(60, 0.0, 1.0)]
    fake_midi = _fake_midi_data(fresh_notes)

    # A cached, complete default-settings result exists (the idempotent
    # tests confirm that's normally reused untouched). Passing a
    # non-default onset_threshold must still force a real re-run, writing
    # to its OWN settings-tagged file rather than colliding with or
    # overwriting the default cache.
    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, fresh_notes)) as mock_predict, \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi(str(tmp_path / "input.wav"), str(output_path), onset_threshold=0.8)

    mock_predict.assert_called_once()
    custom_path = _tagged(output_path, onset_threshold=0.8)
    assert custom_path != default_path
    fake_midi.write.assert_called_once_with(custom_path)
    assert result["midi_path"] == custom_path
    assert result["note_count"] == 1
    # The pre-existing default-settings cache must be untouched, not
    # overwritten by this custom-settings run.
    assert _count_midi_notes_helper(default_path) == 5


def _count_midi_notes_helper(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return sum(len(instrument.notes) for instrument in pm.instruments)


def test_transcribe_to_midi_merges_fragmented_notes_before_writing(tmp_path):
    # Same pitch, split into 3 close fragments by frame-flicker - should
    # collapse to 1 sustained note in the file that actually gets written.
    fragments = [_note(60, 0.0, 0.5), _note(60, 0.52, 1.0), _note(60, 1.03, 1.5)]
    fake_midi = _fake_midi_data(fragments)

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, fragments)), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi("input.wav", str(tmp_path / "out.mid"))

    assert result["note_count"] == 1
    assert fake_midi.instruments[0].notes[0].end == 1.5


def test_transcribe_to_midi_creates_output_directory(tmp_path):
    fake_midi = _fake_midi_data([])

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        output_path = str(tmp_path / "nested" / "dir" / "out.mid")
        transcribe_to_midi("input.wav", output_path)

    assert os.path.isdir(tmp_path / "nested" / "dir")


def _write_simple_midi(path, note_count):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [
        pretty_midi.Note(velocity=100, pitch=60, start=float(i), end=float(i) + 0.5)
        for i in range(note_count)
    ]
    pm.instruments.append(instrument)
    pm.write(str(path))


def test_transcribe_to_midi_is_idempotent_and_skips_model_when_output_exists(tmp_path):
    output_path = tmp_path / "out.mid"
    tagged_path = _tagged(output_path)
    base, ext = os.path.splitext(tagged_path)
    _write_simple_midi(tagged_path, 5)
    _write_simple_midi(f"{base}_bass{ext}", 2)
    _write_simple_midi(f"{base}_melody{ext}", 1)
    _write_simple_midi(f"{base}_chords{ext}", 2)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("predict() must not be called when output already exists")

    with patch("songforge_mcp.midi_transcription.predict", side_effect=fail_if_called):
        result = transcribe_to_midi("input.wav", str(output_path))

    assert result["note_count"] == 5
    assert result["bass_note_count"] == 2
    assert result["melody_note_count"] == 1
    assert result["chords_note_count"] == 2


def test_transcribe_to_midi_reruns_when_a_split_file_is_missing(tmp_path):
    output_path = tmp_path / "out.mid"
    _write_simple_midi(_tagged(output_path), 5)
    # bass/melody/chords siblings deliberately not created - incomplete
    # output from a previous run should not be trusted as done.
    rerun_notes = [_note(60, 0.0, 1.0), _note(64, 1.0, 2.0)]
    fake_midi = _fake_midi_data(rerun_notes)

    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, rerun_notes)), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi("input.wav", str(output_path))

    fake_midi.write.assert_called_once_with(_tagged(output_path))
    assert result["note_count"] == 2


def test_get_midi_notes_returns_real_note_data(tmp_path):
    path = tmp_path / "notes.mid"
    _write_simple_midi(path, 3)

    result = get_midi_notes(str(path))

    assert result["total_note_count"] == 3
    assert result["returned_count"] == 3
    assert result["offset"] == 0
    assert result["notes"][0] == {"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 100}
    assert result["notes"][1]["start"] == 1.0


def test_get_midi_notes_sorts_by_start_time(tmp_path):
    path = tmp_path / "notes.mid"
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [
        pretty_midi.Note(velocity=100, pitch=64, start=2.0, end=2.5),
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5),
        pretty_midi.Note(velocity=100, pitch=67, start=1.0, end=1.5),
    ]
    pm.instruments.append(instrument)
    pm.write(str(path))

    result = get_midi_notes(str(path))
    assert [n["pitch"] for n in result["notes"]] == [60, 67, 64]


def test_get_midi_notes_paginates_with_offset_and_max_results(tmp_path):
    path = tmp_path / "notes.mid"
    _write_simple_midi(path, 10)

    page = get_midi_notes(str(path), offset=3, max_results=4)

    assert page["total_note_count"] == 10
    assert page["returned_count"] == 4
    assert page["offset"] == 3
    assert [n["start"] for n in page["notes"]] == [3.0, 4.0, 5.0, 6.0]


def test_get_midi_notes_offset_past_end_returns_empty_page(tmp_path):
    path = tmp_path / "notes.mid"
    _write_simple_midi(path, 2)

    page = get_midi_notes(str(path), offset=100)
    assert page["total_note_count"] == 2
    assert page["returned_count"] == 0
    assert page["notes"] == []


def test_get_midi_notes_clamps_max_results_to_server_side_cap(tmp_path):
    path = tmp_path / "notes.mid"
    _write_simple_midi(path, 600)

    # A caller requesting far more than the cap must not get an
    # unbounded response - that defeats the whole point of pagination
    # existing in the first place (see get_midi_notes' own docstring).
    page = get_midi_notes(str(path), max_results=999_999)

    assert page["total_note_count"] == 600
    assert page["returned_count"] == 500
    assert len(page["notes"]) == 500


def test_transcribe_to_midi_wraps_prediction_failure():
    with patch("songforge_mcp.midi_transcription.analyze_audio", return_value=_FAKE_BPM), \
         patch("songforge_mcp.midi_transcription.predict", side_effect=RuntimeError("model exploded")):
        with pytest.raises(SongForgeMCPError, match="MIDI transcription failed"):
            transcribe_to_midi("input.wav", "out.mid")


def _note(pitch, start, end, velocity=100):
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)


def test_onset_polyphony_counts_simultaneous_notes():
    a = _note(60, 0.0, 1.0)
    b = _note(64, 0.0, 1.0)  # starts with a - both should see polyphony 2
    c = _note(67, 0.5, 1.5)  # starts while a and b are still sounding - polyphony 3
    d = _note(72, 2.0, 3.0)  # starts after everything else ended - polyphony 1

    polyphony = _onset_polyphony([a, b, c, d])
    assert polyphony[id(a)] == 2
    assert polyphony[id(b)] == 2
    assert polyphony[id(c)] == 3
    assert polyphony[id(d)] == 1


def _write_midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = notes
    pm.instruments.append(instrument)
    pm.write(str(path))


def test_split_midi_by_register_separates_bass_from_upper_notes(tmp_path):
    midi_path = tmp_path / "flat.mid"
    bass_note = _note(pitch=36, start=0.0, end=1.0)  # C2, below default threshold 48
    melody_note = _note(pitch=72, start=2.0, end=3.0)  # isolated, high - melody
    _write_midi(midi_path, [bass_note, melody_note])

    result = split_midi_by_register(str(midi_path))

    assert result["bass_note_count"] == 1
    assert result["melody_note_count"] == 1
    assert result["chords_note_count"] == 0
    assert os.path.isfile(result["bass_midi_path"])
    assert os.path.isfile(result["melody_midi_path"])
    assert os.path.isfile(result["chords_midi_path"])

    bass_pm = pretty_midi.PrettyMIDI(result["bass_midi_path"])
    assert len(bass_pm.instruments[0].notes) == 1
    assert bass_pm.instruments[0].notes[0].pitch == 36


def test_split_midi_by_register_classifies_dense_notes_as_chords(tmp_path):
    midi_path = tmp_path / "flat.mid"
    # Three simultaneous upper-register notes - a chord.
    chord = [_note(pitch=60 + i, start=0.0, end=1.0) for i in range(3)]
    # One isolated upper-register note - melody.
    melody_note = _note(pitch=76, start=2.0, end=3.0)
    _write_midi(midi_path, chord + [melody_note])

    result = split_midi_by_register(str(midi_path))

    assert result["chords_note_count"] == 3
    assert result["melody_note_count"] == 1
    assert result["bass_note_count"] == 0


def test_split_midi_by_register_respects_custom_thresholds(tmp_path):
    midi_path = tmp_path / "flat.mid"
    note = _note(pitch=50, start=0.0, end=1.0)
    _write_midi(midi_path, [note])

    result = split_midi_by_register(str(midi_path), bass_pitch_threshold=55)
    assert result["bass_note_count"] == 1


def test_merge_close_notes_glues_same_pitch_fragments():
    # 3 fragments of pitch 60, each gap well under the 60ms default.
    fragments = [_note(60, 0.0, 0.50), _note(60, 0.52, 1.00), _note(60, 1.03, 1.50)]

    merged = _merge_close_notes(fragments)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 1.50


def test_merge_close_notes_leaves_a_real_gap_alone():
    # Same pitch but genuinely far apart (a real repeated note, not a
    # frame-flicker fragment) - must stay two separate notes.
    notes = [_note(60, 0.0, 0.5), _note(60, 2.0, 2.5)]

    merged = _merge_close_notes(notes)

    assert len(merged) == 2


def test_merge_close_notes_never_merges_different_pitches():
    notes = [_note(60, 0.0, 0.50), _note(64, 0.51, 1.00)]

    merged = _merge_close_notes(notes)

    assert len(merged) == 2


def test_close_gaps_extends_note_into_small_following_gap():
    # A bassline with a 100ms sliver of silence between two notes -
    # should become fully legato (no gap).
    notes = [_note(40, 0.0, 0.90), _note(43, 1.00, 2.00)]

    closed = _close_gaps(notes)

    assert closed[0].end == 1.00
    assert closed[1].start == 1.00


def test_close_gaps_leaves_a_real_rest_alone():
    # A full second of silence is a real rest, not transcription noise -
    # must not be closed away.
    notes = [_note(40, 0.0, 0.5), _note(43, 1.5, 2.0)]

    closed = _close_gaps(notes)

    assert closed[0].end == 0.5
    assert closed[1].start == 1.5


def test_split_midi_by_register_closes_small_gaps_in_bass_and_melody(tmp_path):
    midi_path = tmp_path / "flat.mid"
    # Bass: two notes 100ms apart - should end up legato.
    bass_a = _note(pitch=36, start=0.0, end=0.90)
    bass_b = _note(pitch=38, start=1.00, end=2.00)
    # Melody: two isolated notes 100ms apart - should also end up legato.
    melody_a = _note(pitch=72, start=3.0, end=3.90)
    melody_b = _note(pitch=74, start=4.00, end=5.00)
    _write_midi(midi_path, [bass_a, bass_b, melody_a, melody_b])

    result = split_midi_by_register(str(midi_path))

    bass_pm = pretty_midi.PrettyMIDI(result["bass_midi_path"])
    bass_notes = sorted(bass_pm.instruments[0].notes, key=lambda n: n.start)
    assert bass_notes[0].end == 1.00

    melody_pm = pretty_midi.PrettyMIDI(result["melody_midi_path"])
    melody_notes = sorted(melody_pm.instruments[0].notes, key=lambda n: n.start)
    assert melody_notes[0].end == 4.00
