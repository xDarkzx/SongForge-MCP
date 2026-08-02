import os
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from songforge_mcp.midi_transcription import (
    _onset_polyphony,
    get_midi_notes,
    split_midi_by_register,
    transcribe_to_midi,
)
from songforge_mcp_shared.error_codes import SongForgeMCPError


def _fake_midi_data():
    midi = MagicMock()
    midi.write = MagicMock()
    return midi


_FAKE_SPLIT_RESULT = {
    "bass_midi_path": "/fake/out_bass.mid", "bass_note_count": 0,
    "melody_midi_path": "/fake/out_melody.mid", "melody_note_count": 0,
    "chords_midi_path": "/fake/out_chords.mid", "chords_note_count": 0,
}


def test_transcribe_to_midi_writes_file_and_returns_note_count(tmp_path):
    fake_midi = _fake_midi_data()
    fake_notes = [object(), object(), object()]

    with patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, fake_notes)), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        output_path = str(tmp_path / "out.mid")
        result = transcribe_to_midi("input.wav", output_path)

    fake_midi.write.assert_called_once_with(output_path)
    assert result["midi_path"] == output_path
    assert result["note_count"] == 3
    assert result["bass_midi_path"] == "/fake/out_bass.mid"


def test_transcribe_to_midi_creates_output_directory(tmp_path):
    fake_midi = _fake_midi_data()

    with patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [])), \
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
    _write_simple_midi(output_path, 5)
    _write_simple_midi(tmp_path / "out_bass.mid", 2)
    _write_simple_midi(tmp_path / "out_melody.mid", 1)
    _write_simple_midi(tmp_path / "out_chords.mid", 2)

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
    _write_simple_midi(output_path, 5)
    # bass/melody/chords siblings deliberately not created - incomplete
    # output from a previous run should not be trusted as done.
    fake_midi = _fake_midi_data()

    with patch("songforge_mcp.midi_transcription.predict", return_value=(None, fake_midi, [1, 2])), \
         patch("songforge_mcp.midi_transcription.split_midi_by_register", return_value=_FAKE_SPLIT_RESULT):
        result = transcribe_to_midi("input.wav", str(output_path))

    fake_midi.write.assert_called_once_with(str(output_path))
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
    with patch("songforge_mcp.midi_transcription.predict", side_effect=RuntimeError("model exploded")):
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
