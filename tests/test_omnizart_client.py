import os
import platform
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from songforge_mcp.omnizart_client import (
    _chunk_audio,
    _detect_key_from_notes,
    _find_off_scale_notes,
    _require_configured,
    _resolve_model_path,
    transcribe_with_omnizart,
)
from songforge_mcp_shared.constants import OmnizartTranscription
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_EXE_NAME = "omnizart.exe" if platform.system() == "Windows" else "omnizart"


def _make_fake_venv(tmp_path):
    """Mirrors test_separator_client.py's fixture - a fake interpreter +
    console script on disk so _require_configured's existence checks pass
    without a real omnizart install."""
    venv_dir = tmp_path / "fake_omnizart_venv"
    venv_dir.mkdir()
    python_exe = venv_dir / ("python.exe" if platform.system() == "Windows" else "python")
    python_exe.write_text("")
    omnizart_exe = venv_dir / _EXE_NAME
    omnizart_exe.write_text("")
    return str(python_exe)


def _note(pitch, start, end, velocity=100):
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)


def _write_wav(path, seconds=1.0, sr=22050):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), y, sr)


def _write_chunk_midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = notes
    pm.instruments.append(instrument)
    pm.write(str(path))


# ───────────────────────── _require_configured ─────────────────────────

def test_require_configured_raises_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("SONGFORGE_OMNIZART_PYTHON", raising=False)
    with pytest.raises(SongForgeMCPError) as exc_info:
        _require_configured()
    assert exc_info.value.code == ErrorCode.OMNIZART_NOT_CONFIGURED


def test_require_configured_raises_when_exe_missing_alongside_python(tmp_path, monkeypatch):
    python_exe = tmp_path / ("python.exe" if platform.system() == "Windows" else "python")
    python_exe.write_text("")
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", str(python_exe))
    with pytest.raises(SongForgeMCPError) as exc_info:
        _require_configured()
    assert exc_info.value.code == ErrorCode.OMNIZART_NOT_CONFIGURED


def test_require_configured_returns_both_paths_when_valid(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)
    resolved_python, resolved_omnizart = _require_configured()
    assert resolved_python == python_exe
    assert os.path.isfile(resolved_omnizart)


# ───────────────────────── _resolve_model_path ─────────────────────────

def _lib_dir_name():
    return "Lib" if platform.system() == "Windows" else "lib"


def test_resolve_model_path_uses_tuned_checkpoint_when_present(tmp_path):
    venv_dir = tmp_path / "venv"
    python_exe = venv_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("")

    music_dir = venv_dir / _lib_dir_name() / "site-packages" / "omnizart" / "checkpoints" / "music"
    # The tuned checkpoint's stock sibling always exists in a real install
    # (install_omnizart.py creates the tuned copy FROM it) - _resolve_model_path
    # requires it as a signal this is really Omnizart's own checkpoint tree,
    # not an unrelated package's same-named directory.
    (music_dir / "music_piano").mkdir(parents=True)
    tuned_dir = music_dir / OmnizartTranscription.TUNED_PIANO_CHECKPOINT_NAME
    tuned_dir.mkdir(parents=True)

    resolved = _resolve_model_path(str(python_exe), "Piano")

    assert resolved == str(tuned_dir)


def test_resolve_model_path_falls_back_to_bare_mode_when_tuned_checkpoint_missing(tmp_path):
    venv_dir = tmp_path / "venv"
    python_exe = venv_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("")

    # A real music/ checkpoints dir exists, but no tuned variant inside it -
    # e.g. an Omnizart env installed before this tuning existed.
    music_dir = venv_dir / _lib_dir_name() / "site-packages" / "omnizart" / "checkpoints" / "music"
    music_dir.mkdir(parents=True)

    resolved = _resolve_model_path(str(python_exe), "Piano")

    assert resolved == "Piano"


def test_resolve_model_path_passes_through_non_piano_modes_unchanged(tmp_path):
    venv_dir = tmp_path / "venv"
    python_exe = venv_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("")
    music_dir = venv_dir / _lib_dir_name() / "site-packages" / "omnizart" / "checkpoints" / "music"
    (music_dir / OmnizartTranscription.TUNED_PIANO_CHECKPOINT_NAME).mkdir(parents=True)

    # The tuning was only measured/validated for Piano - Pop/Stream/an
    # explicit custom path must never be silently redirected to it.
    assert _resolve_model_path(str(python_exe), "Pop") == "Pop"
    assert _resolve_model_path(str(python_exe), "Stream") == "Stream"
    assert _resolve_model_path(str(python_exe), "/some/custom/checkpoint") == "/some/custom/checkpoint"


# ───────────────────────── _chunk_audio ─────────────────────────

def test_chunk_audio_splits_into_expected_number_of_chunks(tmp_path):
    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=10.0)
    out_dir = tmp_path / "chunks"
    out_dir.mkdir()

    chunks = _chunk_audio(str(audio_path), chunk_seconds=3.0, out_dir=str(out_dir))

    assert len(chunks) == 4  # 0-3, 3-6, 6-9, 9-10
    assert [c["start"] for c in chunks] == [0.0, 3.0, 6.0, 9.0]
    for c in chunks:
        assert os.path.isfile(c["path"])


def test_chunk_audio_single_chunk_when_shorter_than_chunk_size(tmp_path):
    audio_path = tmp_path / "short.wav"
    _write_wav(audio_path, seconds=2.0)
    out_dir = tmp_path / "chunks"
    out_dir.mkdir()

    chunks = _chunk_audio(str(audio_path), chunk_seconds=30.0, out_dir=str(out_dir))

    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0


# ───────────────────────── key detection / off-scale ─────────────────────────

def test_detect_key_from_notes_finds_c_major():
    # C major triad, held long enough to dominate the weighted histogram.
    notes = [_note(60, 0, 4), _note(64, 0, 4), _note(67, 0, 4)]
    root, mode, confidence = _detect_key_from_notes(notes)
    assert root == "C"
    assert mode == "major"
    assert confidence > 0


def test_detect_key_from_notes_handles_empty_list():
    root, mode, confidence = _detect_key_from_notes([])
    assert (root, mode, confidence) == (None, None, 0.0)


def test_find_off_scale_notes_flags_notes_outside_detected_key():
    in_key = [_note(60, 0, 1), _note(62, 1, 2), _note(64, 2, 3)]  # C, D, E
    out_of_key = [_note(61, 3, 4)]  # C#, not in C major
    result = _find_off_scale_notes(in_key + out_of_key, "C", "major")
    assert result == out_of_key


def test_find_off_scale_notes_returns_empty_when_no_key_detected():
    notes = [_note(60, 0, 1)]
    assert _find_off_scale_notes(notes, None, None) == []


# ───────────────────────── transcribe_with_omnizart ─────────────────────────

def test_transcribe_with_omnizart_raises_when_audio_missing(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        transcribe_with_omnizart(str(tmp_path / "missing.wav"), str(tmp_path / "out.mid"))
    assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


def test_transcribe_with_omnizart_chunks_calls_subprocess_per_chunk_and_offsets_notes(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=7.0)  # 3 chunks at chunk_seconds=3: 0-3,3-6,6-7

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        chunk_wav = cmd[3]
        output_idx = cmd.index("--output") + 1
        chunk_midi = cmd[output_idx]
        # Each fake chunk transcription reports one note at time 0.5s
        # relative to ITS OWN chunk - transcribe_with_omnizart must shift
        # this by the chunk's real start offset.
        _write_chunk_midi(chunk_midi, [_note(60, 0.5, 1.0)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        result = transcribe_with_omnizart(
            str(audio_path), str(tmp_path / "out.mid"),
            chunk_seconds=3.0, midi_tempo=120.0,
        )

    assert call_count["n"] == 3
    assert result["chunk_count"] == 3
    assert result["note_count"] == 3

    pm = pretty_midi.PrettyMIDI(result["midi_path"])
    starts = sorted(n.start for inst in pm.instruments for n in inst.notes)
    # One note per chunk, each originally at +0.5s within its own chunk,
    # shifted by that chunk's real start (0, 3, 6).
    assert starts == pytest.approx([0.5, 3.5, 6.5])


def test_transcribe_with_omnizart_merges_fragments_across_chunk_output(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=3.0)

    def fake_run(cmd, **kwargs):
        output_idx = cmd.index("--output") + 1
        chunk_midi = cmd[output_idx]
        # Two close same-pitch fragments within one chunk - should merge.
        _write_chunk_midi(chunk_midi, [_note(60, 0.0, 0.5), _note(60, 0.52, 1.0)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        result = transcribe_with_omnizart(
            str(audio_path), str(tmp_path / "out.mid"),
            chunk_seconds=30.0, midi_tempo=120.0,
        )

    assert result["note_count"] == 1


def test_transcribe_with_omnizart_does_not_close_gaps_across_different_pitches(tmp_path, monkeypatch):
    # Regression test: _close_gaps assumes a single pitch/line (see its
    # docstring in midi_transcription.py) - applying it to the flat,
    # all-pitches-mixed note list Omnizart's default Piano mode produces
    # would wrongly stretch one note's end into an unrelated, different-
    # pitch note's onset just because they're chronologically close.
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=3.0)

    def fake_run(cmd, **kwargs):
        output_idx = cmd.index("--output") + 1
        chunk_midi = cmd[output_idx]
        # Two different-pitch notes with a small (legato-closeable) gap
        # between them - must NOT be glued together despite the gap
        # being small enough that same-pitch notes would be.
        _write_chunk_midi(chunk_midi, [_note(60, 0.0, 0.5), _note(64, 0.6, 1.0)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        result = transcribe_with_omnizart(
            str(audio_path), str(tmp_path / "out.mid"),
            chunk_seconds=30.0, midi_tempo=120.0,
        )

    assert result["note_count"] == 2
    pm = pretty_midi.PrettyMIDI(result["midi_path"])
    notes_by_pitch = {n.pitch: n for inst in pm.instruments for n in inst.notes}
    assert notes_by_pitch[60].end == pytest.approx(0.5)  # not stretched to 0.6
    assert notes_by_pitch[64].start == pytest.approx(0.6)


def test_transcribe_with_omnizart_raises_on_nonzero_exit_code(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="model failed to load")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        with pytest.raises(SongForgeMCPError) as exc_info:
            transcribe_with_omnizart(
                str(audio_path), str(tmp_path / "out.mid"), midi_tempo=120.0,
            )
    assert exc_info.value.code == ErrorCode.OMNIZART_TRANSCRIPTION_FAILED


def test_transcribe_with_omnizart_raises_subprocess_timeout_as_songforge_error(tmp_path, monkeypatch):
    # Matches SeparatorClient's established pattern: a raw
    # subprocess.TimeoutExpired must never escape to the caller as-is.
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        with pytest.raises(SongForgeMCPError) as exc_info:
            transcribe_with_omnizart(
                str(audio_path), str(tmp_path / "out.mid"), midi_tempo=120.0,
            )
    assert exc_info.value.code == ErrorCode.SUBPROCESS_TIMEOUT


def test_transcribe_with_omnizart_raises_when_output_file_missing_despite_success(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)

    def fake_run(cmd, **kwargs):
        # Reports success but never actually writes the expected chunk file.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        with pytest.raises(SongForgeMCPError) as exc_info:
            transcribe_with_omnizart(
                str(audio_path), str(tmp_path / "out.mid"), midi_tempo=120.0,
            )
    assert exc_info.value.code == ErrorCode.OMNIZART_TRANSCRIPTION_FAILED


def test_transcribe_with_omnizart_is_idempotent_and_skips_subprocess_when_cached(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)

    def fail_if_called(cmd, **kwargs):
        raise AssertionError("subprocess.run must not be called on a cache hit")

    # Seed a cached output file at the exact tagged path this call would
    # compute, matching the settings-tag pattern from midi_transcription.py.
    from songforge_mcp.omnizart_client import _settings_tag

    tag = _settings_tag("omnizart", "Piano", 30.0, 0.06, 0.35, 120.0)
    output_path = tmp_path / "out.mid"
    tagged_path = f"{output_path.with_suffix('')}__{tag}.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [_note(60, 0, 1), _note(62, 1, 2)]
    pm.instruments.append(instrument)
    pm.write(tagged_path)

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fail_if_called):
        result = transcribe_with_omnizart(
            str(audio_path), str(output_path),
            model_mode="Piano", chunk_seconds=30.0, midi_tempo=120.0,
        )

    assert result["midi_path"] == tagged_path
    assert result["note_count"] == 2
    assert result["chunk_count"] is None  # not tracked for a cache hit
    assert "detected_key" in result
    assert "off_scale_notes" in result


def test_transcribe_with_omnizart_different_settings_use_separate_cache(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)
    monkeypatch.setenv("SONGFORGE_OMNIZART_PYTHON", python_exe)

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        output_idx = cmd.index("--output") + 1
        _write_chunk_midi(cmd[output_idx], [_note(60, 0.0, 0.5)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        result_a = transcribe_with_omnizart(
            str(audio_path), str(tmp_path / "out.mid"), model_mode="Piano", midi_tempo=120.0,
        )
        result_b = transcribe_with_omnizart(
            str(audio_path), str(tmp_path / "out.mid"), model_mode="Pop", midi_tempo=120.0,
        )

    assert call_count["n"] == 2  # both ran fresh, no cache collision
    assert result_a["midi_path"] != result_b["midi_path"]


def test_transcribe_with_omnizart_cache_key_changes_when_tuned_checkpoint_appears(tmp_path):
    # Regression test for the exact bug class already confirmed and fixed
    # twice this session (separator_client._model_tag,
    # midi_transcription._settings_tag): the cache key must be based on
    # what ACTUALLY runs, not just the caller-supplied mode name. "Piano"
    # resolves to a different real checkpoint depending on whether
    # scripts/install_omnizart.py has created a tuned variant - if the
    # cache key only recorded "Piano", a stale-settings hit here would be
    # exactly as real and exactly as wrong as the earlier two bugs.
    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, seconds=2.0)
    output_path = tmp_path / "out.mid"

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        output_idx = cmd.index("--output") + 1
        _write_chunk_midi(cmd[output_idx], [_note(60, 0.0, 0.5)])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # First run: no tuned checkpoint in this fake venv - resolves to bare "Piano".
    python_exe_untuned = _make_fake_venv(tmp_path)
    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"SONGFORGE_OMNIZART_PYTHON": python_exe_untuned}):
            result_untuned = transcribe_with_omnizart(
                str(audio_path), str(output_path), model_mode="Piano", midi_tempo=120.0,
            )

    # Second run: a DIFFERENT fake venv that DOES have the tuned checkpoint.
    venv2 = tmp_path / "venv2"
    python_exe_tuned = venv2 / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    python_exe_tuned.parent.mkdir(parents=True)
    python_exe_tuned.write_text("")
    (venv2 / ("Scripts" if platform.system() == "Windows" else "bin") / _EXE_NAME).write_text("")
    music_dir = venv2 / _lib_dir_name() / "site-packages" / "omnizart" / "checkpoints" / "music"
    (music_dir / "music_piano").mkdir(parents=True)
    (music_dir / OmnizartTranscription.TUNED_PIANO_CHECKPOINT_NAME).mkdir(parents=True)

    with patch("songforge_mcp.omnizart_client.subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"SONGFORGE_OMNIZART_PYTHON": str(python_exe_tuned)}):
            result_tuned = transcribe_with_omnizart(
                str(audio_path), str(output_path), model_mode="Piano", midi_tempo=120.0,
            )

    assert call_count["n"] == 2  # second call must NOT hit the first's cache
    assert result_untuned["midi_path"] != result_tuned["midi_path"]
