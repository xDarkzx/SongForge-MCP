import numpy as np
import pytest
import soundfile as sf

from songforge_mcp.audio_analysis import (
    aggregate_vocal_pitch_range,
    analyze_audio,
    compare_vocal_pitch_range,
    detect_key,
    measure_vocal_pitch_range,
)
from songforge_mcp_shared.error_codes import SongForgeMCPError

_NOTE_FREQS = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63, "F": 349.23,
    "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}


def _chroma_bump(*pitch_classes, magnitude=1.0):
    """A synthetic 12-bin chroma vector with mass only on the given pitch
    classes, plus a small uniform floor so correlation isn't degenerate."""
    chroma = np.full(12, 0.05)
    order = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for pc in pitch_classes:
        chroma[order.index(pc)] = magnitude
    return chroma


def test_detect_key_identifies_c_major_triad():
    chroma = _chroma_bump("C", "E", "G")
    key, mode, corr = detect_key(chroma)
    assert (key, mode) == ("C", "major")
    assert corr > 0.5


def test_detect_key_identifies_a_minor_triad():
    chroma = _chroma_bump("A", "C", "E")
    key, mode, corr = detect_key(chroma)
    assert (key, mode) == ("A", "minor")
    assert corr > 0.5


def test_detect_key_transposes_correctly():
    chroma = _chroma_bump("G#", "C", "D#")  # G# major triad
    key, mode, _ = detect_key(chroma)
    assert (key, mode) == ("G#", "major")


def _write_synthetic_song(path, bpm=120.0, samplerate=22050, duration_seconds=8.0):
    """A synthetic clip with a clear beat (percussive clicks at exactly
    `bpm`) and a sustained C major chord underneath, so both beat-tracking
    and key detection have real, unambiguous signal to lock onto."""
    n = int(duration_seconds * samplerate)
    t = np.linspace(0, duration_seconds, n, endpoint=False)

    chord_tone = np.zeros(n)
    for pc in ("C", "E", "G"):
        freq = _NOTE_FREQS[pc]
        chord_tone += 0.15 * np.sin(2 * np.pi * freq * t)

    beat_interval = 60.0 / bpm
    click = np.zeros(n)
    click_len = int(0.02 * samplerate)
    beat_time = 0.0
    rng = np.random.default_rng(0)
    while beat_time < duration_seconds:
        start = int(beat_time * samplerate)
        end = min(start + click_len, n)
        click[start:end] += 0.6 * rng.standard_normal(end - start)
        beat_time += beat_interval

    audio = chord_tone + click
    audio = audio / np.max(np.abs(audio))
    sf.write(str(path), audio * 0.9, samplerate)


def _write_tone(path, freq_hz, duration_seconds=2.0, samplerate=22050):
    """A pure sine tone at a fixed, known pitch - a stand-in for a
    steady, sustained vocal note, letting pyin's detected f0 be checked
    against a known ground truth rather than a real, ambiguous voice."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * samplerate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(str(path), audio, samplerate)


def test_measure_vocal_pitch_range_detects_known_steady_pitch(tmp_path):
    path = tmp_path / "steady_note.wav"
    _write_tone(path, freq_hz=440.0)  # A4

    result = measure_vocal_pitch_range(str(path))

    assert abs(result["median_hz"] - 440.0) < 5.0
    assert abs(result["p5_hz"] - 440.0) < 5.0
    assert abs(result["p95_hz"] - 440.0) < 5.0
    assert result["voiced_frame_count"] > 0


def test_measure_vocal_pitch_range_rejects_silence(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(22050 * 2), 22050)

    with pytest.raises(SongForgeMCPError, match="no voiced pitch"):
        measure_vocal_pitch_range(str(path))


def test_aggregate_vocal_pitch_range_combines_multiple_files(tmp_path):
    low_path = tmp_path / "low.wav"
    high_path = tmp_path / "high.wav"
    _write_tone(low_path, freq_hz=220.0)  # A3
    _write_tone(high_path, freq_hz=880.0)  # A5

    result = aggregate_vocal_pitch_range([str(low_path), str(high_path)])

    # Combined range should span from near the low file's pitch to near
    # the high file's pitch - neither file alone would show this spread.
    assert result["p5_hz"] < 300.0
    assert result["p95_hz"] > 700.0


def test_compare_vocal_pitch_range_flags_when_candidate_exceeds_reference():
    reference = {"p5_hz": 200.0, "p95_hz": 550.0}
    candidate = {"p5_hz": 200.0, "p95_hz": 1100.0}  # an octave above reference's top

    result = compare_vocal_pitch_range(candidate, reference)

    assert result["within_range"] is False
    assert result["exceeds_reference_high"] is True
    assert result["exceeds_reference_low"] is False
    assert result["high_excess_semitones"] == pytest.approx(12.0, abs=0.1)
    assert result["low_excess_semitones"] == 0.0


def test_compare_vocal_pitch_range_reports_within_range():
    reference = {"p5_hz": 200.0, "p95_hz": 550.0}
    candidate = {"p5_hz": 220.0, "p95_hz": 500.0}

    result = compare_vocal_pitch_range(candidate, reference)

    assert result["within_range"] is True
    assert result["exceeds_reference_high"] is False
    assert result["exceeds_reference_low"] is False


def test_analyze_audio_measures_tempo_key_and_duration(tmp_path):
    path = tmp_path / "synthetic_song.wav"
    _write_synthetic_song(path, bpm=120.0, duration_seconds=8.0)

    result = analyze_audio(str(path))

    assert set(result) == {
        "bpm", "key", "mode", "key_confidence", "duration_seconds", "bpm_low_confidence",
    }
    assert abs(result["bpm"] - 120.0) < 10.0
    assert result["key"] == "C"
    assert result["mode"] == "major"
    assert result["duration_seconds"] == 8.0
    # 120 BPM is well above the octave-ambiguity risk band (75-105) -
    # confirmed a real, correctly-measured tempo, not a candidate for the
    # tempo-under-detection failure mode.
    assert result["bpm_low_confidence"] is False
    assert "bpm_alt" not in result


def test_analyze_audio_flags_bpm_in_octave_ambiguity_band(tmp_path):
    path = tmp_path / "synthetic_song_90bpm.wav"
    _write_synthetic_song(path, bpm=90.0, duration_seconds=8.0)

    result = analyze_audio(str(path))

    # 90 BPM sits inside the confirmed risk band (75-105) where a real
    # fast tempo commonly gets under-detected as roughly half its true
    # value - this must be surfaced, not silently trusted.
    assert result["bpm_low_confidence"] is True
    assert abs(result["bpm_alt"] - result["bpm"] * 2) < 0.01
    assert "bpm_note" in result and "bpm_alt" in result["bpm_note"]


def test_analyze_audio_does_not_flag_bpm_outside_ambiguity_band(tmp_path):
    path = tmp_path / "synthetic_song_60bpm.wav"
    _write_synthetic_song(path, bpm=60.0, duration_seconds=8.0)

    result = analyze_audio(str(path))

    assert result["bpm_low_confidence"] is False
    assert "bpm_alt" not in result
    assert "bpm_note" not in result
