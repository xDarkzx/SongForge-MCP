"""Real audio measurement (BPM, key) via librosa.

Promoted from ad-hoc scratch scripts used during live debugging of the
vocal-stem + new-instrumental workflow into a proper, reusable capability.
The point: BPM/Key values fed into generate_vocal_track's
advanced_settings should come from actually measuring a reference clip,
never from a person (or the calling model) guessing per-song.

Deliberately does NOT attempt chord-progression detection. A DSP
chord-template matcher was tried and measured (bar-synchronous chroma +
major/minor triad templates) - its overall key call was actually correct
(confirmed against a real chord chart for a real, identified song), but
its chord-by-chord output ("C# - B - D#m...") did not match that song's
real, published chords ("Gb - Db - Ab - Bbm") and was rejected by ear as
not fitting the vocal. For a remix workflow specifically, the real chords
are knowable directly - the source song is real and (for anything with an
online chord chart) has a real, correct progression already documented -
so guessing them from audio via DSP is the wrong tool for that job. The
calling model should identify the song and look up its real chord chart
using its own general web search/fetch capability, then feed that real
progression into generate_vocal_track's caption directly. This module
still measures BPM/key from audio - a much better-established, more
reliable DSP task than chord recognition - as a cross-check or for audio
with no identifiable song/online source (e.g. private, unreleased
recordings).
"""
import numpy as np
import librosa

from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler / Krumhansl-Kessler (1982) key-profile weights - a
# standard, well-established perceptual key-fit model, not invented for
# this project.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_ANALYSIS_SR = 22050


def detect_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Correlates a 12-bin chroma vector against all 24 (12 roots x 2
    modes) Krumhansl-Schmuckler profiles and returns the best-fitting
    (root, mode, correlation)."""
    best_root, best_mode, best_corr = None, None, -2.0
    for profile, mode in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")):
        for shift in range(12):
            rotated = np.roll(profile, shift)
            corr = float(np.corrcoef(rotated, chroma_mean)[0, 1])
            if corr > best_corr:
                best_root, best_mode, best_corr = _PITCH_CLASSES[shift], mode, corr
    return best_root, best_mode, best_corr


def _pitch_stats(f0_hz: np.ndarray) -> dict:
    """Percentile-based summary of a set of voiced f0 (Hz) samples - not
    raw min/max. pyin (like any monophonic pitch tracker) produces
    occasional octave errors and picks up noise/breath as spuriously low
    or high frames; a single bad frame shouldn't define "the range".
    Confirmed on real data: sampling 40 real vocal clips measured a raw
    min of 65.4 Hz (C2, implausibly low for the actual singer) against a
    5th-percentile of 201.7 Hz (G3, a plausible real floor) - the
    percentile is the trustworthy number, raw min/max are included only
    for reference."""
    return {
        "min_hz": round(float(f0_hz.min()), 1),
        "p5_hz": round(float(np.percentile(f0_hz, 5)), 1),
        "p25_hz": round(float(np.percentile(f0_hz, 25)), 1),
        "median_hz": round(float(np.median(f0_hz)), 1),
        "p75_hz": round(float(np.percentile(f0_hz, 75)), 1),
        "p95_hz": round(float(np.percentile(f0_hz, 95)), 1),
        "max_hz": round(float(f0_hz.max()), 1),
        "practical_low_note": librosa.hz_to_note(float(np.percentile(f0_hz, 5))),
        "practical_high_note": librosa.hz_to_note(float(np.percentile(f0_hz, 95))),
        "voiced_frame_count": int(len(f0_hz)),
    }


def _voiced_f0(path: str, fmin_note: str, fmax_note: str) -> np.ndarray:
    try:
        y, sr = librosa.load(path, sr=_ANALYSIS_SR, mono=True)
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER, f"could not load {path!r} for pitch analysis: {e}"
        ) from e
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz(fmin_note), fmax=librosa.note_to_hz(fmax_note), sr=sr
    )
    return f0[voiced_flag]


def measure_vocal_pitch_range(path: str, fmin_note: str = "C2", fmax_note: str = "C6") -> dict:
    """Measures the actual sung pitch range of a vocal-only audio file via
    librosa's pyin (probabilistic YIN) f0 tracker - a well-established DSP
    technique for singing-voice pitch tracking, not invented for this
    project. ACE-Step has no vocal pitch-range setting anywhere in its own
    UI/source (confirmed by inspection) - this exists because "does this
    melody go outside the singer's natural range" has to be measured
    after the fact, there is no parameter to constrain it during
    generation."""
    voiced_f0 = _voiced_f0(path, fmin_note, fmax_note)
    if len(voiced_f0) == 0:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"no voiced pitch detected in {path!r} - is this actually a vocal-only file?",
        )
    return _pitch_stats(voiced_f0)


def aggregate_vocal_pitch_range(paths: list[str], fmin_note: str = "C2", fmax_note: str = "C6") -> dict:
    """Like measure_vocal_pitch_range, but combines voiced pitch across
    many files into one range - for establishing a reference singer's
    natural range from a sample pack of many short clips rather than a
    single file, which is far more robust than trusting any one clip's
    own range (a single short ad-lib might not span the singer's real
    range at all)."""
    all_f0: list[float] = []
    for path in paths:
        all_f0.extend(_voiced_f0(path, fmin_note, fmax_note).tolist())
    if not all_f0:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER, "no voiced pitch detected across any of the provided files"
        )
    return _pitch_stats(np.array(all_f0))


def compare_vocal_pitch_range(candidate: dict, reference: dict) -> dict:
    """Compares a candidate vocal's measured range (from
    measure_vocal_pitch_range/aggregate_vocal_pitch_range) against a
    reference range, in semitones - the measured version of "this melody
    goes outside the singer's natural range", instead of a subjective
    "sounds robotic when it's high" impression. Compares practical range
    (p5-p95), not raw min/max, for the same reason _pitch_stats uses
    percentiles - a single bad pitch-tracking frame at either end
    shouldn't decide whether something is flagged."""
    exceeds_high = candidate["p95_hz"] > reference["p95_hz"]
    exceeds_low = candidate["p5_hz"] < reference["p5_hz"]
    return {
        "within_range": not (exceeds_high or exceeds_low),
        "exceeds_reference_high": exceeds_high,
        "exceeds_reference_low": exceeds_low,
        "high_excess_semitones": (
            round(12 * np.log2(candidate["p95_hz"] / reference["p95_hz"]), 2) if exceeds_high else 0.0
        ),
        "low_excess_semitones": (
            round(12 * np.log2(reference["p5_hz"] / candidate["p5_hz"]), 2) if exceeds_low else 0.0
        ),
    }


# librosa.beat.beat_track's own documented failure mode (also confirmed
# independently in beat_alignment.py's _correct_octave_ambiguity, and
# again directly during this project's own testing: a real song's known
# 155 BPM measured as 86.1 BPM via librosa AND as 76.9-84.5 BPM via
# madmom's separately-trained neural beat tracker - two methodologically
# different algorithms landing on the same wrong answer, both close to
# half the true tempo). The bias is one-directional: fast tempos get
# UNDER-detected toward a slower "typical" prior, not the reverse - so a
# raw reading in this band is specifically the signature of a real tempo
# that's roughly double it, not a random ambiguity in both directions.
# Band chosen so its doubled range (150-210) covers the documented "150+
# BPM" risk zone without flagging confidently-slow material (a real
# 60-70 BPM ballad has no comparable evidence of being mismeasured).
_OCTAVE_AMBIGUITY_LOW = 75.0
_OCTAVE_AMBIGUITY_HIGH = 105.0


def analyze_audio(path: str) -> dict:
    """Measures real tempo and overall key/mode from an audio file. Used
    to auto-populate generate_vocal_track's advanced_settings
    ("BPM (Beats Per Minute)", "Key") instead of guessing either by ear or
    by convention. Does not attempt chord-progression detection - see
    module docstring for why; look up the real song's chord chart instead
    when one exists.

    bpm is a real DSP measurement, not a guess - but tempo tracking has a
    real, confirmed failure mode of its own (see _OCTAVE_AMBIGUITY_LOW/
    HIGH above): a raw reading between them is reported alongside
    bpm_alt (exactly double it) and bpm_low_confidence=True, because
    that band is specifically where this project has confirmed real
    tempos get mismeasured as roughly half their true value. This is
    surfaced, not silently resolved - picking the wrong one automatically
    without a trusted reference (unlike beat_alignment.py's
    _correct_octave_ambiguity, which has one) would just swap a visible
    uncertainty for a silent wrong answer."""
    try:
        y, sr = librosa.load(path, sr=_ANALYSIS_SR, mono=True)
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER, f"could not load {path!r} for audio analysis: {e}"
        ) from e

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode, confidence = detect_key(chroma.mean(axis=1))

    low_confidence = _OCTAVE_AMBIGUITY_LOW <= tempo <= _OCTAVE_AMBIGUITY_HIGH
    result = {
        "bpm": round(tempo, 1),
        "key": key,
        "mode": mode,
        "key_confidence": round(confidence, 3),
        "duration_seconds": round(len(y) / sr, 1),
        "bpm_low_confidence": low_confidence,
    }
    if low_confidence:
        result["bpm_alt"] = round(tempo * 2, 1)
        result["bpm_note"] = (
            f"Measured {round(tempo, 1)} BPM, but this falls in a known "
            f"tempo-tracker risk band where a real fast tempo commonly gets "
            f"under-detected as roughly half its true value (confirmed on "
            f"real material in this project - see analyze_audio's "
            f"docstring). If this seems too slow for the actual song, the "
            f"real tempo may be bpm_alt instead."
        )
    return result
