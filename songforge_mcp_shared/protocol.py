"""Validation helpers for ACE-Step generation requests, and output parsing."""
import os

import pretty_midi
import soundfile as sf

from songforge_mcp_shared.constants import (
    MAX_CAPTION_LENGTH,
    MAX_LYRICS_LENGTH,
    Paths,
)
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def validate_caption(caption: str) -> None:
    """Caption is free-form genre/mood/instrumentation/vocal-style text
    (e.g. "melodic dubstep, female vocals, dreamy, atmospheric, 150 BPM").
    ACE-Step has no note/phoneme input — composition is entirely prompt +
    lyrics driven."""
    if not caption or not caption.strip():
        raise SongForgeMCPError(ErrorCode.MISSING_PARAMETER, "caption must not be empty")
    if len(caption) > MAX_CAPTION_LENGTH:
        raise SongForgeMCPError(
            ErrorCode.VALUE_OUT_OF_RANGE,
            f"caption length {len(caption)} exceeds the {MAX_CAPTION_LENGTH}-character limit",
        )


def validate_lyrics(lyrics: str) -> None:
    """Lyrics use ACE-Step's structure tags: [verse], [chorus], [bridge],
    [pre-chorus], [outro], [instrumental]/[inst] for instrumental-only
    sections. At least one structure tag is required so the model has a
    real song structure to follow, not a single unstructured blob."""
    if not lyrics or not lyrics.strip():
        raise SongForgeMCPError(ErrorCode.MISSING_PARAMETER, "lyrics must not be empty")
    if len(lyrics) > MAX_LYRICS_LENGTH:
        raise SongForgeMCPError(
            ErrorCode.VALUE_OUT_OF_RANGE,
            f"lyrics length {len(lyrics)} exceeds the {MAX_LYRICS_LENGTH}-character limit",
        )
    if "[" not in lyrics or "]" not in lyrics:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            "lyrics must include at least one structure tag, e.g. [verse] or [chorus] "
            "— ACE-Step uses these to understand song structure",
        )


_ALLOWED_OUTPUT_FORMATS = frozenset({"flac", "mp3", "opus", "aac", "wav", "wav32"})


def validate_output_format(output_format: str) -> str:
    """output_format must be one of ACE-Step's own real save formats
    (acestep/audio_utils.py) — same set exposed by acestep_client's
    OUTPUT_FORMATS. Returns the lowercased value on success."""
    normalized = (output_format or "").strip().lower()
    if normalized not in _ALLOWED_OUTPUT_FORMATS:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"output_format {output_format!r} is not one of {sorted(_ALLOWED_OUTPUT_FORMATS)}",
        )
    return normalized


def validate_audio_file_path(path: str, *, param_name: str) -> str:
    """Guards every file path this server accepts from a calling model
    (`reference_audio_path`, `split_vocal_stems`'s `audio_path`) before
    it's ever opened, uploaded to a browser session, or handed to a
    subprocess. A calling model could pass any path string, accidentally
    or via a manipulated/injected instruction — this does not restrict
    *which* directory a file may live in (pointing at a personal sample
    library anywhere on disk is legitimate use), but it does insist the
    target actually exists, is a real file (not a directory or a symlink
    resolving somewhere unexpected), has an audio-like extension, and
    parses as genuine audio content — rejecting a renamed non-audio file
    rather than trusting its extension alone. Returns the resolved
    absolute path on success.
    """
    if not path or not str(path).strip():
        raise SongForgeMCPError(ErrorCode.MISSING_PARAMETER, f"{param_name} must not be empty")

    resolved = os.path.realpath(path)
    if not os.path.isfile(resolved):
        raise SongForgeMCPError(
            ErrorCode.FILE_NOT_FOUND, f"{param_name} does not exist or is not a file: {path}"
        )

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"{param_name} has an unsupported extension {ext!r} — expected one of "
            f"{sorted(_ALLOWED_AUDIO_EXTENSIONS)}",
        )

    try:
        sf.info(resolved)
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"{param_name} does not appear to be a valid audio file (failed to parse "
            f"as audio despite its extension): {e}",
        ) from e

    return resolved


def validate_output_dir_audio_path(path: str, *, param_name: str) -> str:
    """Like validate_audio_file_path, but additionally refuses any path that
    doesn't resolve inside Paths.OUTPUT_DIR — the folder this server itself
    writes generated/rendered audio into. Used for split_vocal_stems'
    audio_path: unlike reference_audio_path (a legitimate way to point at a
    user's own sample library anywhere on disk), the only intended input
    here is a file this server already produced via generate_vocal_track,
    so there is no legitimate use case for a broader path and no reason to
    give a calling model the ability to make this server open arbitrary
    files elsewhere on the filesystem.
    """
    resolved = validate_audio_file_path(path, param_name=param_name)

    allowed_root = os.path.realpath(Paths.OUTPUT_DIR)
    try:
        inside_output_dir = os.path.commonpath([resolved, allowed_root]) == allowed_root
    except ValueError:
        inside_output_dir = False  # different drives on Windows, etc.
    if not inside_output_dir:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"{param_name} must be a file inside {allowed_root} (the output folder this "
            f"server writes generated audio into) — got {path}",
        )

    return resolved


def validate_output_dir_midi_path(path: str, *, param_name: str) -> str:
    """Like validate_output_dir_audio_path, but for a .mid file this
    server itself produced (transcribe_instrumental_to_midi's outputs) —
    refuses anything outside Paths.OUTPUT_DIR, requires a .mid extension,
    and confirms it actually parses as MIDI rather than trusting the
    extension alone."""
    if not path or not str(path).strip():
        raise SongForgeMCPError(ErrorCode.MISSING_PARAMETER, f"{param_name} must not be empty")

    resolved = os.path.realpath(path)
    if not os.path.isfile(resolved):
        raise SongForgeMCPError(
            ErrorCode.FILE_NOT_FOUND, f"{param_name} does not exist or is not a file: {path}"
        )

    if os.path.splitext(resolved)[1].lower() != ".mid":
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER, f"{param_name} must have a .mid extension, got {path!r}"
        )

    allowed_root = os.path.realpath(Paths.OUTPUT_DIR)
    try:
        inside_output_dir = os.path.commonpath([resolved, allowed_root]) == allowed_root
    except ValueError:
        inside_output_dir = False
    if not inside_output_dir:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"{param_name} must be a file inside {allowed_root} (the output folder this "
            f"server writes generated files into) — got {path}",
        )

    try:
        pretty_midi.PrettyMIDI(resolved)
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.INVALID_PARAMETER,
            f"{param_name} does not appear to be a valid MIDI file: {e}",
        ) from e

    return resolved


def measure_wav_duration_seconds(wav_path: str) -> float:
    """Reads duration via soundfile (libsndfile), not the stdlib `wave`
    module. `wave` only understands canonical integer-PCM WAV (format
    code 1) and raises "unknown format" on anything else — real problem,
    not theoretical: ACE-Step's "WAV (16-bit)" output was observed
    actually landing as 32-bit float PCM (format code 3) on disk, which
    `wave.open` cannot parse at all. soundfile handles float/extensible
    WAV (and every other real-world variant) correctly."""
    try:
        info = sf.info(wav_path)
        return info.frames / info.samplerate
    except Exception as e:
        raise SongForgeMCPError(
            ErrorCode.SYNTHESIS_FAILED, f"could not read rendered audio at {wav_path}: {e}"
        ) from e


