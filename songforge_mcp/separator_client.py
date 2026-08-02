"""Vocal/instrumental stem separation via audio-separator (Roformer models).

`Separator.DEFAULT_MODEL` (see songforge_mcp_shared/constants.py) picks
the model — overridable per-call via `separate(model_filename=...)` and
per-environment via SONGFORGE_SEPARATOR_MODEL, since SDR benchmarks
aren't genre-specific and the right model for dense/loud mixes (heavy
synths/distortion masking vocal frequencies) may not be the best
general-purpose one. Previously hardcoded to audio-separator's own
default (model_bs_roformer_ep_317_sdr_12.9755.ckpt) — real-world reports
of vocals still bleeding into the instrumental stem on dense mixes
prompted moving to a configurable, higher-SDR default instead. This wraps
the `audio-separator` CLI rather than importing it directly, matching
this project's general pattern of subprocessing into external,
independently-versioned tools instead of vendoring them as library
dependencies.

Runs from a synchronous method (`separate`) — callers on the asyncio side
must dispatch it via `asyncio.to_thread` (see `tools/separate_tools.py`),
same reasoning as any blocking subprocess call from an async tool.
"""
import os
import platform
import re
import subprocess
import threading
from pathlib import Path

from songforge_mcp_shared.constants import Paths, Separator, Timeouts, ensure_private_dir, no_window_popen_kwargs
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_MODEL_TAG_RE = re.compile(r"[^A-Za-z0-9]+")


def _model_tag(model_filename: str) -> str:
    """Filesystem-safe tag identifying which model produced a stem file,
    so cached output from one model is never mistaken for another's."""
    return _MODEL_TAG_RE.sub("_", model_filename).strip("_")


_STEM_LABEL_RE = re.compile(r"\(([^)]+)\)")


def _find_stem_output(out_dir: str, stem: str) -> tuple[Path | None, Path | None]:
    """Returns (vocals_path, instrumental_path) among {stem}_*.wav files
    in out_dir, or (None, None) if not both present.

    Confirmed the hard way: audio-separator's stem-label naming is not
    consistent across models. The old default
    (model_bs_roformer_ep_317_sdr_12.9755.ckpt) writes
    "..._(Vocals)_..."/"..._(Instrumental)_...". The new default
    (vocals_mel_band_roformer.ckpt, a single-target vocal-isolation model)
    writes lowercase "..._(vocals)_..."/"..._(other)_..." instead -
    matching on "(Vocals)"/"(Instrumental)" literally missed every file
    from the new model.

    A first fix attempt classified by whether "vocal" appears anywhere in
    the filename - also wrong, and in a sneakier way: this model's own
    filename (vocals_mel_band_roformer.ckpt) gets appended to BOTH output
    files as a suffix, so "vocal" matched both and nothing was left for
    instrumental. Extracting specifically the parenthesized stem label
    (the "(vocals)"/"(other)"/"(Instrumental)" part) and classifying only
    that avoids false-matching the model name."""
    candidates = list(Path(out_dir).glob(f"{stem}_*.wav"))
    vocals = []
    instrumental = []
    for f in candidates:
        label_match = _STEM_LABEL_RE.search(f.stem)
        label = label_match.group(1).lower() if label_match else ""
        (vocals if "vocal" in label else instrumental).append(f)
    if not vocals or not instrumental:
        return None, None
    return vocals[0], instrumental[0]


def _find_labeled_stems(out_dir: str, stem: str) -> dict[str, Path]:
    """Returns {label.lower(): path} for every {stem}_(Label)_*.wav file
    found in out_dir - e.g. {"drums": ..., "bass": ..., "guitar": ...}.
    Unlike _find_stem_output (which expects exactly two known buckets,
    vocals vs. everything else), this doesn't assume which labels a
    model produces - Demucs models vary (4-stem vs. 6-stem), and the
    caller shouldn't need to hardcode that per model. Returns {} if no
    labeled files are found at all."""
    candidates = list(Path(out_dir).glob(f"{stem}_*.wav"))
    stems: dict[str, Path] = {}
    for f in candidates:
        label_match = _STEM_LABEL_RE.search(f.stem)
        if label_match:
            stems[label_match.group(1).lower()] = f
    return stems

_SEPARATOR_EXE_NAME = "audio-separator.exe" if platform.system() == "Windows" else "audio-separator"

# `audio-separator --list_models` prints a fixed-width table; columns are
# separated by runs of 2+ spaces, which single spaces inside e.g.
# "vocals* (12.1), instrumental (16.3)" never satisfy - this also
# naturally skips the header/separator lines (their column gaps are only
# 1 space or none), no special-casing needed.
_LIST_MODELS_ROW_RE = re.compile(r"^(\S+)\s{2,}(\S+)\s{2,}(.+?)\s{2,}(.+)$")
_VOCALS_SDR_RE = re.compile(r"vocals\*?\s*\(([\d.]+)\)")


class SeparatorClient:
    def __init__(self, separator_venv_python: str | None = None):
        self.python_exe = separator_venv_python or os.environ.get(
            "SONGFORGE_SEPARATOR_PYTHON", ""
        )
        ensure_private_dir(Paths.OUTPUT_DIR)
        # audio-separator uses onnxruntime-gpu against the same GPU
        # ACE-Step runs on - a plain threading.Lock (not asyncio.Lock,
        # since this class is driven via asyncio.to_thread rather than
        # natively async) keeps two separation calls from competing for
        # GPU memory at once.
        self._lock = threading.Lock()

    def _require_configured(self) -> str:
        """Returns the resolved path to the audio-separator console
        script living alongside the configured Python interpreter."""
        if not self.python_exe or not os.path.isfile(self.python_exe):
            raise SongForgeMCPError(
                ErrorCode.SEPARATOR_NOT_CONFIGURED,
                "SONGFORGE_SEPARATOR_PYTHON is not set or does not point to a "
                "valid Python interpreter with audio-separator installed. See docs/INSTALLATION.md.",
            )
        separator_exe = os.path.join(os.path.dirname(self.python_exe), _SEPARATOR_EXE_NAME)
        if not os.path.isfile(separator_exe):
            raise SongForgeMCPError(
                ErrorCode.SEPARATOR_NOT_CONFIGURED,
                f"{_SEPARATOR_EXE_NAME} not found alongside {self.python_exe} — "
                "is audio-separator actually installed in that environment?",
            )
        return separator_exe

    def separate(self, audio_path: str, model_filename: str | None = None) -> dict:
        """Returns {"vocals_path": str, "instrumental_path": str}.

        model_filename: audio-separator model to use (see
        `audio-separator --list_models`). Defaults to
        Separator.DEFAULT_MODEL. Different models handle dense/loud mixes
        differently — pass Separator.ALT_MODEL (or any other listed
        model) to A/B test against the default.

        Idempotent per (source file, model) pair: if this exact source
        was already separated with this exact model (matching output
        files already exist in the stems folder), returns those
        immediately without re-running the separator. A real, recurring
        failure mode is the calling model losing track of an earlier
        result over a long conversation and asking to split the same
        file again — there is no reason to burn GPU time and wait
        through a real separation run twice for the same (source, model)
        pair. Testing a *different* model against the same source always
        re-runs, by design — that's the whole point of A/B testing."""
        separator_exe = self._require_configured()
        if not os.path.isfile(audio_path):
            raise SongForgeMCPError(
                ErrorCode.FILE_NOT_FOUND, f"audio_path does not exist: {audio_path}"
            )
        model_filename = model_filename or Separator.DEFAULT_MODEL

        # A subfolder per model, rather than trying to match the model
        # name back out of audio-separator's own output filenames (its
        # naming convention for --model_filename output is not something
        # this codebase controls or has verified) - scoping by directory
        # is unambiguous regardless of what audio-separator names files.
        model_tag = _model_tag(model_filename)
        out_dir = os.path.join(Paths.OUTPUT_DIR, "stems", model_tag)
        ensure_private_dir(out_dir)
        stem = Path(audio_path).stem

        existing_vocals, existing_instrumental = _find_stem_output(out_dir, stem)
        if existing_vocals and existing_instrumental:
            return {
                "vocals_path": str(existing_vocals),
                "instrumental_path": str(existing_instrumental),
            }

        with self._lock:
            try:
                result = subprocess.run(
                    [
                        separator_exe, audio_path,
                        "--output_dir", out_dir,
                        "--output_format", "wav",
                        "--model_filename", model_filename,
                    ],
                    capture_output=True, text=True, timeout=Timeouts.SEPARATION, check=False,
                    **no_window_popen_kwargs(),
                )
            except subprocess.TimeoutExpired as e:
                raise SongForgeMCPError(
                    ErrorCode.SUBPROCESS_TIMEOUT, f"separation exceeded {Timeouts.SEPARATION}s"
                ) from e

        if result.returncode != 0:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"audio-separator exited {result.returncode}: {result.stderr.strip()[-2000:]}",
            )

        vocals, instrumental = _find_stem_output(out_dir, stem)
        if not vocals or not instrumental:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"separation reported success but expected output files were not found in {out_dir}",
            )
        return {"vocals_path": str(vocals), "instrumental_path": str(instrumental)}

    def separate_extra_stems(self, audio_path: str, model_filename: str = "htdemucs_6s.yaml") -> dict:
        """Returns {"{label}_path": str, ...} for every non-vocal stem
        the given model produces - e.g. htdemucs_6s.yaml (the only
        model in audio-separator's catalog that outputs guitar/piano)
        gives drums_path/bass_path/guitar_path/piano_path/other_path;
        htdemucs_ft.yaml gives drums_path/bass_path/other_path only.
        The vocals stem this model also produces is always dropped -
        call separate() for vocals instead, which uses a dedicated
        model with meaningfully higher vocal SDR.

        Runs on audio_path directly (the original full mix), never on
        an already vocal-separated file - Demucs models are trained on
        full mixes with vocals present, so feeding one a vocal-stripped
        file would be out-of-domain input with unpredictable quality
        impact on the remaining stems.

        Idempotent per (source file, model) pair, same reasoning as
        separate(): if at least two non-vocal stems already exist for
        this exact (audio_path, model_filename) pair, returns those
        without re-running. The "at least two" floor (rather than
        exactly matching the model's full expected stem count, which
        this method deliberately doesn't hardcode) still catches an
        obviously incomplete previous run - e.g. a process killed
        mid-write leaving a single stray file - without needing to know
        in advance how many stems a given model produces."""
        separator_exe = self._require_configured()
        if not os.path.isfile(audio_path):
            raise SongForgeMCPError(
                ErrorCode.FILE_NOT_FOUND, f"audio_path does not exist: {audio_path}"
            )

        model_tag = _model_tag(model_filename)
        out_dir = os.path.join(Paths.OUTPUT_DIR, "stems", model_tag)
        ensure_private_dir(out_dir)
        stem = Path(audio_path).stem

        existing = {k: v for k, v in _find_labeled_stems(out_dir, stem).items() if k != "vocals"}
        if len(existing) >= 2:
            return {f"{label}_path": str(path) for label, path in existing.items()}

        with self._lock:
            try:
                result = subprocess.run(
                    [
                        separator_exe, audio_path,
                        "--output_dir", out_dir,
                        "--output_format", "wav",
                        "--model_filename", model_filename,
                    ],
                    capture_output=True, text=True, timeout=Timeouts.SEPARATION, check=False,
                    **no_window_popen_kwargs(),
                )
            except subprocess.TimeoutExpired as e:
                raise SongForgeMCPError(
                    ErrorCode.SUBPROCESS_TIMEOUT, f"separation exceeded {Timeouts.SEPARATION}s"
                ) from e

        if result.returncode != 0:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"audio-separator exited {result.returncode}: {result.stderr.strip()[-2000:]}",
            )

        stems = {k: v for k, v in _find_labeled_stems(out_dir, stem).items() if k != "vocals"}
        if len(stems) < 2:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"separation reported success but expected stem output files were not found in {out_dir}",
            )
        return {f"{label}_path": str(path) for label, path in stems.items()}

    def list_models(self, vocal_only: bool = True) -> list[dict]:
        """Returns audio-separator's model catalog, sorted by vocal SDR
        descending (models with no numeric vocal score sort last). Each
        entry: {"filename": str, "arch": str, "scores": str,
        "friendly_name": str, "vocal_sdr": float | None}. `filename` is
        what to pass as `separate(model_filename=...)` - any model in
        this list works, not just Separator.DEFAULT_MODEL/ALT_MODEL.

        vocal_only=True (default) excludes models with no vocal-stem
        score at all (denoise/deverb/crowd-removal models etc. that don't
        do a vocal/instrumental split). Set False to see the full catalog."""
        separator_exe = self._require_configured()
        try:
            result = subprocess.run(
                [separator_exe, "--list_models"],
                capture_output=True, text=True, timeout=60, check=False,
                **no_window_popen_kwargs(),
            )
        except subprocess.TimeoutExpired as e:
            raise SongForgeMCPError(
                ErrorCode.SUBPROCESS_TIMEOUT, "audio-separator --list_models exceeded 60s"
            ) from e
        if result.returncode != 0:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"audio-separator --list_models exited {result.returncode}: {result.stderr.strip()[-1000:]}",
            )

        models = []
        for line in result.stdout.splitlines():
            match = _LIST_MODELS_ROW_RE.match(line.rstrip())
            if not match:
                continue
            filename, arch, scores, friendly_name = (g.strip() for g in match.groups())
            sdr_match = _VOCALS_SDR_RE.search(scores)
            vocal_sdr = float(sdr_match.group(1)) if sdr_match else None
            if vocal_only and vocal_sdr is None:
                continue
            models.append({
                "filename": filename,
                "arch": arch,
                "scores": scores,
                "friendly_name": friendly_name,
                "vocal_sdr": vocal_sdr,
            })
        models.sort(key=lambda m: (m["vocal_sdr"] is None, -(m["vocal_sdr"] or 0.0)))
        return models
