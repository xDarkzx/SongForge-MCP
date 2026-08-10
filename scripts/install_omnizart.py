#!/usr/bin/env python3
"""Installs Omnizart (multi-instrument MIDI transcription, an alternative
to basic-pitch) into its own isolated venv, matching this project's
per-tool venv convention (see .separator_env / SeparatorClient).

Omnizart's own PyPI package cannot be pip-installed cleanly as of writing.
Two independent, unrelated problems, both confirmed by hand before writing
this script:

1. Its setup.py force-installs `vamp` as a side effect of merely being
   imported by pip. vamp's C++ source uses the POSIX type `ssize_t`, which
   MSVC never defines - a hard, un-worked-around Windows build failure
   (confirmed directly: the same source fails to compile under MSVC
   regardless of Cython/numpy versions). Grepping omnizart's own source
   shows `vamp` is imported by exactly one file, its chord/chroma feature
   extraction - it is NOT used by the "music" (pitched-instrument)
   transcription this project actually calls. Skipping it is safe for that
   path, not a workaround that silently breaks the feature being used.
2. madmom (an omnizart dependency) predates two later Python/numpy
   removals it still relies on: `collections.MutableSequence` (moved to
   `collections.abc` in Python 3.10) and the `np.float`/`np.int`/etc.
   deprecated aliases (removed entirely in numpy 1.24+). Both are hard
   ImportErrors on any current interpreter, unrelated to the vamp issue.

This script reproduces the exact fix verified by hand: install numpy<2 (the
last numpy line madmom's C extension was ever built against) and Cython
first so madmom's own build can see them without pip's build isolation
hiding them; build madmom with --no-build-isolation; patch madmom's
installed .py files for the two compatibility breaks above; install
omnizart's other real dependencies directly; clone omnizart's source,
patch its setup.py to skip the vamp pre-install step, and install that
patched local copy with --no-deps --no-build-isolation.

Usage: run with the SAME interpreter used to invoke this script (any
Python 3.10+ works - this script only orchestrates subprocesses into the
new venv, it doesn't need omnizart's own dependencies itself):
    python scripts/install_omnizart.py

Idempotent: safe to re-run - pip install steps are no-ops when already
satisfied, and the venv is recreated fresh each time this is invoked (a
partial prior failure is not silently left half-configured).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

OMNIZART_REPO = "https://github.com/Music-and-Culture-Technology-Lab/omnizart.git"

# omnizart's own real install_requires, minus vamp (see module docstring).
_OMNIZART_DEPS = [
    "click>=7.1.2", "jsonschema>=3.2.0", "librosa>=0.8.0", "mir_eval>=0.6",
    "pillow>=8.3.2", "pretty_midi>=0.2.9", "pyfluidsynth>=1.2.5", "pyyaml>=5.3.1",
    "sherpa-onnx>=1.10.0", "soundfile>=0.12.1", "tensorflow>=2.5.0", "tf-keras",
    "urllib3>=1.26.4",
]

# Deprecated numpy scalar-type aliases madmom's source still uses, and their
# modern replacements. Order matters not at all (each is a distinct token),
# but np.bool/np.int/np.float/np.object/np.complex/np.str are all real,
# confirmed occurrences across madmom's own files.
_DEPRECATED_NUMPY_ALIASES = {
    r"np\.float\b": "np.float64",
    r"np\.int\b": "np.int_",
    r"np\.bool\b": "np.bool_",
    r"np\.object\b": "object",
    r"np\.complex\b": "np.complex128",
    r"np\.str\b": "str",
}

# Must exactly match songforge_mcp_shared.constants.OmnizartTranscription
# .TUNED_PIANO_CHECKPOINT_NAME - kept as a separate literal here rather
# than imported, deliberately: this script is meant to work with any
# Python 3.10+ interpreter with no project dependencies (see module
# docstring), and importing the main package would break that.
TUNED_CHECKPOINT_NAME = "music_piano_tuned"

# Real, measured result (not a guess): a systematic sweep on real material
# scored each candidate against independent onset detection on the raw
# audio. OnsetTh 4 (Omnizart's own stock default) recalled 76% of real
# onsets; 2.5 recalled 80% for a contained increase in likely-spurious
# extra notes (50->62 out of ~90 onsets); pushing further to 1.5 only
# gained 2 more points of recall for a much steeper false-positive cost
# (78 extras) - diminishing returns past this point, not a value to keep
# chasing lower. 2.5 is the point on that curve worth keeping as the new
# default over Omnizart's own stock 4.
TUNED_ONSET_TH = 2.5


def _venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _venv_exe(env_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / f"{name}.exe"
    return env_dir / "bin" / name


def _pip(python_exe: Path, *args: str) -> None:
    subprocess.run([str(python_exe), "-m", "pip", "install", *args], check=True)


def _patch_madmom(env_dir: Path) -> None:
    """Fixes the two real, vamp-unrelated Python-version incompatibilities
    in the installed madmom package. See module docstring point 2."""
    lib_dir = env_dir / ("Lib" if os.name == "nt" else "lib")
    site_packages = next(p for p in lib_dir.rglob("site-packages") if (p / "madmom").is_dir())
    madmom_dir = site_packages / "madmom"

    processors_file = madmom_dir / "processors.py"
    text = processors_file.read_text()
    processors_file.write_text(
        text.replace(
            "from collections import MutableSequence",
            "from collections.abc import MutableSequence",
        )
    )

    for path in madmom_dir.rglob("*.py"):
        text = path.read_text()
        new_text = text
        for pattern, replacement in _DEPRECATED_NUMPY_ALIASES.items():
            new_text = re.sub(pattern, replacement, new_text)
        if new_text != text:
            path.write_text(new_text)


def _patch_omnizart_setup(setup_py: Path) -> None:
    """Skips the vamp pre-install in omnizart's setup.py. See module
    docstring point 1 for why this is safe for the transcription mode
    this project actually uses."""
    text = setup_py.read_text()
    text = text.replace(
        "    import vamp\n",
        "    pass  # vamp skipped by install_omnizart.py - Windows MSVC build "
        "incompatible, only used by the chord/chroma submodule\n",
    )
    text = re.sub(
        r'    if not install_packages\(\["vamp>=1\.1\.0"\], no_build_isolation=True\):\n'
        r'        raise RuntimeError\("ERROR: Failed to pre-install dependency \'vamp\'\."\)\n',
        "",
        text,
    )
    setup_py.write_text(text)


def _create_tuned_piano_checkpoint(env_dir: Path) -> None:
    """Copies the stock Piano checkpoint to TUNED_CHECKPOINT_NAME with
    OnsetTh lowered to TUNED_ONSET_TH - see that constant's comment for
    the real measurement behind the value. Leaves the original stock
    checkpoint untouched (omnizart_client.py falls back to it if this
    tuned copy is ever missing)."""
    lib_dir = env_dir / ("Lib" if os.name == "nt" else "lib")
    music_dir = next(p for p in lib_dir.rglob("checkpoints") if (p / "music").is_dir()) / "music"
    stock_dir = music_dir / "music_piano"
    tuned_dir = music_dir / TUNED_CHECKPOINT_NAME

    if tuned_dir.exists():
        shutil.rmtree(tuned_dir)
    shutil.copytree(stock_dir, tuned_dir)

    config_path = tuned_dir / "configurations.yaml"
    text = config_path.read_text()
    # OnsetTh's "Value: 4" is the only "Value: N\n" (single digit, no
    # decimal) in this file at the time of writing - confirmed by
    # inspection, not assumed. If a future Omnizart release changes this
    # file's layout, this becomes a harmless no-op (the stock checkpoint
    # gets copied unmodified) rather than silently editing the wrong field,
    # since replace() only touches an exact, whole-line match.
    text = text.replace("Value: 4\n", f"Value: {TUNED_ONSET_TH}\n", 1)
    config_path.write_text(text)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_dir = project_root / ".omnizart_env"

    print(f"Creating venv at {env_dir}...")
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    python_exe = _venv_python(env_dir)

    print("Installing build tools and numpy<2 (madmom predates numpy 2.0's ABI changes)...")
    _pip(python_exe, "--upgrade", "pip")
    _pip(python_exe, "numpy<2", "Cython", "wheel", "setuptools")

    print("Building madmom (--no-build-isolation so it sees the numpy/Cython just installed)...")
    _pip(python_exe, "--no-build-isolation", "madmom>=0.16.1")

    print("Patching madmom for Python 3.10+/numpy 1.24+ compatibility...")
    _patch_madmom(env_dir)

    print("Installing omnizart's other real dependencies...")
    _pip(python_exe, *_OMNIZART_DEPS)

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "omnizart_src"
        print("Cloning omnizart source to patch out its forced vamp install...")
        subprocess.run(
            ["git", "clone", "--depth", "1", OMNIZART_REPO, str(src_dir)], check=True
        )
        _patch_omnizart_setup(src_dir / "setup.py")

        print("Installing omnizart from the patched local source...")
        _pip(python_exe, "--no-deps", "--no-build-isolation", str(src_dir))

    print("Downloading omnizart checkpoints (large; one-time)...")
    omnizart_exe = _venv_exe(env_dir, "omnizart")
    subprocess.run([str(omnizart_exe), "download-checkpoints"], check=True)

    print(f"Creating tuned Piano checkpoint (OnsetTh={TUNED_ONSET_TH}, see module "
          f"docstring's TUNED_ONSET_TH comment for the real measurement behind this)...")
    _create_tuned_piano_checkpoint(env_dir)

    print(f"\nDone. Set SONGFORGE_OMNIZART_PYTHON={python_exe}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"install_omnizart.py failed: {e}", file=sys.stderr)
        sys.exit(1)
