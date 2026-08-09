# Install-time GPU VRAM Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `install.bat`/`install.sh` automatically detect the user's GPU VRAM and set `SONGFORGE_ACESTEP_CHECKPOINT` to the right-sized ACE-Step checkpoint, instead of always downloading the ~28GB XL-SFT checkpoint regardless of hardware — widening the supported floor from ~12GB down to ~6GB.

**Architecture:** One shared, pure-function-testable Python script (`scripts/detect_gpu_checkpoint.py`) encodes the VRAM→checkpoint mapping once. Both installers invoke it via ACE-Step's own venv python (the only interpreter guaranteed to have `torch`) right after that venv is set up, and only act on the result if `SONGFORGE_ACESTEP_CHECKPOINT` isn't already set.

**Tech Stack:** Python (stdlib `sys` only — no new dependency), pytest with `unittest.mock` for torch mocking, batch (`install.bat`), bash (`install.sh`).

## Global Constraints

- Never overwrite an existing `SONGFORGE_ACESTEP_CHECKPOINT` — only set it if unset.
- Any detection failure (no GPU, no CUDA, no torch, any error) must leave behavior identical to today: nothing gets set, the existing hardcoded `acestep-v15-xl-sft` default in `songforge_mcp_shared/constants.py` (unchanged by this plan) is what actually gets used.
- Mapping (from `docs/2026-08-03-gpu-checkpoint-detection-design.md`): `<8GB` → `acestep-v15-turbo`, `8-20GB` → `acestep-v15-sft`, `≥20GB` → `acestep-v15-xl-sft`.
- `detect_gpu_checkpoint.py`'s contract: prints **only** the checkpoint name to stdout and exits 0 on success; prints nothing to stdout (diagnostics to stderr only) and exits non-zero on any failure.

---

## Task 1: `scripts/detect_gpu_checkpoint.py` + tests

**Files:**
- Create: `scripts/__init__.py` (empty — makes `scripts` importable from tests; this project's `pyproject.toml` already has `pythonpath = ["."]`)
- Create: `scripts/detect_gpu_checkpoint.py`
- Test: `tests/test_detect_gpu_checkpoint.py`

**Interfaces:**
- Produces: `pick_checkpoint(vram_gb: float) -> str` (pure function, no I/O). `detect_vram_gb() -> float` (raises `RuntimeError` on any detection failure; lazily `import torch` inside the function body — this project's own `.venv` has no `torch` dependency at all, so a module-level import would break every test that imports this module). `main() -> int` (0/1 exit code, per the contract above). Tasks 2/3 invoke this script as a subprocess (`python scripts/detect_gpu_checkpoint.py`), not by importing it — they only need to know its CLI contract (stdout/exit-code), not these function names.

- [ ] **Step 1: Write the failing tests**

Create `scripts/__init__.py` (empty file — just needs to exist).

Create `tests/test_detect_gpu_checkpoint.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from scripts.detect_gpu_checkpoint import detect_vram_gb, main, pick_checkpoint


@pytest.mark.parametrize(
    "vram_gb,expected",
    [
        (2.0, "acestep-v15-turbo"),
        (6.0, "acestep-v15-turbo"),
        (7.9, "acestep-v15-turbo"),
        (8.0, "acestep-v15-sft"),
        (12.0, "acestep-v15-sft"),
        (19.9, "acestep-v15-sft"),
        (20.0, "acestep-v15-xl-sft"),
        (24.0, "acestep-v15-xl-sft"),
        (48.0, "acestep-v15-xl-sft"),
    ],
)
def test_pick_checkpoint_boundaries(vram_gb, expected):
    assert pick_checkpoint(vram_gb) == expected


def test_detect_vram_gb_returns_total_memory_in_gb():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_props = MagicMock()
    fake_props.total_memory = 8 * 1024**3
    fake_torch.cuda.get_device_properties.return_value = fake_props

    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert detect_vram_gb() == pytest.approx(8.0)


def test_detect_vram_gb_raises_when_no_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="no CUDA-capable GPU"):
            detect_vram_gb()


def test_detect_vram_gb_raises_when_torch_not_installed():
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(ImportError):
            detect_vram_gb()


def test_main_prints_checkpoint_to_stdout_and_returns_zero(capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_props = MagicMock()
    fake_props.total_memory = 6 * 1024**3
    fake_torch.cuda.get_device_properties.return_value = fake_props

    with patch.dict("sys.modules", {"torch": fake_torch}):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "acestep-v15-turbo"


def test_main_returns_nonzero_and_prints_nothing_to_stdout_on_failure(capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "detection failed" in captured.err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detect_gpu_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.detect_gpu_checkpoint'` (the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `scripts/detect_gpu_checkpoint.py`:

```python
#!/usr/bin/env python3
"""Detects available NVIDIA GPU VRAM and prints the recommended ACE-Step
DiT checkpoint name for it, per
docs/2026-08-03-gpu-checkpoint-detection-design.md's mapping table.

Run via ACE-Step's own venv python specifically (e.g.
".venv/bin/python scripts/detect_gpu_checkpoint.py" from the ACE-Step
checkout) - it's the only interpreter this project guarantees has torch
installed. This project's own venv does NOT depend on torch at all,
which is exactly why every torch access here is a lazy, in-function
import rather than a module-level one - a module-level import would
break importing this module under this project's own test suite.

Contract: prints ONLY the checkpoint name to stdout and exits 0 on
success. Prints nothing to stdout (diagnostics go to stderr only) and
exits non-zero on any failure (no CUDA, no torch, any error) - callers
must branch on exit code, not by parsing stdout for an error string,
and must leave behavior unchanged on failure rather than guessing.
"""
import sys


def pick_checkpoint(vram_gb: float) -> str:
    """Pure mapping function: VRAM (GB) -> recommended ACE-Step DiT
    checkpoint name. Thresholds and reasoning:
    docs/2026-08-03-gpu-checkpoint-detection-design.md."""
    if vram_gb < 8:
        return "acestep-v15-turbo"
    if vram_gb < 20:
        return "acestep-v15-sft"
    return "acestep-v15-xl-sft"


def detect_vram_gb() -> float:
    """Returns total VRAM (GB) of GPU 0. Raises RuntimeError if no CUDA
    GPU is available, or ImportError if torch itself isn't installed in
    the interpreter running this script."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA-capable GPU detected")
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def main() -> int:
    try:
        vram_gb = detect_vram_gb()
    except Exception as e:
        print(f"GPU VRAM detection failed: {e}", file=sys.stderr)
        return 1

    checkpoint = pick_checkpoint(vram_gb)
    print(f"Detected {vram_gb:.1f}GB VRAM -> recommending {checkpoint}", file=sys.stderr)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detect_gpu_checkpoint.py -v`
Expected: PASS — all 9 tests.

- [ ] **Step 5: Verify against real GPU hardware**

This machine already has a real ACE-Step-1.5 checkout with a working CUDA
torch install (`C:\DansProject\voice-production\ACE-Step-1.5\.venv`) —
use it to confirm the script works end-to-end against real hardware, not
just mocks, before trusting it in an installer:

Run: `"C:\DansProject\voice-production\ACE-Step-1.5\.venv\Scripts\python.exe" scripts\detect_gpu_checkpoint.py`
(run from the `Vocal-Synth-MCP` directory)

Expected: prints a `Detected N.NGB VRAM -> recommending ...` line to
stderr and a bare checkpoint name to stdout, matching this machine's
actual GPU. Confirm the printed VRAM figure is plausible for the real
card in this machine before moving on — if it looks wrong, stop and
investigate before Task 2/3 wire this into the installers.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/detect_gpu_checkpoint.py tests/test_detect_gpu_checkpoint.py
git commit -m "Add GPU VRAM -> ACE-Step checkpoint detection script

Pure pick_checkpoint() mapping function plus a thin CLI wrapper
(detect_vram_gb/main) that installers will call as a subprocess.
Verified against real GPU hardware, not just mocks."
```

---

## Task 2: Wire into `install.bat`

**Files:**
- Modify: `install.bat:57-76` (ACE-Step venv setup block) and `install.bat:166-179` (final summary block)

**Interfaces:**
- Consumes: `scripts\detect_gpu_checkpoint.py`'s CLI contract from Task 1 (stdout = checkpoint name on success, exit code signals success/failure) — invoked as a subprocess via `".venv\Scripts\python.exe"`, not imported.

- [ ] **Step 1: Add detection right after the ACE-Step venv's pip installs, before `popd`**

In `install.bat`, find:

```batch
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install -e "acestep\third_parts\nano-vllm"
".venv\Scripts\python.exe" -m pip install -e . --no-deps
popd

setx SONGFORGE_ACESTEP_HOME "%ACESTEP_DEST%"
set "SONGFORGE_ACESTEP_HOME=%ACESTEP_DEST%"
```

Replace with:

```batch
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install -e "acestep\third_parts\nano-vllm"
".venv\Scripts\python.exe" -m pip install -e . --no-deps

REM ---- GPU VRAM detection: pick the right-sized ACE-Step checkpoint for
REM this hardware, unless the user already configured one explicitly.
REM Never overwrites an existing SONGFORGE_ACESTEP_CHECKPOINT. Detection
REM failure (no GPU, non-NVIDIA, any error) changes nothing - falls back
REM to today's exact default, set in songforge_mcp_shared/constants.py.
if "%SONGFORGE_ACESTEP_CHECKPOINT%"=="" (
    set "DETECTED_CHECKPOINT="
    for /f "delims=" %%C in ('".venv\Scripts\python.exe" "%~dp0scripts\detect_gpu_checkpoint.py"') do set "DETECTED_CHECKPOINT=%%C"
    if not "!DETECTED_CHECKPOINT!"=="" (
        echo Detected GPU - using ACE-Step checkpoint: !DETECTED_CHECKPOINT!
        setx SONGFORGE_ACESTEP_CHECKPOINT "!DETECTED_CHECKPOINT!"
        set "SONGFORGE_ACESTEP_CHECKPOINT=!DETECTED_CHECKPOINT!"
    ) else (
        echo GPU VRAM detection unavailable - using ACE-Step's default checkpoint selection.
    )
)
popd

setx SONGFORGE_ACESTEP_HOME "%ACESTEP_DEST%"
set "SONGFORGE_ACESTEP_HOME=%ACESTEP_DEST%"
```

(`%~dp0` still correctly resolves to this batch file's own directory here
even while `pushd`'d into `%ACESTEP_DEST%` — it's a static property of how
the script was invoked, unaffected by `pushd`/`cd`. `!DETECTED_CHECKPOINT!`
delayed-expansion syntax is safe here — this script already uses it
elsewhere, e.g. `!ACESTEP_DEST!` a few lines earlier, confirming
`setlocal enabledelayedexpansion` is already active.)

- [ ] **Step 2: Show the checkpoint in the final summary, only if set**

In `install.bat`, find:

```batch
echo   SONGFORGE_ACESTEP_HOME     = %ACESTEP_DEST%
echo   SONGFORGE_SEPARATOR_PYTHON = %SEPARATOR_HOME%\Scripts\python.exe
echo   SONGFORGE_YTDLP_PYTHON     = %~dp0.venv\Scripts\python.exe
echo.
```

Replace with:

```batch
echo   SONGFORGE_ACESTEP_HOME     = %ACESTEP_DEST%
echo   SONGFORGE_SEPARATOR_PYTHON = %SEPARATOR_HOME%\Scripts\python.exe
echo   SONGFORGE_YTDLP_PYTHON     = %~dp0.venv\Scripts\python.exe
if not "%SONGFORGE_ACESTEP_CHECKPOINT%"=="" (
    echo   SONGFORGE_ACESTEP_CHECKPOINT = %SONGFORGE_ACESTEP_CHECKPOINT%
)
echo.
```

- [ ] **Step 3: Verify syntax by hand-tracing (no automated test — installer scripts aren't covered by this project's test suite)**

Re-read the full modified `install.bat` top to bottom once. Confirm: every
`(` has a matching `)`, `!DETECTED_CHECKPOINT!` is only used inside the
delayed-expansion-active scope (the whole file, confirmed in Step 1), and
the `for /f` loop's captured variable is reset (`set "DETECTED_CHECKPOINT="`)
before the loop so a prior run's leftover value can never leak through on
failure.

- [ ] **Step 4: Commit**

```bash
git add install.bat
git commit -m "Wire GPU VRAM detection into install.bat

Detects VRAM right after ACE-Step's venv/torch install, sets
SONGFORGE_ACESTEP_CHECKPOINT via setx if not already configured.
Falls back silently to today's default on any detection failure."
```

---

## Task 3: Wire into `install.sh`

**Files:**
- Modify: `install.sh:20-22` (capture project root before any `cd`/`pushd`), `install.sh:98-124` (ACE-Step venv setup block), `install.sh:198-211` (final summary block)

**Interfaces:**
- Consumes: same CLI contract as Task 2, plus this file's own existing `detect_nvidia_gpu()` shell function (`install.sh:71-73`) — reused, not duplicated.

- [ ] **Step 1: Capture the project root before any `pushd`**

In `install.sh`, find:

```bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
```

Replace with:

```bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(pwd)"
```

- [ ] **Step 2: Add detection right after the ACE-Step venv's pip installs, before `popd`**

In `install.sh`, find:

```bash
".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" -m pip install -e "acestep/third_parts/nano-vllm"
".venv/bin/python" -m pip install -e . --no-deps
popd >/dev/null
```

Replace with:

```bash
".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" -m pip install -e "acestep/third_parts/nano-vllm"
".venv/bin/python" -m pip install -e . --no-deps

# ---- GPU VRAM detection: pick the right-sized ACE-Step checkpoint for
# this hardware, unless the user already configured one explicitly.
# Never overwrites an existing SONGFORGE_ACESTEP_CHECKPOINT. Detection
# failure (no GPU, non-NVIDIA, any error) changes nothing - falls back
# to today's exact default, set in songforge_mcp_shared/constants.py.
# "|| true" matters here: this script runs under `set -e`, and without
# it a detection failure (a deliberately non-zero exit, per this
# script's own contract) would abort the whole installer instead of
# falling back gracefully.
if [ -z "${SONGFORGE_ACESTEP_CHECKPOINT:-}" ] && detect_nvidia_gpu; then
    DETECTED_CHECKPOINT="$(".venv/bin/python" "$PROJECT_ROOT/scripts/detect_gpu_checkpoint.py")" || true
    if [ -n "${DETECTED_CHECKPOINT:-}" ]; then
        echo "Detected GPU - using ACE-Step checkpoint: $DETECTED_CHECKPOINT"
        export SONGFORGE_ACESTEP_CHECKPOINT="$DETECTED_CHECKPOINT"
    else
        echo "GPU VRAM detection unavailable - using ACE-Step's default checkpoint selection."
    fi
fi
popd >/dev/null
```

- [ ] **Step 3: Show the checkpoint in the final summary, only if set**

In `install.sh`, find:

```bash
echo "  SONGFORGE_ACESTEP_HOME     = $ACESTEP_DEST"
echo "  SONGFORGE_SEPARATOR_PYTHON = $SEPARATOR_HOME/bin/python"
echo "  SONGFORGE_YTDLP_PYTHON     = $YTDLP_PYTHON"
echo
echo "Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.) and restart"
echo "your terminal / Claude Desktop for them to take effect:"
echo
echo "  export SONGFORGE_ACESTEP_HOME=\"$ACESTEP_DEST\""
echo "  export SONGFORGE_SEPARATOR_PYTHON=\"$SEPARATOR_HOME/bin/python\""
echo "  export SONGFORGE_YTDLP_PYTHON=\"$YTDLP_PYTHON\""
echo
```

Replace with:

```bash
echo "  SONGFORGE_ACESTEP_HOME     = $ACESTEP_DEST"
echo "  SONGFORGE_SEPARATOR_PYTHON = $SEPARATOR_HOME/bin/python"
echo "  SONGFORGE_YTDLP_PYTHON     = $YTDLP_PYTHON"
if [ -n "${SONGFORGE_ACESTEP_CHECKPOINT:-}" ]; then
    echo "  SONGFORGE_ACESTEP_CHECKPOINT = $SONGFORGE_ACESTEP_CHECKPOINT"
fi
echo
echo "Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.) and restart"
echo "your terminal / Claude Desktop for them to take effect:"
echo
echo "  export SONGFORGE_ACESTEP_HOME=\"$ACESTEP_DEST\""
echo "  export SONGFORGE_SEPARATOR_PYTHON=\"$SEPARATOR_HOME/bin/python\""
echo "  export SONGFORGE_YTDLP_PYTHON=\"$YTDLP_PYTHON\""
if [ -n "${SONGFORGE_ACESTEP_CHECKPOINT:-}" ]; then
    echo "  export SONGFORGE_ACESTEP_CHECKPOINT=\"$SONGFORGE_ACESTEP_CHECKPOINT\""
fi
echo
```

- [ ] **Step 4: Verify syntax**

Run: `bash -n install.sh`
Expected: no output (syntax-only check — parses the script without
running it, catches unmatched quotes/brackets/etc.). No GPU-dependent
automated test exists for the real invocation, matching Task 2.

- [ ] **Step 5: Commit**

```bash
git add install.sh
git commit -m "Wire GPU VRAM detection into install.sh

Mirrors install.bat: detects VRAM right after ACE-Step's venv/torch
install (reusing the existing detect_nvidia_gpu check), exports
SONGFORGE_ACESTEP_CHECKPOINT if not already configured, and adds it to
the final export-lines summary. Falls back silently to today's default
on any detection failure - the '|| true' matters under this script's
set -e."
```

---

## Task 4: Update documentation

**Files:**
- Modify: `docs/INSTALLATION.md` (VRAM requirements table)
- Modify: `README.md` (GPU badge)
- Modify: `CHANGELOG.md` (new entry)

**Interfaces:**
- None — this task only changes prose/docs, no code interfaces.

- [ ] **Step 1: Update the VRAM requirements table**

In `docs/INSTALLATION.md`, find:

```
| VRAM | This project is configured to use the **XL-SFT** model (higher quality than the smaller 2B model) — ACE-Step's own docs put that at **≥12GB VRAM with CPU offload + quantization, ≥20GB for it to run without offload** (their GPU_COMPATIBILITY.md tier table calls 12-16GB "marginal" for XL specifically). Below ~12GB VRAM, expect either failures or a very degraded/offloaded configuration — a smaller 2B-class model would be a better fit for that hardware, which isn't what this installer sets up |
```

Replace with:

```
| VRAM | Both installers now detect available VRAM automatically (`scripts/detect_gpu_checkpoint.py`, run right after ACE-Step's own venv/torch install) and pick the right-sized checkpoint for it, setting `SONGFORGE_ACESTEP_CHECKPOINT` accordingly — **below 8GB**: `acestep-v15-turbo` (2B), **8-20GB**: `acestep-v15-sft` (2B), **20GB+**: `acestep-v15-xl-sft` (4B, this project's original hardcoded default). This never overrides a `SONGFORGE_ACESTEP_CHECKPOINT` you've already set yourself, and any detection failure (no GPU, non-NVIDIA, any error) falls back to the XL-SFT default unchanged — see `docs/2026-08-03-gpu-checkpoint-detection-design.md` for the full reasoning behind each threshold |
```

Then find:

```
If your hardware doesn't meet the VRAM guidance above, running ACE-Step
1.5 directly (outside this server) with a smaller model and its own
GPU-tier auto-configuration is a better starting point than trying to
force the XL-SFT setup this installer targets.
```

Replace with:

```
Detection requires an NVIDIA GPU (same as the rest of this installer) —
on AMD/Intel/Apple Silicon/no-GPU hardware, detection fails silently and
you fall back to the XL-SFT default regardless of actual VRAM. Running
ACE-Step 1.5 directly (outside this server) with its own GPU-tier
auto-configuration remains the better starting point for that hardware.
```

- [ ] **Step 2: Update the README GPU badge**

In `README.md`, find:

```
  <a href="https://github.com/ace-step/ACE-Step-1.5"><img src="https://img.shields.io/badge/GPU-12GB%2B%20VRAM-red.svg" alt="GPU 12GB+ VRAM" /></a>
```

Replace with:

```
  <a href="https://github.com/ace-step/ACE-Step-1.5"><img src="https://img.shields.io/badge/GPU-6GB%2B%20VRAM-red.svg" alt="GPU 6GB+ VRAM" /></a>
```

- [ ] **Step 3: Add a CHANGELOG entry**

In `CHANGELOG.md`, add a new entry at the top of the `## Unreleased` section (create that section above `## 0.4.0` if it doesn't already exist — this repo folds it into the version heading at release time, matching how `0.4.0` itself was released):

```markdown
## Unreleased

- **Both installers now auto-detect GPU VRAM and pick the right-sized
  ACE-Step checkpoint** (`scripts/detect_gpu_checkpoint.py`), instead of
  always targeting the ~28GB XL-SFT checkpoint regardless of hardware.
  Confirmed against ACE-Step's own install docs that 8GB cards were
  simply locked out before this — now: below 8GB gets `acestep-v15-turbo`
  (2B), 8-20GB gets `acestep-v15-sft` (2B), 20GB+ keeps the original
  `acestep-v15-xl-sft` (4B) default. Never overrides an explicitly-set
  `SONGFORGE_ACESTEP_CHECKPOINT`, and any detection failure changes
  nothing — falls back to today's exact default.
```

- [ ] **Step 4: Commit**

```bash
git add docs/INSTALLATION.md README.md CHANGELOG.md
git commit -m "Document GPU VRAM auto-detection in installer docs

Updates the VRAM requirements table, README badge (12GB+ -> 6GB+), and
CHANGELOG to reflect scripts/detect_gpu_checkpoint.py."
```

---

## Self-Review Notes

- **Spec coverage:** design doc's install-time-only timing, never-overwrite
  behavior, silent-fallback-on-failure behavior, and the exact VRAM
  mapping table are all covered by Tasks 1-3; the doc/README/CHANGELOG
  updates it calls for are covered by Task 4.
- **Type consistency:** `pick_checkpoint(vram_gb: float) -> str` and
  `detect_vram_gb() -> float` are used identically in Task 1's
  implementation and its tests. Tasks 2/3 only depend on the script's
  CLI contract (stdout/exit code), not its internal function names, so
  there's no signature to drift between them.
- **Explicitly out of scope** (per design doc, not covered by this plan):
  runtime/dynamic re-detection, changing `GradioServer.CHECKPOINT`'s own
  default in `constants.py`, AMD/Intel/Apple Silicon VRAM detection.
