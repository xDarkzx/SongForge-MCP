# Install-time GPU VRAM detection → checkpoint selection — Design

Date: 2026-08-03
Status: Approved through discussion. Ready for implementation planning.

## Motivation

This project hardcodes `SONGFORGE_ACESTEP_CHECKPOINT` to `acestep-v15-xl-sft`
(the XL/4B family), which ACE-Step's own docs put at ≥12GB VRAM realistically
(with CPU offload + quantization). Confirmed against ACE-Step's real
`docs/en/INSTALL.md`: 8GB cards (a common consumer tier — RTX 3060/3070/4060)
are simply locked out today, even though ACE-Step itself has a separate,
smaller 2B checkpoint family (`acestep-v15-base`/`-sft`/`-turbo`) that fits
comfortably down to ≤6GB. The fix isn't new ACE-Step capability — it's this
project actually using the checkpoint choice ACE-Step already supports,
picked automatically instead of requiring a user to know `SONGFORGE_ACESTEP_CHECKPOINT`
exists and which value to set.

## What's reusable vs. what isn't

Checked ACE-Step's own `acestep/gpu_config.py` before designing this, to
avoid reinventing something that already exists:

- **VRAM detection itself** — `check_gpu.py` (ACE-Step's own diagnostic
  script) already demonstrates the exact call:
  `torch.cuda.get_device_properties(0).total_memory / (1024**3)`. Reused
  directly, not reinvented.
- **`GPUConfig`/`get_gpu_config()`** — real, already handles LM model choice
  and runtime settings (offload/quantization/batch size/duration limits)
  for a checkpoint you've *already chosen*. Does **not** recommend which
  DiT checkpoint (XL vs 2B, turbo vs sft) to download in the first place —
  that mapping only exists as a table in ACE-Step's docs, not as a
  callable function. This project has to encode that mapping itself.

## Decisions (resolved through discussion)

- **Runs at install time**, not at server-launch runtime. `install.bat`/
  `install.sh` detect VRAM once, right after ACE-Step's own venv (and its
  pinned CUDA torch) is installed — matches this project's existing
  installer conventions (one-shot setup, same pattern as the separator
  model pre-download step). A hardware upgrade later means re-running the
  installer or setting the env var manually, same as any other installer
  choice this project makes once.
- **Never overwrites an existing `SONGFORGE_ACESTEP_CHECKPOINT`.** Only
  sets it if unset. Re-running the installer is always safe, and anyone
  who's already configured a checkpoint on purpose keeps their choice.
- **Detection failure (no GPU / non-NVIDIA / any error) changes nothing** —
  falls back to today's exact behavior (hardcoded `acestep-v15-xl-sft`
  default), with a printed warning. Nobody's experience gets worse than
  it is today; this is additive only.

## VRAM → checkpoint mapping

From ACE-Step's own `docs/en/INSTALL.md` recommendation table, made
concrete where that table was ambiguous ("turbo/sft" without picking one):

| VRAM | Checkpoint | Reasoning |
|---|---|---|
| <8GB | `acestep-v15-turbo` (2B) | Table's explicit recommendation for ≤8GB; nothing bigger reliably fits |
| 8-20GB | `acestep-v15-sft` (2B) | Table says "turbo/sft" (8-16GB) and "2B sft or XL turbo" (16-20GB) ambiguously — picking **sft** both times, to match this project's own established preference (accepting slower generation for better reference-audio/voice-cloning fidelity — turbo's no-CFG distillation measurably weakens reference-audio conditioning, already documented in `songforge_mcp_shared/constants.py`'s `GradioServer.CHECKPOINT` comment for the exact same tradeoff at the XL tier) |
| ≥20GB | `acestep-v15-xl-sft` | Matches the existing default — unchanged for the hardware tier it already works well on |

## Implementation shape

One shared Python script, not duplicated batch/bash logic — the mapping
table only needs to be correct in one place:

- **New: `scripts/detect_gpu_checkpoint.py`** (this project's own
  `scripts/` folder, alongside the existing `check_disk_space.ps1`/
  `install_python.ps1` installer helpers — first Python one, since this
  needs `torch`, which only the ACE-Step venv has installed, not a bare
  system Python or PowerShell). Contract: prints the recommended
  checkpoint name to stdout and exits 0 on success; prints nothing to
  stdout (diagnostic info to stderr only) and exits non-zero on any
  failure (no CUDA, no `torch`, detection error) — callers branch on
  exit code, not on parsing stdout for an error string.
- **`install.bat`** — invoke `".venv\Scripts\python.exe" "%~dp0scripts\detect_gpu_checkpoint.py"`
  (using ACE-Step's own venv's python, still valid via `%~dp0` even while
  `pushd`'d into `%ACESTEP_DEST%`) immediately after the pinned torch
  install succeeds, before `popd`. If it exits 0 and
  `SONGFORGE_ACESTEP_CHECKPOINT` isn't already set, `setx` it to the
  printed value, matching the existing `SONGFORGE_ACESTEP_HOME` `setx`
  pattern a few lines below.
- **`install.sh`** — same script, same placement, in the branch where a
  real NVIDIA GPU was already detected (this script already checks for
  `nvidia-smi` to decide CUDA vs CPU torch — reuse that same detected-GPU
  branch, don't duplicate the check). `export`/persist via the same
  mechanism the script already uses for `SONGFORGE_ACESTEP_HOME`.
- **`docs/INSTALLATION.md`** — replace the current "below ~12GB VRAM,
  expect failures" guidance with the real mapping table above, and note
  it's automatic now, not something to hand-configure.
- **README GPU badge** — update from "GPU 12GB+ VRAM" to reflect the new
  effective floor (~6GB via the 2B turbo path).

## Testing

- Unit-test `detect_gpu_checkpoint.py`'s mapping function in isolation
  (pure function of a VRAM float → checkpoint string), mocking
  `torch.cuda` rather than requiring real GPU hardware to run in CI —
  covers every boundary in the table above (just under/at/just over each
  threshold) plus the no-CUDA/no-torch failure paths.
- No automated test for the actual `install.bat`/`install.sh` invocation
  (installer scripts aren't covered by this project's test suite at all
  today) — real validation is running the installer for real and
  confirming the env var gets set correctly, which needs real hardware
  and is a manual verification step, not a CI-able one.

## Explicitly out of scope for this pass

- Runtime/dynamic re-detection (covered above — install-time only, by
  design).
- Any change to `GradioServer.CHECKPOINT`'s own default value in
  `constants.py` — that stays `acestep-v15-xl-sft` as the fallback;
  this feature only changes what gets `setx`'d into the env var before
  that default is ever consulted.
- AMD ROCm / Intel XPU / Apple Silicon VRAM detection — this project's
  installers already only target NVIDIA CUDA (documented, existing
  limitation); this feature doesn't expand that.
