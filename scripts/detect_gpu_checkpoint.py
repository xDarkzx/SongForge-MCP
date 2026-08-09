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
