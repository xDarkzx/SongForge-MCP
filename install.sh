#!/usr/bin/env bash
# SongForge-MCP installer (ACE-Step 1.5 backend) - Linux/macOS.
#
# Mirrors install.bat's steps. Two things are deliberately NOT automated
# here, unlike the Windows script:
#   1. Installing Python itself - package managers differ too much
#      (apt/dnf/pacman/brew) to do this unattended and safely. If python3
#      is missing, this prints the right command for your OS and exits.
#   2. Persisting environment variables - there is no cross-shell
#      equivalent of Windows' `setx`. This prints the export lines for you
#      to add to your shell profile (~/.bashrc, ~/.zshrc, etc.) rather than
#      editing that file for you.
#
# GPU note: the torch/onnxruntime builds pinned below are CUDA (NVIDIA)
# builds. On Linux with an NVIDIA GPU this matches install.bat exactly. On
# macOS (no CUDA) or Linux without an NVIDIA GPU, this falls back to plain
# CPU builds - ACE-Step and audio-separator will both run, but slower;
# this has not been performance-tested by this project, only confirmed to
# be the correct fallback per each package's own published wheel support.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "SongForge-MCP installer (ACE-Step 1.5 backend)"
echo "============================================================"

# ---- Step 1: Python check ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found on PATH."
    case "$(uname -s)" in
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                echo "Install it with: sudo apt-get install -y python3 python3-venv python3-pip"
            elif command -v dnf >/dev/null 2>&1; then
                echo "Install it with: sudo dnf install -y python3 python3-pip"
            elif command -v pacman >/dev/null 2>&1; then
                echo "Install it with: sudo pacman -S python python-pip"
            else
                echo "Install Python 3.10+ using your distro's package manager, then re-run this script."
            fi
            ;;
        Darwin)
            echo "Install it with: brew install python@3.11"
            echo "(or from https://python.org if you don't use Homebrew)"
            ;;
        *)
            echo "Install Python 3.10+ manually, then re-run this script."
            ;;
    esac
    exit 1
fi

PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "$PY_OK" != "1" ]; then
    echo "python3 is older than 3.10 ($(python3 --version)). Install 3.10+ and re-run."
    exit 1
fi

# ---- Step 2: disk space check (ACE-Step's checkpoint alone is ~28GB) ----
REQUIRED_GB=40
AVAILABLE_KB=$(df -Pk . | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo "Not enough free disk space to continue safely (need ~${REQUIRED_GB}GB free, found ~${AVAILABLE_GB}GB)."
    echo "This project hit 100% disk usage once already from this same download -"
    echo "free up space first (pip cache purge, old container/VM image bloat, etc.)."
    exit 1
fi

detect_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1
}

# ACE-Step 1.5 requires Python 3.11-3.12 specifically (unlike this server's
# own venv, which is fine with any 3.10+) - blindly reusing whatever
# `python3` resolves on PATH breaks silently under pyenv/pyenv-win-style
# version managers, where the global/shim default can be a different,
# incompatible version (confirmed on Windows: github.com/xDarkzx/SongForge-MCP/issues/1).
# Checks versioned binaries first (python3.12/python3.11, as installed by
# most package managers and pyenv), then falls back to python3 itself if
# its own version happens to already qualify.
find_acestep_python() {
    for candidate in python3.12 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    if command -v python3 >/dev/null 2>&1; then
        local ver
        ver="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        if [ "$ver" = "3.11" ] || [ "$ver" = "3.12" ]; then
            echo "python3"
            return 0
        fi
    fi
    return 1
}

# ---- Step 3: this server's own venv ----
echo
echo "Setting up songforge-mcp's own venv..."
python3 -m venv .venv
".venv/bin/python" -m pip install -e ".[dev]"
".venv/bin/python" -m playwright install chromium

# ---- Step 4: ACE-Step 1.5 checkout ----
if [ -z "${SONGFORGE_ACESTEP_HOME:-}" ]; then
    ACESTEP_DEST="$(pwd)/../ACE-Step-1.5"
    echo
    echo "SONGFORGE_ACESTEP_HOME not set - cloning ACE-Step 1.5 to ${ACESTEP_DEST}..."
    if [ ! -d "$ACESTEP_DEST" ]; then
        git clone https://github.com/ace-step/ACE-Step-1.5.git "$ACESTEP_DEST"
    fi
else
    ACESTEP_DEST="$SONGFORGE_ACESTEP_HOME"
fi

ACESTEP_PYTHON="$(find_acestep_python)" || {
    echo "No Python 3.11 or 3.12 interpreter found for ACE-Step 1.5 (it needs"
    echo "this range specifically - see docs/INSTALLATION.md)."
    echo "Install one, e.g.:"
    echo "  pyenv install 3.12.7 && pyenv local 3.12.7   (if using pyenv)"
    echo "  or via your package manager: apt-get install python3.12 / brew install python@3.12"
    exit 1
}

echo
echo "Setting up ACE-Step 1.5's own venv - this is the slow part"
echo "(torch + flash-attn are large downloads, checkpoint download is ~28GB"
echo " and happens on first server launch, not during this step)..."
echo "Using $ACESTEP_PYTHON for ACE-Step 1.5's venv."
pushd "$ACESTEP_DEST" >/dev/null
"$ACESTEP_PYTHON" -m venv .venv
if detect_nvidia_gpu; then
    # Pinned exactly, matching install.bat - do NOT let this float to a
    # newer torch. A newer torch breaks the prebuilt flash-attn wheel,
    # which in turn breaks nano-vllm's CUDA graph capture path (a much
    # worse failure than just losing the speed benefit - confirmed by
    # testing on Windows; assumed to hold on Linux since it's the same
    # torch/flash-attn/nano-vllm version combination).
    if ! ".venv/bin/python" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128; then
        echo "torch install failed - see docs/INSTALLATION.md for alternatives. Aborting."
        popd >/dev/null
        exit 1
    fi
else
    echo "No NVIDIA GPU detected (nvidia-smi not found) - installing CPU torch build."
    echo "This is untested by this project; generation will likely be much slower."
    if ! ".venv/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1; then
        echo "torch install failed. Aborting."
        popd >/dev/null
        exit 1
    fi
fi
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

# ---- Step 5: stem separator - isolated venv, unrelated dependency needs ----
echo
echo "Setting up stem-separator venv (isolated from ACE-Step's - different,"
echo "unrelated dependencies, matching this project's per-tool venv convention)..."
SEPARATOR_HOME="$(pwd)/.separator_env"
python3 -m venv "$SEPARATOR_HOME"
if detect_nvidia_gpu; then
    "$SEPARATOR_HOME/bin/python" -m pip install audio-separator onnxruntime-gpu
else
    echo "No NVIDIA GPU detected - installing CPU onnxruntime. Separation will be slower."
    "$SEPARATOR_HOME/bin/python" -m pip install audio-separator onnxruntime
fi

echo "Pre-downloading separator models (avoids a slow first-use download"
echo "during a real generation) - see songforge_mcp_shared/constants.py's"
echo "Separator class for why these two specifically..."
"$SEPARATOR_HOME/bin/audio-separator" --download_model_only -m vocals_mel_band_roformer.ckpt
"$SEPARATOR_HOME/bin/audio-separator" --download_model_only -m model_bs_roformer_ep_368_sdr_12.9628.ckpt

# ---- Step 6: yt-dlp (already a dependency of this server's own venv) ----
YTDLP_PYTHON="$(pwd)/.venv/bin/python"

# ---- Step 6b: Omnizart - optional alternative multi-instrument
# transcription engine. Own isolated venv, same per-tool venv convention
# as the separator step above. Interactive, unlike the separator step:
# genuinely optional, heavier install (large checkpoints, several
# compatibility patches applied automatically - see
# scripts/install_omnizart.py for why).
echo
OMNIZART_PYTHON=""
read -r -p "Install Omnizart (optional alternative multi-instrument MIDI transcription)? (y/n): " INSTALL_OMNIZART
if [ "$INSTALL_OMNIZART" = "y" ] || [ "$INSTALL_OMNIZART" = "Y" ]; then
    if ".venv/bin/python" "$PROJECT_ROOT/scripts/install_omnizart.py"; then
        OMNIZART_PYTHON="$(pwd)/.omnizart_env/bin/python"
    else
        echo "Omnizart install failed - skipping. Basic-pitch transcription still works."
    fi
fi

# ---- Step 7: Configure Claude Desktop ----
SONGFORGE_EXE="$(pwd)/.venv/bin/songforge-mcp"
echo
case "$(uname -s)" in
    Darwin)
        CONFIG_DIR="$HOME/Library/Application Support/Claude"
        CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
        read -r -p "Configure Claude Desktop for SongForge-MCP? (y/n): " CONFIGURE_CLAUDE
        if [ "$CONFIGURE_CLAUDE" = "y" ] || [ "$CONFIGURE_CLAUDE" = "Y" ]; then
            mkdir -p "$CONFIG_DIR"
            if [ -f "$CONFIG_FILE" ]; then
                if grep -q '"songforge-mcp"' "$CONFIG_FILE" 2>/dev/null; then
                    echo "Claude Desktop config already has a songforge-mcp entry - skipping."
                else
                    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
                    echo "Backed up existing config to: $CONFIG_FILE.bak"
                    echo
                    echo "Found existing Claude Desktop config at: $CONFIG_FILE"
                    echo "You need to MANUALLY add this inside your \"mcpServers\" block:"
                    echo
                    echo "  \"songforge-mcp\": {"
                    echo "    \"command\": \"$SONGFORGE_EXE\""
                    echo "  }"
                fi
            else
                cat > "$CONFIG_FILE" <<EOF
{
  "mcpServers": {
    "songforge-mcp": {
      "command": "$SONGFORGE_EXE"
    }
  }
}
EOF
                echo "Created Claude Desktop config at: $CONFIG_FILE"
            fi
        else
            echo "Skipped. See docs/INSTALLATION.md for manual setup."
        fi
        ;;
    *)
        echo "Claude Desktop's official app doesn't run natively on this OS -"
        echo "if you're using it through another means, add this to its MCP config:"
        echo
        echo "  \"songforge-mcp\": {"
        echo "    \"command\": \"$SONGFORGE_EXE\""
        echo "  }"
        ;;
esac

echo
echo "============================================================"
echo "Done."
echo "  SONGFORGE_ACESTEP_HOME     = $ACESTEP_DEST"
echo "  SONGFORGE_SEPARATOR_PYTHON = $SEPARATOR_HOME/bin/python"
echo "  SONGFORGE_YTDLP_PYTHON     = $YTDLP_PYTHON"
if [ -n "$OMNIZART_PYTHON" ]; then
    echo "  SONGFORGE_OMNIZART_PYTHON = $OMNIZART_PYTHON"
fi
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
if [ -n "$OMNIZART_PYTHON" ]; then
    echo "  export SONGFORGE_OMNIZART_PYTHON=\"$OMNIZART_PYTHON\""
fi
if [ -n "${SONGFORGE_ACESTEP_CHECKPOINT:-}" ]; then
    echo "  export SONGFORGE_ACESTEP_CHECKPOINT=\"$SONGFORGE_ACESTEP_CHECKPOINT\""
fi
echo
echo "First real generation will trigger ACE-Step's ~28GB checkpoint download"
echo "(XL-SFT + 5Hz LM + VAE) - this only happens once."
echo
echo "Run 'songforge-mcp' (inside .venv) to start the server."
echo "============================================================"
