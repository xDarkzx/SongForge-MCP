@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================================
echo SongForge-MCP installer (ACE-Step 1.5 backend)
echo ============================================================

REM ---- Step 1: Python bootstrap ----
where python >nul 2>&1
if errorlevel 1 (
    echo Python not found on PATH. Installing Python 3.11...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_python.ps1"
    if errorlevel 1 (
        echo Failed to install Python automatically.
        echo Install Python 3.11+ manually from python.org and re-run this script.
        exit /b 1
    )
    echo Python installed. You may need to restart this terminal for PATH changes
    echo to take effect before re-running this script.
    exit /b 0
)

REM This server's own venv needs >=3.10 (pyproject.toml) - unlike ACE-Step's
REM narrower 3.11-3.12 requirement (checked separately in Step 4), any 3.10+
REM works here. install.sh already enforces this floor; matching it here so
REM a stale/old `python` on PATH fails fast with a clear message instead of
REM silently building a broken venv.
for /f "delims=" %%V in ('python -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2^>nul') do set "PY_OK=%%V"
if not "%PY_OK%"=="1" (
    echo Python on PATH is older than 3.10 - install Python 3.10+ ^(3.11 or
    echo 3.12 recommended, to also satisfy ACE-Step's own requirement^) and
    echo re-run this script.
    exit /b 1
)

REM ---- Step 2: disk space check (ACE-Step's checkpoint alone is ~28GB) ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\check_disk_space.ps1" -RequiredGB 40
if errorlevel 1 (
    echo Not enough free disk space to continue safely ^(need ~40GB free^).
    echo This project hit 100%% disk usage once already from this same download -
    echo free up space first ^(pip cache purge, old Docker/WSL bloat, etc.^).
    exit /b 1
)

REM ---- Step 3: this server's own venv ----
echo.
echo Setting up songforge-mcp's own venv...
python -m venv .venv
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
".venv\Scripts\python.exe" -m playwright install chromium

REM ---- Step 4: ACE-Step 1.5 checkout ----
if "%SONGFORGE_ACESTEP_HOME%"=="" (
    set "ACESTEP_DEST=%~dp0..\ACE-Step-1.5"
    echo.
    echo SONGFORGE_ACESTEP_HOME not set - cloning ACE-Step 1.5 to !ACESTEP_DEST!...
    if not exist "!ACESTEP_DEST!" (
        git clone https://github.com/ace-step/ACE-Step-1.5.git "!ACESTEP_DEST!"
    )
) else (
    set "ACESTEP_DEST=%SONGFORGE_ACESTEP_HOME%"
)

REM ACE-Step 1.5 requires Python 3.11-3.12 specifically (unlike this
REM server's own venv, which is fine with any 3.10+) - blindly reusing
REM whatever `python` resolves on PATH breaks silently under pyenv-win,
REM where the global/shim default can be a different, incompatible
REM version (confirmed: github.com/xDarkzx/SongForge-MCP/issues/1).
set "ACESTEP_PYTHON="
for /f "delims=" %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\find_acestep_python.ps1"') do set "ACESTEP_PYTHON=%%P"
if "%ACESTEP_PYTHON%"=="" (
    echo No Python 3.11 or 3.12 interpreter found for ACE-Step 1.5 ^(it needs
    echo this range specifically - see docs/INSTALLATION.md^).
    echo If you use pyenv-win, install one and make it the active shell version:
    echo   pyenv install 3.12.7
    echo   pyenv shell 3.12.7
    echo then re-run this installer. Or install Python 3.11/3.12 from python.org
    echo so it's discoverable via the "py" launcher ^(py -3.12^).
    exit /b 1
)
echo.
echo Setting up ACE-Step 1.5's own venv - this is the slow part
echo ^(torch + flash-attn are large downloads, checkpoint download is ~28GB
echo   and happens on first server launch, not during this step^)...
echo Using %ACESTEP_PYTHON% for ACE-Step 1.5's venv.
pushd "%ACESTEP_DEST%"
"%ACESTEP_PYTHON%" -m venv .venv
REM Pinned exactly - do NOT let this float to a newer torch. A newer torch
REM breaks the prebuilt flash-attn wheel below, which in turn breaks
REM nano-vllm's CUDA graph capture path (a much worse failure than just
REM losing the speed benefit - confirmed by testing).
".venv\Scripts\python.exe" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo torch install failed - if this GPU isn't cu128-compatible, see
    echo docs/INSTALLATION.md for alternatives. Aborting.
    popd
    exit /b 1
)
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

REM ---- Step 5: stem separator - isolated venv, unrelated dependency needs ----
echo.
echo Setting up stem-separator venv (isolated from ACE-Step's - different,
echo unrelated dependencies, matching this project's per-tool venv convention)...
set "SEPARATOR_HOME=%~dp0.separator_env"
python -m venv "%SEPARATOR_HOME%"
"%SEPARATOR_HOME%\Scripts\python.exe" -m pip install audio-separator onnxruntime-gpu
setx SONGFORGE_SEPARATOR_PYTHON "%SEPARATOR_HOME%\Scripts\python.exe"

echo Pre-downloading separator models (avoids a slow first-use download
echo during a real generation) - see songforge_mcp_shared/constants.py's
echo Separator class for why these two specifically...
"%SEPARATOR_HOME%\Scripts\audio-separator.exe" --download_model_only -m vocals_mel_band_roformer.ckpt
"%SEPARATOR_HOME%\Scripts\audio-separator.exe" --download_model_only -m model_bs_roformer_ep_368_sdr_12.9628.ckpt

REM ---- Step 6: yt-dlp (already a dependency of this server's own venv) ----
setx SONGFORGE_YTDLP_PYTHON "%~dp0.venv\Scripts\python.exe"

REM ---- Step 6b: Omnizart - optional alternative multi-instrument
REM transcription engine. Own isolated venv (heavy TF + madmom deps,
REM plus checkpoint downloads), same per-tool venv convention as the
REM separator step above. Interactive, unlike the separator step: this
REM is a genuinely optional alternative, not required for core
REM generation, and its install is heavier (large checkpoints, several
REM compatibility patches applied automatically - see
REM scripts/install_omnizart.py for why).
echo.
set /p INSTALL_OMNIZART="Install Omnizart (optional alternative multi-instrument MIDI transcription)? (y/n): "
if /i "!INSTALL_OMNIZART!"=="y" (
    ".venv\Scripts\python.exe" "%~dp0scripts\install_omnizart.py"
    if not errorlevel 1 (
        setx SONGFORGE_OMNIZART_PYTHON "%~dp0.omnizart_env\Scripts\python.exe"
        set "SONGFORGE_OMNIZART_PYTHON=%~dp0.omnizart_env\Scripts\python.exe"
    ) else (
        echo Omnizart install failed - skipping. Basic-pitch transcription still works.
    )
)

REM ---- Step 7: Configure Claude Desktop ----
echo.
echo Configuring Claude Desktop...

REM Claude Desktop installed via the Microsoft Store runs in an app
REM container that redirects its own AppData\Roaming to a virtualized
REM per-package folder under LOCALAPPDATA\Packages - a config file written
REM to the normal %APPDATA%\Claude path is silently never read in that case
REM (confirmed the hard way: writing a valid config there did nothing,
REM because the Store-sandboxed app was never looking at that path to begin
REM with). Detect which install this actually is before writing anywhere.
set "CONFIG_DIR="
for /d %%D in ("%LOCALAPPDATA%\Packages\Claude_*") do (
    if exist "%%D\LocalCache\Roaming\Claude" set "CONFIG_DIR=%%D\LocalCache\Roaming\Claude"
)
if "%CONFIG_DIR%"=="" (
    set "CONFIG_DIR=%APPDATA%\Claude"
    echo Detected standard Claude Desktop install.
) else (
    echo Detected Microsoft Store Claude Desktop install ^(sandboxed config path^).
)

set "CONFIG_FILE=%CONFIG_DIR%\claude_desktop_config.json"
set "SONGFORGE_EXE=%~dp0.venv\Scripts\songforge-mcp.exe"

set /p CONFIGURE_CLAUDE="Configure Claude Desktop for SongForge-MCP? (y/n): "
if /i not "!CONFIGURE_CLAUDE!"=="y" (
    echo Skipped. See docs/INSTALLATION.md for manual setup.
    goto :skip_config
)

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

if exist "%CONFIG_FILE%" (
    findstr /c:"\"songforge-mcp\"" "%CONFIG_FILE%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Claude Desktop config already has a songforge-mcp entry - skipping.
        goto :skip_config
    )
    copy "%CONFIG_FILE%" "%CONFIG_FILE%.bak" >nul 2>&1
    echo Backed up existing config to: %CONFIG_FILE%.bak
    echo.
    echo Found existing Claude Desktop config at:
    echo %CONFIG_FILE%
    echo.
    echo You need to MANUALLY add this inside your "mcpServers" block:
    echo.
    echo   "songforge-mcp": {
    echo     "command": "%SONGFORGE_EXE%"
    echo   }
    echo.
    echo Opening the config file for you...
    notepad "%CONFIG_FILE%"
    goto :skip_config
)

(
echo {
echo   "mcpServers": {
echo     "songforge-mcp": {
echo       "command": "%SONGFORGE_EXE:\=\\%"
echo     }
echo   }
echo }
) > "%CONFIG_FILE%"
echo Created Claude Desktop config at:
echo %CONFIG_FILE%

:skip_config

echo.
echo ============================================================
echo Done.
echo   SONGFORGE_ACESTEP_HOME     = %ACESTEP_DEST%
echo   SONGFORGE_SEPARATOR_PYTHON = %SEPARATOR_HOME%\Scripts\python.exe
echo   SONGFORGE_YTDLP_PYTHON     = %~dp0.venv\Scripts\python.exe
if not "%SONGFORGE_OMNIZART_PYTHON%"=="" (
    echo   SONGFORGE_OMNIZART_PYTHON = %SONGFORGE_OMNIZART_PYTHON%
)
if not "%SONGFORGE_ACESTEP_CHECKPOINT%"=="" (
    echo   SONGFORGE_ACESTEP_CHECKPOINT = %SONGFORGE_ACESTEP_CHECKPOINT%
)
echo.
echo These are now set as permanent user environment variables (setx).
echo Restart your terminal / Claude Desktop for them to take effect.
echo.
echo First real generation will trigger ACE-Step's ~28GB checkpoint download
echo (XL-SFT + 5Hz LM + VAE) - this only happens once.
echo.
echo Run 'songforge-mcp' (inside .venv) to start the server.
echo ============================================================
