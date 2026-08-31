# Locates a Python 3.11 or 3.12 interpreter for ACE-Step 1.5, which requires
# exactly that range (docs/INSTALLATION.md) - unlike this server's own venv,
# which is permissive to any 3.10+. install.bat used to blindly reuse
# whatever `python` resolved on PATH for both venvs; under pyenv-win the
# global/shim default can silently be a different, incompatible version
# (github.com/xDarkzx/SongForge-MCP/issues/1). Checks the `py` launcher's
# -3.12 and -3.11 first (these select an installed version directly,
# independent of PATH/shims), then falls back to whatever `python` is on
# PATH if its own version happens to already qualify.
#
# Prints the resolved interpreter's path and exits 0 on success; exits 1
# with nothing printed if no qualifying interpreter is found.
$ErrorActionPreference = "Stop"

function Get-Version($exe, $exeArgs) {
    try {
        & $exe @exeArgs -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    } catch {
        $null
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($launcherArg in @("-3.12", "-3.11")) {
        $ver = Get-Version "py" @($launcherArg)
        if ($ver -eq "3.11" -or $ver -eq "3.12") {
            & py $launcherArg -c "import sys; print(sys.executable)"
            exit 0
        }
    }
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    $ver = Get-Version "python" @()
    if ($ver -eq "3.11" -or $ver -eq "3.12") {
        & python -c "import sys; print(sys.executable)"
        exit 0
    }
}

exit 1
