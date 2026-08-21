import subprocess

from songforge_mcp.banner import print_ready_banner, print_startup_banner
from songforge_mcp.job_registry import JobRegistry


def test_print_startup_banner_writes_only_to_stderr(capsys):
    print_startup_banner(JobRegistry())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SongForge" in captured.err


def test_print_ready_banner_writes_only_to_stderr(capsys):
    print_ready_banner("acestep-v15-xl-sft")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "READY" in captured.err


def test_print_ready_banner_survives_missing_nvidia_smi(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    print_ready_banner("acestep-v15-xl-sft")  # must not raise

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GPU" not in captured.err
