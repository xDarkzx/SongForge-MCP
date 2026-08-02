import os
import platform
from types import SimpleNamespace

import pytest

from songforge_mcp.separator_client import SeparatorClient
from songforge_mcp_shared import constants
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError

_EXE_NAME = "audio-separator.exe" if platform.system() == "Windows" else "audio-separator"


def _make_fake_venv(tmp_path):
    """Creates a fake python interpreter + audio-separator console script
    on disk so SeparatorClient._require_configured passes its existence
    checks without needing a real venv."""
    venv_dir = tmp_path / "fake_venv"
    venv_dir.mkdir()
    python_exe = venv_dir / ("python.exe" if platform.system() == "Windows" else "python")
    python_exe.write_text("")
    separator_exe = venv_dir / _EXE_NAME
    separator_exe.write_text("")
    return str(python_exe)


def test_separate_raises_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("SONGFORGE_SEPARATOR_PYTHON", raising=False)
    # A path that definitely doesn't point to a real interpreter - "" would
    # be falsy and silently fall through to any real env var still set in
    # the environment, which isn't what this test means to exercise.
    client = SeparatorClient(separator_venv_python=str(tmp_path / "no_such_python.exe"))
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate(str(tmp_path / "input.wav"))
    assert exc_info.value.code == ErrorCode.SEPARATOR_NOT_CONFIGURED


def test_separate_raises_when_input_file_missing(tmp_path):
    python_exe = _make_fake_venv(tmp_path)
    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate(str(tmp_path / "does_not_exist.wav"))
    assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


def test_separate_is_idempotent_and_skips_subprocess_when_stems_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "vocals_mel_band_roformer_ckpt"
    stems_dir.mkdir(parents=True)
    existing_vocals = stems_dir / "render_(Vocals)_vocals_mel_band_roformer.wav"
    existing_instrumental = stems_dir / "render_(Instrumental)_vocals_mel_band_roformer.wav"
    existing_vocals.write_text("fake vocals")
    existing_instrumental.write_text("fake instrumental")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when stems already exist")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fail_if_called)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate(str(input_path), model_filename="vocals_mel_band_roformer.ckpt")

    assert result == {
        "vocals_path": str(existing_vocals),
        "instrumental_path": str(existing_instrumental),
    }


def test_separate_different_model_does_not_reuse_other_models_cached_stems(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "vocals_mel_band_roformer_ckpt"
    stems_dir.mkdir(parents=True)
    (stems_dir / "render_(Vocals)_vocals_mel_band_roformer.wav").write_text("fake vocals a")
    (stems_dir / "render_(Instrumental)_vocals_mel_band_roformer.wav").write_text("fake instrumental a")

    called = {}

    def fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "render_(Vocals)_model_bs_roformer.wav"), "w") as f:
            f.write("fake vocals b")
        with open(os.path.join(out_dir, "render_(Instrumental)_model_bs_roformer.wav"), "w") as f:
            f.write("fake instrumental b")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate(
        str(input_path), model_filename="model_bs_roformer_ep_368_sdr_12.9628.ckpt"
    )

    assert "cmd" in called, "a different model must re-run separation, not reuse the other model's cached stems"
    assert "model_bs_roformer_ep_368_sdr_12_9628_ckpt" in result["vocals_path"]


def test_separate_runs_subprocess_and_finds_output_when_no_existing_stems(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "render_(Vocals)_vocals_mel_band_roformer.wav"), "w") as f:
            f.write("fake vocals")
        with open(os.path.join(out_dir, "render_(Instrumental)_vocals_mel_band_roformer.wav"), "w") as f:
            f.write("fake instrumental")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate(str(input_path))

    assert result["vocals_path"].endswith("render_(Vocals)_vocals_mel_band_roformer.wav")
    assert result["instrumental_path"].endswith("render_(Instrumental)_vocals_mel_band_roformer.wav")


def test_separate_classifies_stems_correctly_when_model_name_itself_contains_vocal(tmp_path, monkeypatch):
    """Regression test for a real production bug: vocals_mel_band_roformer's
    own filename contains "vocal", and gets appended as a suffix to BOTH
    output files - naively checking for "vocal" anywhere in the filename
    misclassified both as vocals, leaving nothing for instrumental. Must
    classify only the parenthesized stem label, e.g. "(other)" here."""
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "drowning-in-you.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "drowning-in-you_(vocals)_vocals_mel_band_roformer.wav"), "w") as f:
            f.write("fake vocals")
        with open(os.path.join(out_dir, "drowning-in-you_(other)_vocals_mel_band_roformer.wav"), "w") as f:
            f.write("fake other")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate(str(input_path), model_filename="vocals_mel_band_roformer.ckpt")

    assert result["vocals_path"].endswith("(vocals)_vocals_mel_band_roformer.wav")
    assert result["instrumental_path"].endswith("(other)_vocals_mel_band_roformer.wav")


def test_separate_raises_on_nonzero_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="model failed to load")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate(str(input_path))
    assert exc_info.value.code == ErrorCode.SEPARATION_FAILED


_FAKE_LIST_MODELS_OUTPUT = """\
Model Filename                                                      Arch    Output Stems (SDR)                                            Friendly Name
-----------------------------------------------------------------------------------------------------------------------------------------------------
vocals_mel_band_roformer.ckpt                                       MDXC    vocals* (12.6), instrumental (16.9)                           Roformer Model: MelBand Roformer | Vocals by Kimberley Jensen
model_bs_roformer_ep_368_sdr_12.9628.ckpt                           MDXC    vocals* (12.1), instrumental (16.3)                           Roformer Model: BS-Roformer-Viperx-1296
denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt                   MDXC    dry*, other                                                    Roformer Model: Mel-Roformer-Denoise-Aufr33
"""


def test_list_models_parses_and_sorts_by_vocal_sdr_descending(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)

    def fake_run(cmd, **kwargs):
        assert cmd[1] == "--list_models"
        return SimpleNamespace(returncode=0, stdout=_FAKE_LIST_MODELS_OUTPUT, stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    models = client.list_models()

    assert [m["filename"] for m in models] == [
        "vocals_mel_band_roformer.ckpt",
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    ]
    assert models[0]["vocal_sdr"] == 12.6
    assert models[0]["arch"] == "MDXC"
    assert "Kimberley Jensen" in models[0]["friendly_name"]


def test_list_models_vocal_only_false_includes_non_vocal_models(tmp_path, monkeypatch):
    python_exe = _make_fake_venv(tmp_path)

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=_FAKE_LIST_MODELS_OUTPUT, stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    models = client.list_models(vocal_only=False)

    filenames = [m["filename"] for m in models]
    assert "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt" in filenames
    denoise_entry = next(m for m in models if m["filename"].startswith("denoise_"))
    assert denoise_entry["vocal_sdr"] is None


def test_find_labeled_stems_parses_demucs_six_stem_output(tmp_path):
    from songforge_mcp.separator_client import _find_labeled_stems

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stem = "mytrack"
    for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
        (out_dir / f"{stem}_({label})_htdemucs_6s.wav").write_text("fake")

    result = _find_labeled_stems(str(out_dir), stem)

    assert set(result.keys()) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    assert result["drums"].name == f"{stem}_(Drums)_htdemucs_6s.wav"


def test_find_labeled_stems_returns_empty_dict_when_no_matches(tmp_path):
    from songforge_mcp.separator_client import _find_labeled_stems

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = _find_labeled_stems(str(out_dir), "mytrack")

    assert result == {}


def test_separate_extra_stems_runs_subprocess_and_excludes_vocals(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert set(result.keys()) == {"drums_path", "bass_path", "guitar_path", "piano_path", "other_path"}
    assert "vocals_path" not in result
    assert result["drums_path"].endswith("(Drums)_htdemucs_6s.wav")


def test_separate_extra_stems_defaults_to_htdemucs_6s(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    client.separate_extra_stems(str(input_path))

    assert "htdemucs_6s.yaml" in captured["cmd"]


def test_separate_extra_stems_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "htdemucs_6s_yaml"
    stems_dir.mkdir(parents=True)
    for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
        (stems_dir / f"render_({label})_htdemucs_6s.wav").write_text(f"fake {label}")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when extra stems already exist")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fail_if_called)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert set(result.keys()) == {"drums_path", "bass_path", "guitar_path", "piano_path", "other_path"}


def test_separate_extra_stems_reruns_when_existing_stems_are_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "htdemucs_6s_yaml"
    stems_dir.mkdir(parents=True)
    # Only one stray file present - e.g. a previous run was killed mid-write.
    (stems_dir / "render_(Drums)_htdemucs_6s.wav").write_text("fake drums")

    called = {}

    def fake_run(cmd, **kwargs):
        called["ran"] = True
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert called.get("ran") is True
    assert len(result) == 5


def test_separate_extra_stems_raises_on_nonzero_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="model failed to load")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")
    assert exc_info.value.code == ErrorCode.SEPARATION_FAILED


def test_separate_extra_stems_raises_when_input_file_missing(tmp_path):
    python_exe = _make_fake_venv(tmp_path)
    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate_extra_stems(str(tmp_path / "does_not_exist.wav"))
    assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND
