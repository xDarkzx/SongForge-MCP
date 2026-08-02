import asyncio

import numpy as np
import pytest
import soundfile as sf
from mcp.server.fastmcp import FastMCP

from songforge_mcp.tools import separate_tools
from songforge_mcp_shared import constants
from songforge_mcp_shared.constants import Separator
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError


def _register() -> FastMCP:
    mcp = FastMCP("test")
    separate_tools.register(mcp)
    return mcp


def _write_tone(path, duration_seconds=1.0, samplerate=48000):
    t = np.linspace(0, duration_seconds, int(duration_seconds * samplerate), endpoint=False)
    data = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), data, samplerate)


def test_split_vocal_stems_rejects_path_outside_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    outside_path = tmp_path / "elsewhere.wav"
    _write_tone(outside_path)

    mcp = _register()
    tool = mcp._tool_manager.get_tool("split_vocal_stems")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(audio_path=str(outside_path)))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_split_vocal_stems_returns_job_id_immediately_and_completes(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        assert path == str(audio_path)
        assert model_filename == Separator.DEFAULT_MODEL
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path))
        assert "job_id" in result
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "complete"
    assert job.result == {
        "vocals_path": "/fake/vocals.wav",
        "instrumental_path": "/fake/instrumental.wav",
        "model": Separator.DEFAULT_MODEL,
    }


def test_split_vocal_stems_passes_explicit_model_through(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        assert model_filename == Separator.ALT_MODEL
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path), model=Separator.ALT_MODEL)
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "complete"
    assert job.result["model"] == Separator.ALT_MODEL


def test_split_vocal_stems_job_reports_error(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        raise SongForgeMCPError(ErrorCode.SEPARATION_FAILED, "boom")

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path))
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "error"
    assert "boom" in job.error


def test_split_vocal_stems_and_generate_vocal_track_share_one_job_registry():
    from songforge_mcp.tools import generate_tools

    assert generate_tools._jobs is separate_tools._jobs


def test_list_separator_models_returns_client_result_directly(monkeypatch):
    fake_models = [
        {"filename": "vocals_mel_band_roformer.ckpt", "arch": "MDXC", "scores": "vocals* (12.6)",
         "friendly_name": "MelBand Roformer | Vocals", "vocal_sdr": 12.6},
    ]

    def fake_list_models(vocal_only=True):
        assert vocal_only is True
        return fake_models

    monkeypatch.setattr(separate_tools._client, "list_models", fake_list_models)

    mcp = _register()
    tool = mcp._tool_manager.get_tool("list_separator_models")

    result = asyncio.run(tool.fn())

    assert result == {"models": fake_models}


def test_list_separator_models_passes_vocal_only_through(monkeypatch):
    captured = {}

    def fake_list_models(vocal_only=True):
        captured["vocal_only"] = vocal_only
        return []

    monkeypatch.setattr(separate_tools._client, "list_models", fake_list_models)

    mcp = _register()
    tool = mcp._tool_manager.get_tool("list_separator_models")

    asyncio.run(tool.fn(vocal_only=False))

    assert captured["vocal_only"] is False
