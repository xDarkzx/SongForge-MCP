import asyncio

import pytest
from mcp.server.fastmcp import FastMCP

from songforge_mcp.tools import generate_tools
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError


class _StubContext:
    """Minimal stand-in for FastMCP's Context, sufficient for exercising
    validation logic that raises before any real progress reporting would
    occur — these tests never reach the point of calling report_progress
    for real."""

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        pass


def _register() -> FastMCP:
    mcp = FastMCP("test")
    generate_tools.register(mcp)
    return mcp


def test_generate_vocal_track_rejects_empty_caption():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(caption="", lyrics="[verse]\nsome words", ctx=_StubContext()))
    assert exc_info.value.code == ErrorCode.MISSING_PARAMETER


def test_generate_vocal_track_rejects_lyrics_without_structure_tags():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(caption="melodic dubstep", lyrics="no structure tags here", ctx=_StubContext()))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_rejects_both_reference_sources_at_once():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nsome words",
            ctx=_StubContext(),
            reference_audio_path="/some/path.wav",
            reference_youtube_url="https://youtube.com/watch?v=abc",
        ))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_rejects_both_remix_sources_at_once():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nsome words",
            ctx=_StubContext(),
            remix_source_path="/some/path.wav",
            remix_source_youtube_url="https://youtube.com/watch?v=abc",
        ))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_rejects_remix_and_reference_together():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nsome words",
            ctx=_StubContext(),
            reference_audio_path="/some/path.wav",
            remix_source_youtube_url="https://youtube.com/watch?v=abc",
        ))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_rejects_out_of_range_remix_strength():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nsome words",
            ctx=_StubContext(),
            remix_source_youtube_url="https://youtube.com/watch?v=abc",
            remix_strength=1.5,
        ))
    assert exc_info.value.code == ErrorCode.VALUE_OUT_OF_RANGE


def test_generate_vocal_track_returns_immediately_with_a_job_id(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")

    async def slow_generate(**kwargs):
        await asyncio.sleep(3600)  # would fail any reasonable test timeout if awaited
        raise AssertionError("should never actually run to completion in this test")

    monkeypatch.setattr(generate_tools._client, "generate", slow_generate)

    async def scenario():
        return await asyncio.wait_for(
            gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext()),
            timeout=1.0,
        )

    result = asyncio.run(scenario())
    assert "job_id" in result and result["job_id"]


def test_check_vocal_track_status_rejects_unknown_job_id():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("check_vocal_track_status")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(job_id="does-not-exist"))
    assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


def test_generate_vocal_track_job_completes_and_status_reports_it(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        await on_progress(0.5, "halfway")
        return {"audio_path": "/fake/render.mp3", "generation_seconds": 1.23}

    opened = []
    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)
    monkeypatch.setattr(generate_tools, "play_audio_now", opened.append)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    assert status[0]["audio_path"] == "/fake/render.mp3"
    assert opened == ["/fake/render.mp3"]


def test_generate_vocal_track_split_stems_includes_stem_paths_in_completion(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        return {"audio_path": "/fake/render.wav", "generation_seconds": 1.0}

    def fake_separate(audio_path):
        assert audio_path == "/fake/render.wav"
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    opened = []
    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)
    monkeypatch.setattr(generate_tools._separator_client, "separate", fake_separate)
    monkeypatch.setattr(generate_tools, "measure_wav_duration_seconds", lambda path: 30.0)
    monkeypatch.setattr(generate_tools, "play_audio_now", opened.append)

    async def scenario():
        result = await gen_tool.fn(
            caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext(), split_stems=True
        )
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    assert status[0]["vocals_path"] == "/fake/vocals.wav"
    # Auto-launches the full mix, not the stems - opening both stems at once
    # would mean two overlapping audio players fighting for playback.
    assert opened == ["/fake/render.wav"]


def test_generate_vocal_track_still_completes_if_auto_launch_fails(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        return {"audio_path": "/fake/render.wav", "generation_seconds": 1.0}

    def failing_open(path):
        raise OSError("no associated application")

    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)
    monkeypatch.setattr(generate_tools, "measure_wav_duration_seconds", lambda path: 30.0)
    monkeypatch.setattr(generate_tools, "play_audio_now", failing_open)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    assert status[0]["audio_path"] == "/fake/render.wav"


def test_generate_vocal_track_without_split_stems_omits_stem_paths(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        return {"audio_path": "/fake/render.mp3", "generation_seconds": 1.0}

    def fake_separate(audio_path):
        raise AssertionError("separate() must not be called when split_stems is False")

    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)
    monkeypatch.setattr(generate_tools._separator_client, "separate", fake_separate)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    assert "vocals_path" not in status[0]
    assert "instrumental_path" not in status[0]


def test_check_vocal_track_status_waits_for_completion_within_wait_seconds(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        await asyncio.sleep(0.2)
        return {"audio_path": "/fake/render.mp3", "generation_seconds": 0.2}

    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        # Poll once, immediately, with a wait comfortably longer than the
        # fake generation - this should observe completion server-side
        # rather than needing a second poll.
        return await asyncio.wait_for(
            status_tool.fn(job_id=result["job_id"], wait_seconds=2.0), timeout=3.0
        )

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"


def test_check_vocal_track_status_returns_running_immediately_when_wait_seconds_is_zero(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def slow_generate(**kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(generate_tools._client, "generate", slow_generate)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        return await asyncio.wait_for(
            status_tool.fn(job_id=result["job_id"], wait_seconds=0.0), timeout=1.0
        )

    status = asyncio.run(scenario())
    assert status[0]["status"] == "running"


def test_generate_vocal_track_rejects_invalid_output_format():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nsome words",
            ctx=_StubContext(),
            output_format="ogg",
        ))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_defaults_to_wav_output_format(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")
    seen_formats = []

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        seen_formats.append(output_format)
        # .mp3, not .wav, so this doesn't trigger real wave-file parsing
        # against a path that doesn't exist - this test only checks that
        # the output_format parameter itself propagates correctly.
        return {"audio_path": "/fake/render.mp3", "generation_seconds": 1.0}

    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert seen_formats == ["wav"]
    assert status[0]["diagnostics"]["output_format"] == "wav"


def test_generate_vocal_track_job_reports_error_status(monkeypatch):
    mcp = _register()
    gen_tool = mcp._tool_manager.get_tool("generate_vocal_track")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    async def failing_generate(**kwargs):
        raise SongForgeMCPError(ErrorCode.SYNTHESIS_FAILED, "boom")

    monkeypatch.setattr(generate_tools._client, "generate", failing_generate)

    async def scenario():
        result = await gen_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext())
        for _ in range(50):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "error"
    assert "SYNTHESIS_FAILED" in status[0]["error"]


def test_generate_vocal_track_takes_rejects_too_few_takes():
    # Regression test for a real incident: the calling model hit this
    # exact error for num_takes=1, then silently rounded up to 2 takes
    # instead of surfacing the error or switching tools - doubling
    # generation time against what the user actually asked for. The
    # error message must name generate_vocal_track directly so there's
    # no ambiguous "just clamp to the minimum" reading left.
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext(), num_takes=1))
    assert exc_info.value.code == ErrorCode.VALUE_OUT_OF_RANGE
    assert "generate_vocal_track" in exc_info.value.message


def test_generate_vocal_track_takes_rejects_too_many_takes():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext(), num_takes=6))
    assert exc_info.value.code == ErrorCode.VALUE_OUT_OF_RANGE


def test_generate_vocal_track_takes_rejects_empty_caption():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(caption="", lyrics="[verse]\nwords", ctx=_StubContext()))
    assert exc_info.value.code == ErrorCode.MISSING_PARAMETER


def test_generate_vocal_track_takes_rejects_both_reference_sources_at_once():
    mcp = _register()
    tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(tool.fn(
            caption="melodic dubstep",
            lyrics="[verse]\nwords",
            ctx=_StubContext(),
            reference_audio_path="/some/path.wav",
            reference_youtube_url="https://youtube.com/watch?v=abc",
        ))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_generate_vocal_track_takes_runs_each_take_and_reports_seeds_and_measurements(monkeypatch):
    mcp = _register()
    takes_tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    seen_seeds = []

    async def fake_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        seen_seeds.append(advanced_settings["Seed"])
        return {"audio_path": f"/fake/take_{advanced_settings['Seed']}.mp3", "generation_seconds": 1.0}

    monkeypatch.setattr(generate_tools._client, "generate", fake_generate)
    monkeypatch.setattr(generate_tools, "analyze_audio", lambda path: {"bpm": 120.0, "key": "A", "mode": "minor", "key_confidence": 0.9})

    async def scenario():
        result = await takes_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext(), num_takes=3)
        for _ in range(200):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    takes = status[0]["takes"]
    assert len(takes) == 3
    # Every take got its own distinct seed, and each one's diagnostics/
    # measurement reflect that exact seed's generation, not a shared/stale value.
    assert len({t["seed"] for t in takes}) == 3
    assert len(set(seen_seeds)) == 3
    for take in takes:
        assert take["measured"]["bpm"] == 120.0
        assert take["audio_path"] == f"/fake/take_{take['seed']}.mp3"


def test_generate_vocal_track_takes_records_per_take_error_without_failing_whole_job(monkeypatch):
    mcp = _register()
    takes_tool = mcp._tool_manager.get_tool("generate_vocal_track_takes")
    status_tool = mcp._tool_manager.get_tool("check_vocal_track_status")

    call_count = {"n": 0}

    async def flaky_generate(*, caption, lyrics, reference_audio_path, advanced_settings, output_format, remix_source_path, remix_strength, remix_melody_retention, remix_no_fsq, song_title, lora_path, on_progress):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise SongForgeMCPError(ErrorCode.SYNTHESIS_FAILED, "take 2 blew up")
        return {"audio_path": "/fake/ok.mp3", "generation_seconds": 1.0}

    monkeypatch.setattr(generate_tools._client, "generate", flaky_generate)
    monkeypatch.setattr(generate_tools, "analyze_audio", lambda path: {"bpm": 120.0, "key": "A", "mode": "minor", "key_confidence": 0.9})

    async def scenario():
        result = await takes_tool.fn(caption="melodic dubstep", lyrics="[verse]\nwords", ctx=_StubContext(), num_takes=3)
        for _ in range(200):
            await asyncio.sleep(0)
        return await status_tool.fn(job_id=result["job_id"])

    status = asyncio.run(scenario())
    assert status[0]["status"] == "complete"
    takes = status[0]["takes"]
    assert len(takes) == 3
    assert "error" in takes[1] and "SYNTHESIS_FAILED" in takes[1]["error"]
    assert "audio_path" in takes[0] and "audio_path" in takes[2]
