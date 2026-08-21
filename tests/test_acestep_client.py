import asyncio
import os
import time

import numpy as np
import psutil
import pytest
import soundfile as sf

from songforge_mcp.acestep_client import (
    _REFERENCE_TRIM_WINDOW_SECONDS,
    ACEStepClient,
    _slugify_title,
    _trim_reference_for_key_coherence,
)
from songforge_mcp_shared import constants
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError


class _FakeLocator:
    """Minimal async stand-in for a Playwright Locator - just the methods
    _set_field_by_label actually calls."""

    def __init__(self, *, count=1, visible=True, value="", checked=False,
                 raise_on_select_option=True, raise_on_fill=False,
                 fill_is_cosmetic=False):
        self._count = count
        self._visible = visible
        self._value = value
        self._checked = checked
        self._raise_on_select_option = raise_on_select_option
        self._raise_on_fill = raise_on_fill
        # Simulates the real confirmed bug: fill() succeeds (doesn't
        # throw) but only types into a combobox's filter input without
        # ever registering a real backend selection - the field's real
        # value never actually changes.
        self._fill_is_cosmetic = fill_is_cosmetic

    @property
    def first(self):
        return self

    async def count(self):
        return self._count

    async def is_visible(self):
        return self._visible

    async def click(self, timeout=None):
        pass

    async def select_option(self, value, timeout=None):
        if self._raise_on_select_option:
            raise Exception("not a native <select>")
        self._value = value

    async def fill(self, value, timeout=None):
        if self._raise_on_fill:
            raise Exception("fill() failed")
        if not self._fill_is_cosmetic:
            self._value = value

    async def input_value(self, timeout=None):
        return self._value

    async def check(self, timeout=None):
        if self._count == 0:
            raise Exception("no element matches this locator")
        self._checked = True

    async def uncheck(self, timeout=None):
        if self._count == 0:
            raise Exception("no element matches this locator")
        self._checked = False

    async def is_checked(self):
        return self._checked

    def locator(self, selector):
        # The real chain (text=X -> ancestor block -> inner control)
        # traverses real DOM; the fake just resolves straight through to
        # whatever terminal field the test registered under that text.
        return self

    async def wait_for(self, state=None, timeout=None):
        pass


class _FakePage:
    """Minimal stand-in for a Playwright Page - looks up a locator by
    whatever key (label text, placeholder text, or block text) it's
    asked for from a single dict, matching how the real page resolves
    any of them."""

    def __init__(self, fields: dict):
        self._fields = fields

    def get_by_label(self, label, exact=True):
        return self._fields.get(label, _FakeLocator(count=0))

    def get_by_placeholder(self, placeholder, exact=True):
        return self._fields.get(placeholder, _FakeLocator(count=0))

    def get_by_text(self, text, exact=True):
        return self._fields.get(text, _FakeLocator(count=0))

    def get_by_role(self, role, name=None, exact=False):
        return _FakeLocator(count=1, visible=True)


def test_set_field_by_label_fill_path_verifies_successfully():
    field = _FakeLocator(count=1, visible=True, value="")
    page = _FakePage({"LM Temperature": field})
    asyncio.run(ACEStepClient._set_field_by_label(page, "LM Temperature", "1.1"))
    assert asyncio.run(field.input_value()) == "1.1"


def test_set_field_by_label_raises_when_fill_is_cosmetic_only():
    # The exact confirmed bug class: fill() doesn't throw, but the field's
    # real value never actually changes (a Gradio combobox filter input
    # scenario) - must be caught, not silently trusted as success. Uses
    # the real "Key" field specifically, since it's looked up by its
    # placeholder text rather than its label (a pre-existing special
    # case - "Key"'s real accessible name is broken, concatenated with
    # its tooltip), and this is the exact field behind the real incident
    # that motivated this fix.
    field = _FakeLocator(count=1, visible=True, value="", fill_is_cosmetic=True)
    placeholder = ACEStepClient._PLACEHOLDER_FALLBACK_FOR_LABEL["Key"]
    page = _FakePage({placeholder: field})
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(ACEStepClient._set_field_by_label(page, "Key", "F Minor"))
    assert exc_info.value.code == ErrorCode.SYNTHESIS_FAILED
    assert "Key" in exc_info.value.message
    assert "F Minor" in exc_info.value.message


def test_set_field_by_label_numeric_tolerant_comparison():
    # Gradio number inputs commonly reformat ("150" -> "150.0") - this
    # must not be treated as a failed verification.
    field = _FakeLocator(count=1, visible=True, value="150.0")

    async def fake_fill(value, timeout=None):
        field._value = "150.0"

    field.fill = fake_fill
    page = _FakePage({"BPM (Beats Per Minute)": field})
    asyncio.run(ACEStepClient._set_field_by_label(page, "BPM (Beats Per Minute)", 150))


def test_set_field_by_label_boolean_path_verifies_successfully():
    field = _FakeLocator(count=1, visible=True, checked=False)
    page = _FakePage({"Use LoRA": field})
    asyncio.run(ACEStepClient._set_field_by_label(page, "Use LoRA", True))
    assert asyncio.run(field.is_checked()) is True


def test_set_field_by_label_raises_when_checkbox_does_not_actually_toggle():
    field = _FakeLocator(count=1, visible=True, checked=False)
    field.check = lambda timeout=None: asyncio.sleep(0)  # never actually flips _checked
    page = _FakePage({"Use LoRA": field})
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(ACEStepClient._set_field_by_label(page, "Use LoRA", True))
    assert exc_info.value.code == ErrorCode.SYNTHESIS_FAILED


def test_set_field_by_label_radio_fallback_verifies_successfully():
    main_field = _FakeLocator(count=1, visible=True, raise_on_select_option=True, raise_on_fill=True)
    radio_option = _FakeLocator(count=1, visible=True, checked=False)
    page = _FakePage({
        "Inference Method": main_field,
        "SDE": radio_option,
    })
    asyncio.run(ACEStepClient._set_field_by_label(page, "Inference Method", "SDE"))
    assert asyncio.run(radio_option.is_checked()) is True


def test_set_field_by_label_raises_when_radio_option_not_found():
    main_field = _FakeLocator(count=1, visible=True, raise_on_select_option=True, raise_on_fill=True)
    page = _FakePage({"Inference Method": main_field})
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(ACEStepClient._set_field_by_label(page, "Inference Method", "SDE"))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_set_field_by_label_seed_uses_block_text_fallback_and_unchecks_random_seed():
    # Regression test for a real confirmed incident: get_by_label("Seed",
    # exact=True) matches nothing at all against a live ACE-Step server -
    # not a wording change like "Key", a broken aria association with no
    # placeholder to fall back on either. Must resolve via the field's
    # own containing block's text instead, and "Random Seed" (checked by
    # default in the live UI) must get unchecked or the fill has no real
    # effect - this is what generate_vocal_track_takes relies on for
    # per-take seed variation.
    seed_field = _FakeLocator(count=1, visible=True, value="")
    random_seed_checkbox = _FakeLocator(count=1, visible=True, checked=True)
    page = _FakePage({
        "Seed": seed_field,  # resolved via get_by_text, not get_by_label
        "Random Seed": random_seed_checkbox,
    })
    asyncio.run(ACEStepClient._set_field_by_label(page, "Seed", "123456"))
    assert asyncio.run(seed_field.input_value()) == "123456"
    assert asyncio.run(random_seed_checkbox.is_checked()) is False


def test_set_field_by_label_raises_when_label_not_found_at_all():
    page = _FakePage({})
    with pytest.raises(SongForgeMCPError) as exc_info:
        asyncio.run(ACEStepClient._set_field_by_label(page, "Nonexistent Field", "value"))
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def _write_tone(path, duration_seconds, samplerate=48000):
    t = np.linspace(0, duration_seconds, int(duration_seconds * samplerate), endpoint=False)
    data = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), data, samplerate)


def test_returns_unchanged_path_when_already_short_enough(tmp_path):
    path = tmp_path / "short.wav"
    _write_tone(path, _REFERENCE_TRIM_WINDOW_SECONDS - 5)
    assert _trim_reference_for_key_coherence(str(path)) == str(path)


def test_trims_long_file_to_window_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "REFERENCE_AUDIO_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "long.wav"
    _write_tone(path, 200.0)

    trimmed = _trim_reference_for_key_coherence(str(path))

    assert trimmed != str(path)
    assert os.path.isfile(trimmed)
    info = sf.info(trimmed)
    duration = info.frames / info.samplerate
    assert duration == pytest.approx(_REFERENCE_TRIM_WINDOW_SECONDS, abs=0.1)


def test_trimmed_window_is_taken_from_the_middle_not_the_start(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "REFERENCE_AUDIO_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "long.wav"
    samplerate = 48000
    duration = 200.0
    # Silence everywhere except a distinctive tone right in the middle -
    # if the trim genuinely comes from the middle, the trimmed file should
    # contain real signal, not silence.
    total_frames = int(duration * samplerate)
    data = np.zeros(total_frames)
    mid_start = total_frames // 2 - samplerate
    mid_end = total_frames // 2 + samplerate
    t = np.linspace(0, 2.0, mid_end - mid_start, endpoint=False)
    data[mid_start:mid_end] = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), data, samplerate)

    trimmed = _trim_reference_for_key_coherence(str(path))
    trimmed_data, _ = sf.read(trimmed)
    assert np.abs(trimmed_data).max() > 0.1


def test_slugify_title_produces_hyphenated_lowercase_slug():
    assert _slugify_title("Hearts on a Wire!") == "hearts-on-a-wire"


def test_slugify_title_collapses_repeated_punctuation():
    assert _slugify_title("Hearts -- on   a Wire???") == "hearts-on-a-wire"


def test_slugify_title_truncates_to_max_length():
    long_title = "a" * 100
    assert _slugify_title(long_title, max_length=10) == "a" * 10


def test_slugify_title_returns_empty_for_punctuation_only():
    assert _slugify_title("!!!???") == ""


def test_acquire_generate_lock_succeeds_when_no_lock_exists(tmp_path):
    lock_path = str(tmp_path / "generate.lock")
    assert ACEStepClient._acquire_generate_lock(lock_path) is True
    assert os.path.isfile(lock_path)


def test_acquire_generate_lock_fails_when_already_held(tmp_path):
    lock_path = str(tmp_path / "generate.lock")
    assert ACEStepClient._acquire_generate_lock(lock_path) is True
    # Simulates a second, separate songforge-mcp process trying to drive
    # its own generation while one is already in flight elsewhere - must
    # not also succeed, otherwise two processes can drive Playwright
    # against the same single-GPU ACE-Step server concurrently.
    assert ACEStepClient._acquire_generate_lock(lock_path) is False


def test_acquire_generate_lock_reclaims_stale_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Timeouts, "GENERATION", 1.0)
    lock_path = str(tmp_path / "generate.lock")
    with open(lock_path, "w"):
        pass
    old_time = time.time() - 1000
    os.utime(lock_path, (old_time, old_time))

    assert ACEStepClient._acquire_generate_lock(lock_path) is True


def test_acquire_generate_lock_reclaims_immediately_when_holder_is_dead(tmp_path):
    lock_path = str(tmp_path / "generate.lock")
    dead_pid = 999999
    while psutil.pid_exists(dead_pid):
        dead_pid += 1
    with open(lock_path, "w") as f:
        f.write(str(dead_pid))
    # Fresh mtime (not stale by age) - only reclaimable because the PID
    # written into the lock file does not correspond to a live process,
    # which is exactly what happens when Claude Desktop is force-closed
    # mid-generation and the holding process never runs its cleanup.
    assert ACEStepClient._acquire_generate_lock(lock_path) is True


def test_acquire_generate_lock_does_not_reclaim_when_holder_is_alive(tmp_path):
    lock_path = str(tmp_path / "generate.lock")
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))
    assert ACEStepClient._acquire_generate_lock(lock_path) is False


def test_acquire_launch_lock_succeeds_when_no_lock_exists(tmp_path):
    lock_path = str(tmp_path / "launch.lock")
    assert ACEStepClient._acquire_launch_lock(lock_path) is True
    assert os.path.isfile(lock_path)


def test_acquire_launch_lock_fails_when_already_held(tmp_path):
    lock_path = str(tmp_path / "launch.lock")
    assert ACEStepClient._acquire_launch_lock(lock_path) is True
    # A second, separate attempt (simulating a different OS process) must
    # not also succeed - this is the exact bug this lock exists to close:
    # two processes both launching a competing ACE-Step server.
    assert ACEStepClient._acquire_launch_lock(lock_path) is False


def test_acquire_launch_lock_reclaims_stale_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Timeouts, "SERVER_STARTUP", 1.0)
    lock_path = str(tmp_path / "launch.lock")
    with open(lock_path, "w"):
        pass
    old_time = time.time() - 1000
    os.utime(lock_path, (old_time, old_time))

    # constants.Timeouts.SERVER_STARTUP is patched, but acestep_client
    # imported Timeouts by reference from the same module object, so the
    # patched class attribute is visible through that reference too.
    assert ACEStepClient._acquire_launch_lock(lock_path) is True


def test_log_last_activity_returns_zero_when_no_logs_exist(tmp_path):
    missing = str(tmp_path / "does_not_exist.txt")
    assert ACEStepClient._log_last_activity(missing) == 0.0


def test_log_last_activity_returns_newest_mtime_across_files(tmp_path):
    older = tmp_path / "stdout.txt"
    newer = tmp_path / "stderr.txt"
    older.write_text("x")
    newer.write_text("y")
    old_time = time.time() - 500
    os.utime(str(older), (old_time, old_time))

    result = ACEStepClient._log_last_activity(str(older), str(newer))

    assert result == pytest.approx(os.path.getmtime(str(newer)), abs=0.01)


def test_log_last_activity_ignores_missing_files_mixed_with_real_ones(tmp_path):
    real = tmp_path / "stderr.txt"
    real.write_text("data")
    missing = str(tmp_path / "stdout.txt")

    result = ACEStepClient._log_last_activity(missing, str(real))

    assert result == pytest.approx(os.path.getmtime(str(real)), abs=0.01)
