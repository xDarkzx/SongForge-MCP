from unittest.mock import MagicMock, patch

import pytest

from scripts.detect_gpu_checkpoint import detect_vram_gb, main, pick_checkpoint


@pytest.mark.parametrize(
    "vram_gb,expected",
    [
        (2.0, "acestep-v15-turbo"),
        (6.0, "acestep-v15-turbo"),
        (7.9, "acestep-v15-turbo"),
        (8.0, "acestep-v15-sft"),
        (12.0, "acestep-v15-sft"),
        (19.9, "acestep-v15-sft"),
        (20.0, "acestep-v15-xl-sft"),
        (24.0, "acestep-v15-xl-sft"),
        (48.0, "acestep-v15-xl-sft"),
    ],
)
def test_pick_checkpoint_boundaries(vram_gb, expected):
    assert pick_checkpoint(vram_gb) == expected


def test_detect_vram_gb_returns_total_memory_in_gb():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_props = MagicMock()
    fake_props.total_memory = 8 * 1024**3
    fake_torch.cuda.get_device_properties.return_value = fake_props

    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert detect_vram_gb() == pytest.approx(8.0)


def test_detect_vram_gb_raises_when_no_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="no CUDA-capable GPU"):
            detect_vram_gb()


def test_detect_vram_gb_raises_when_torch_not_installed():
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(ImportError):
            detect_vram_gb()


def test_main_prints_checkpoint_to_stdout_and_returns_zero(capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_props = MagicMock()
    fake_props.total_memory = 6 * 1024**3
    fake_torch.cuda.get_device_properties.return_value = fake_props

    with patch.dict("sys.modules", {"torch": fake_torch}):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "acestep-v15-turbo"


def test_main_returns_nonzero_and_prints_nothing_to_stdout_on_failure(capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "detection failed" in captured.err.lower()
