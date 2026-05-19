import cv2
import numpy as np
import pytest
from pathlib import Path
from gpu_server.droideka_src.make_video import make_video


def _write_jpeg(path: Path, color: tuple = (0, 128, 255)) -> None:
    """Write a solid-colour 960×288 JPEG for test fixtures."""
    img = np.full((288, 960, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_empty_dir_exits(tmp_path):
    with pytest.raises(SystemExit):
        make_video(tmp_path, tmp_path / "out.mp4")


def test_creates_mp4_and_returns_count(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        _write_jpeg(frames / f"{i:06d}.jpg", color=(i * 80, 0, 0))
    out = tmp_path / "out.mp4"
    count = make_video(frames, out, fps=5.0)
    assert count == 3
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_nested_dir_created(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_jpeg(frames / "000001.jpg")
    out = tmp_path / "nested" / "sub" / "out.mp4"
    make_video(frames, out, fps=5.0)
    assert out.exists()


def test_frames_processed_in_sorted_order(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    # Write three frames in non-alphabetical order; all must be processed.
    for name in ["000003.jpg", "000001.jpg", "000002.jpg"]:
        _write_jpeg(frames / name)
    count = make_video(frames, tmp_path / "out.mp4", fps=5.0)
    assert count == 3
