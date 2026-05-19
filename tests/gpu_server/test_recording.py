from pathlib import Path
import pytest
from gpu_server.droideka_src.recording import record_frame


def test_creates_file_with_correct_name(tmp_path):
    record_frame(tmp_path, 42, b"\xff\xd8\xff", active=True)
    out = tmp_path / "000042.jpg"
    assert out.exists()
    assert out.read_bytes() == b"\xff\xd8\xff"


def test_zero_padded_six_digits(tmp_path):
    record_frame(tmp_path, 1, b"data", active=True)
    assert (tmp_path / "000001.jpg").exists()


def test_inactive_skips_write(tmp_path):
    record_frame(tmp_path, 1, b"data", active=False)
    assert not (tmp_path / "000001.jpg").exists()


def test_handles_missing_directory_gracefully(tmp_path):
    bad_dir = tmp_path / "no_such_dir"
    # Must not raise even though the directory does not exist.
    record_frame(bad_dir, 1, b"data", active=True)
