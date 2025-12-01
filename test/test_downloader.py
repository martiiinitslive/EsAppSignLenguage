import sys
from pathlib import Path
import subprocess
import shutil
import importlib.util
import types

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / 'app-back'))

from src.components.downloader import download_youtube


def test_missing_yt_dlp(monkeypatch, tmp_path):
    # Simulate yt-dlp not present
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    try:
        with __import__('pytest').raises(RuntimeError):
            download_youtube('https://youtu.be/dQw4w9WgXcQ', str(tmp_path))
    finally:
        # restore which if needed
        pass


def test_subprocess_failure(monkeypatch, tmp_path):
    # Simulate yt-dlp present but subprocess fails
    monkeypatch.setattr(shutil, 'which', lambda name: 'yt-dlp')

    class FakeResult:
        def __init__(self):
            self.returncode = 1
            self.stdout = 'error'

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeResult())

    import pytest
    with pytest.raises(RuntimeError):
        download_youtube('https://youtu.be/dQw4w9WgXcQ', str(tmp_path))


def test_happy_path_returns_latest_file(monkeypatch, tmp_path):
    # Simulate yt-dlp present and subprocess succeeds, and a file exists in out_dir
    monkeypatch.setattr(shutil, 'which', lambda name: 'yt-dlp')

    class FakeResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = 'ok'

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeResult())

    # create dummy file in out_dir
    f = tmp_path / 'abcd.mp4'
    f.write_bytes(b'X')

    res = download_youtube('https://youtu.be/dQw4w9WgXcQ', str(tmp_path))
    assert str(f) == res
