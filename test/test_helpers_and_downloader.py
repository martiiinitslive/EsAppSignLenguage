import pytest
from pathlib import Path


def test_normalize_text_basic():
    # import inside test so sys.path modifications in fixtures apply
    from src.components.format_text_for_renderer import normalize_text_for_renderer
    s = "¡Hola, señor! ¿Cómo estás?"
    out = normalize_text_for_renderer(s)
    # accents removed and punctuation normalized
    assert "Hola" in out
    assert "señor" not in out or "senor" in out


def test_downloader_missing_ytdlp(monkeypatch):
    from src.components import downloader
    # Simulate yt-dlp not present
    monkeypatch.setattr(downloader, "shutil", downloader.shutil)
    monkeypatch.setattr(downloader.shutil, "which", lambda x: None)
    with pytest.raises(RuntimeError):
        downloader.download_youtube("https://youtu.be/dQw4w9WgXcQ", ".")
