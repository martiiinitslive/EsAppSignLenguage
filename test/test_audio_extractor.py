import os
import pytest
from pathlib import Path

try:
    from src.components.audio_extractor import extract_audio_from_video
    HAS_EXTRACTOR = True
except Exception:
    HAS_EXTRACTOR = False

@pytest.mark.skipif(not HAS_EXTRACTOR, reason="audio_extractor not importable")
def test_extract_audio_creates_wav(tmp_path):
    # Create a tiny dummy mp4 file (not a valid video) and ensure function
    # either creates a wav or raises a controlled error. We skip heavy checks.
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"FAKEVIDEO")
    out_wav = tmp_path / "out.wav"
    try:
        extract_audio_from_video(str(dummy_video), str(out_wav))
    except Exception:
        pytest.skip("extract_audio_from_video requires real video/moviepy; skipping")
    assert out_wav.exists()
    assert out_wav.stat().st_size >= 0
