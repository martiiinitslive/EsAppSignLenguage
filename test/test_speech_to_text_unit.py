import pytest
from pathlib import Path

# We'll import the module and monkeypatch internal recognizer calls to avoid network
from src.components import speech_to_text as st_mod


def test_speech_to_text_google_path(monkeypatch, tmp_path):
    # Monkeypatch the internal _google_transcribe implementation to avoid
    # importing speech_recognition or calling external services.
    def fake_google(path, language="es-ES"):
        return "FAKE TRANSCRIPT"

    monkeypatch.setattr(st_mod, '_google_transcribe', fake_google)

    # Create dummy wav file
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')

    res = st_mod.speech_to_text(str(wav), prefer='google')
    assert res == "FAKE TRANSCRIPT"


def test_speech_to_text_whisper_fallback(monkeypatch, tmp_path):
    # Monkeypatch the internal _whisper_transcribe to simulate faster-whisper
    def fake_whisper(path):
        return 'WHISPER TEXT'

    monkeypatch.setattr(st_mod, '_whisper_transcribe', fake_whisper)

    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')

    out = st_mod.speech_to_text(str(wav), prefer='whisper')
    assert out == 'WHISPER TEXT'
