import importlib.util
from pathlib import Path
import time


def _make_dummy_spec(tmp_dir):
    class Loader:
        def exec_module(self, module):
            def text_to_pose_sequence(text):
                return ["A"] if text else []

            def render_sequence_from_json(json_path, seq, out_path=None, show=False, save=True, **kwargs):
                if out_path is None:
                    out_path = str(Path(tmp_dir) / f"render_{int(time.time())}.mp4")
                Path(out_path).write_bytes(b"MP4DATA")
                return out_path

            module.text_to_pose_sequence = text_to_pose_sequence
            module.render_sequence_from_json = render_sequence_from_json

    class Spec:
        loader = Loader()

    return Spec()


def test_process_video_upload(client, monkeypatch, tmp_output_dir, app_module, tmp_path):
    # Patch audio extractor and speech_to_text
    def fake_extract(video_path, audio_path):
        Path(audio_path).write_bytes(b"WAV")

    def fake_s2t(audio_path):
        return "HELLO"

    monkeypatch.setattr(app_module, "extract_audio_from_video", fake_extract)
    monkeypatch.setattr(app_module, "speech_to_text", fake_s2t)

    # Monkeypatch renderer import
    spec_obj = _make_dummy_spec(tmp_path)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda name, path: spec_obj)

    # upload a fake video
    files = {"file": ("video.mp4", b"FAKEVIDEO", "video/mp4")}
    resp = client.post("/procesar_video/", files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "video_path" in data and Path(data["video_path"]).exists()


def test_transcribe_youtube_endpoint(client, monkeypatch, tmp_output_dir, app_module, tmp_path):
    # Patch download_youtube to create a dummy video file
    def fake_download(url, out_dir):
        out = Path(out_dir) / "yt_sample.mp4"
        out.write_bytes(b"FAKEYT")
        return str(out)

    def fake_extract(video_path, audio_path):
        Path(audio_path).write_bytes(b"WAV")

    def fake_s2t(audio_path):
        return "WORLD"

    monkeypatch.setattr(app_module, "download_youtube", fake_download)
    monkeypatch.setattr(app_module, "extract_audio_from_video", fake_extract)
    monkeypatch.setattr(app_module, "speech_to_text", fake_s2t)

    # Monkeypatch renderer import
    spec_obj = _make_dummy_spec(tmp_path)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda name, path: spec_obj)

    resp = client.post("/transcribe_youtube/", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "video_path" in data and Path(data["video_path"]).exists()
