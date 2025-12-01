import importlib
from pathlib import Path
import sys
import shutil
import os
import time
import json
import pytest
from fastapi.testclient import TestClient

import importlib.util

THIS_DIR = Path(__file__).parent
# Point to the actual backend entry `app-back/main.py`
APP_BACK_DIR = THIS_DIR.parent / "app-back"
MAIN_PATH = APP_BACK_DIR / "main.py"

def load_main_module():
    # load a fresh copy of main.py as a module named "app_main"
    spec = importlib.util.spec_from_file_location("app_main", str(MAIN_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make_mp_environment(base_dir: Path):
    mp_dir = base_dir / "mp"
    out_dir = mp_dir / "output_mp"
    mp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # poses json
    poses = mp_dir / "poses_mediapipe_video.json"
    poses.write_text(json.dumps({"poses": []}))
    # fake renderer script
    script = mp_dir / "run_pose_to_video_mediapipe.py"
    script.write_text(
        "def text_to_pose_sequence(text):\n"
        "    # return a simple non-empty sequence for any non-empty text\n"
        "    if not text:\n"
        "        return []\n"
        "    return [{'pose': text}]\n\n"
        "def render_sequence_from_json(json_path, seq, out_path=None, show=False, save=True):\n"
        "    # create a fake mp4 file and return its path\n"
        "    out = out_path or 'render_test.mp4'\n"
        "    with open(out, 'wb') as f:\n"
        "        f.write(b'FAKEVIDEO')\n"
        "    return out\n"
    )

def cleanup_dir(path: Path):
    try:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    except Exception:
        pass

def ensure_removed(path: Path):
    try:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    except Exception:
        pass

def test_read_root():
    module = load_main_module()
    client = TestClient(module.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Spanish sign-language interpretation API"}

def test_generate_from_text_success_and_download(tmp_path):
    # prepare environment
    module = load_main_module()
    base_dir = APP_BACK_DIR
    make_mp_environment(base_dir)
    client = TestClient(module.app)
    payload = {"text": "hola mundo"}
    r = client.post("/generate_from_text/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "video_path" in data and "download_url" in data
    video_path = Path(data["video_path"])
    assert video_path.exists()
    # test download endpoint serves file
    filename = video_path.name
    r2 = client.get(f"/download_video/{filename}")
    assert r2.status_code == 200
    assert r2.content == b"FAKEVIDEO"
    # cleanup
    module.cleanup_generated_outputs(only_render_prefix=False)
    ensure_removed(base_dir / "mp")

def test_generate_from_text_missing_text():
    module = load_main_module()
    client = TestClient(module.app)
    r = client.post("/generate_from_text/", json={})
    assert r.status_code == 400
    assert "Missing 'text' in payload" in r.json().get("detail", "")

def test_download_video_not_found():
    module = load_main_module()
    client = TestClient(module.app)
    r = client.get("/download_video/nonexistent.mp4")
    assert r.status_code == 404

def test_transcribe_youtube_success(tmp_path):
    module = load_main_module()
    base_dir = APP_BACK_DIR
    make_mp_environment(base_dir)
    # prepare fake downloaded file path
    temp_dir = base_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    downloaded = temp_dir / "yt_test.mp4"
    downloaded.write_bytes(b"VIDEO")
    # monkeypatch functions on module
    module.download_youtube = lambda url, dest: str(downloaded)
    def fake_extract(src, dst):
        Path(dst).write_bytes(b"AUDIO")
    module.extract_audio_from_video = fake_extract
    module.speech_to_text = lambda audio_path: "texto transcrito"
    client = TestClient(module.app)
    r = client.post("/transcribe_youtube/", json={"url": "http://fake"})
    assert r.status_code == 200
    data = r.json()
    assert "video_path" in data and "download_url" in data
    # cleanup
    module.cleanup_generated_outputs(only_render_prefix=False)
    ensure_removed(base_dir / "mp")
    ensure_removed(temp_dir)

def test_process_video_success(tmp_path):
    module = load_main_module()
    base_dir = APP_BACK_DIR
    make_mp_environment(base_dir)
    # monkeypatch audio extraction and speech_to_text
    def fake_extract(src, dst):
        Path(dst).write_bytes(b"AUDIO")
    module.extract_audio_from_video = fake_extract
    module.speech_to_text = lambda audio_path: "hola desde video"
    client = TestClient(module.app)
    files = {"file": ("input.mp4", b"VIDCONTENT")}
    r = client.post("/process_video/", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "video_path" in data and "download_url" in data
    # cleanup
    module.cleanup_generated_outputs(only_render_prefix=False)
    ensure_removed(base_dir / "mp")
    ensure_removed(base_dir / "temp")

def test_cleanup_generated_outputs_behaviour(tmp_path):
    module = load_main_module()
    base_dir = APP_BACK_DIR
    mp_dir = base_dir / "mp"
    out_dir = mp_dir / "output_mp"
    out_dir.mkdir(parents=True, exist_ok=True)
    # create files
    (out_dir / "render_1.mp4").write_bytes(b"x")
    (out_dir / "render_1.ass").write_text("a")
    (out_dir / "keep.mp4").write_bytes(b"x")
    (out_dir / "other.ass").write_text("b")
    # call cleanup only render prefix
    module.cleanup_generated_outputs(only_render_prefix=True)
    assert not (out_dir / "render_1.mp4").exists()
    assert not (out_dir / "render_1.ass").exists()
    assert (out_dir / "keep.mp4").exists()
    assert (out_dir / "other.ass").exists()
    # now remove all
    module.cleanup_generated_outputs(only_render_prefix=False)
    assert not (out_dir / "keep.mp4").exists()
    assert not (out_dir / "other.ass").exists()
    ensure_removed(mp_dir)