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

def load_main_module(base_dir=None):
    # load a fresh copy of main.py as a module named "app_main"
    spec = importlib.util.spec_from_file_location("app_main", str(MAIN_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # If tests provide a custom base_dir (tmp), override module globals so
    # the test operates on an isolated app-back copy instead of the repo.
    if base_dir is not None:
        try:
            module.BASE_DIR = str(base_dir)
            module.OUTPUT_MP_DIR = Path(module.BASE_DIR) / "mp" / "output_mp"
        except Exception:
            pass
        # Also adjust module __file__ so functions that compute BASE_DIR from
        # __file__ use the temporary base_dir instead of the repo location.
        try:
            module.__file__ = str(Path(base_dir) / "main.py")
        except Exception:
            pass
    # If tests requested an isolated base_dir, override renderer loader helper so
    # it uses the test base_dir (main.py computes BASE_DIR_LOCAL from __file__,
    # so we patch the helper to point to the temp location).
    if base_dir is not None:
        import importlib.util as _ilu
        from pathlib import Path as _P

        def _render_from_text_override(text: str):
            BASE = _P(base_dir)
            mp_script = BASE / "mp" / "run_pose_to_video_mediapipe.py"
            json_path = BASE / "mp" / "poses_mediapipe_video.json"
            if not mp_script.exists() or not json_path.exists():
                raise RuntimeError("Renderer or poses JSON missing")
            spec = _ilu.spec_from_file_location("mp_renderer", str(mp_script))
            if spec is None or getattr(spec, "loader", None) is None:
                raise RuntimeError("Could not create spec for renderer")
            try:
                mp_mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mp_mod)
            except Exception:
                import types as _types
                name = getattr(spec, "name", "mp_renderer")
                mp_mod = _types.ModuleType(name)
                spec.loader.exec_module(mp_mod)

            # Normalize text using module's formatting helper if present
            try:
                safe_text = module.format_text_for_renderer.normalize_text_for_renderer(text)
            except Exception:
                safe_text = text
            seq = mp_mod.text_to_pose_sequence(safe_text)
            if not seq:
                raise RuntimeError("Could not convert text to pose sequence")
            out_dir = BASE / "mp" / "output_mp"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / f"render_{int(time.time())}.mp4")
            video_path = mp_mod.render_sequence_from_json(str(json_path), seq, out_path=out_path, show=False, save=True)
            filename = Path(video_path).name
            download_url = f"/download_video/{filename}"
            return str(video_path), download_url

        # attach override into module
        try:
            module._render_from_text = _render_from_text_override
        except Exception:
            pass
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
    # prepare environment in a temporary app-back copy
    base_dir = tmp_path / "app_back"
    make_mp_environment(base_dir)
    module = load_main_module(base_dir=base_dir)
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
    base_dir = tmp_path / "app_back"
    make_mp_environment(base_dir)
    module = load_main_module(base_dir=base_dir)
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
    base_dir = tmp_path / "app_back"
    make_mp_environment(base_dir)
    module = load_main_module(base_dir=base_dir)
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
    base_dir = tmp_path / "app_back"
    module = load_main_module(base_dir=base_dir)
    mp_dir = Path(module.BASE_DIR) / "mp"
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