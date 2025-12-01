import importlib.util
from pathlib import Path
import sys
import types

repo = Path(__file__).resolve().parent.parent

# Load renderer module by path
mod_path = repo / 'app-back' / 'mp' / 'run_pose_to_video_mediapipe.py'
spec = importlib.util.spec_from_file_location('rpp', str(mod_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lerp():
    assert abs(module.lerp(0.0, 10.0, 0.5) - 5.0) < 1e-6
    assert module.lerp(0, 1, 0) == 0


def test_sanitize_pose_name():
    s = 'Pose: A/B\\C?*"<>|'
    out = module._sanitize_pose_name(s)
    # no forbidden chars
    for ch in (' ', '/', '\\', ':', '*', '?', '"', "'", '<', '>', '|'):
        assert ch not in out


def test_cache_dir_and_has_frames(tmp_path):
    cache = module._cache_dir_for('A', 320, 240, 'realistic')
    # create cache and files
    cache.mkdir(parents=True, exist_ok=True)
    p1 = cache / 'frame_0001.png'
    p1.write_bytes(b'X')
    assert module._cache_has_frames(cache) is True

    # empty dir
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert module._cache_has_frames(empty) is False


def test_cleanup_sidecars(tmp_path):
    ass = tmp_path / 'video.ass'
    srt = tmp_path / 'video.srt'
    ass.write_text('ASS')
    srt.write_text('SRT')
    # remove sidecars
    ok = module.cleanup_sidecars(str(ass), keep_sidecars=False)
    assert ok is True
    assert not ass.exists() and not srt.exists()

    # recreate and keep
    ass.write_text('ASS')
    srt.write_text('SRT')
    ok2 = module.cleanup_sidecars(str(ass), keep_sidecars=True)
    assert ok2 is True
    assert ass.exists() and srt.exists()


def test_get_size_prefers_args_and_meta():
    class A: pass
    args = A()
    args.width = None
    args.height = None
    meta = {'denormalization': {'image_width': 1280, 'image_height': 720}}
    w, h = module._get_size(meta, args)
    assert (w, h) == (1280, 720)

    args.width = 640
    args.height = 480
    w2, h2 = module._get_size(meta, args)
    assert (w2, h2) == (640, 480)
