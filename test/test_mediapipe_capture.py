import pytest
from pathlib import Path
import importlib


def _module_path():
    # Return the path to the mediapipe_capture_video module inside app-back/mp/src
    base = Path(__file__).resolve().parent.parent
    return base / 'app-back' / 'mp' / 'src' / 'mediapipe_capture_video.py'


def test_iter_video_dirs_errors(tmp_path):
    # Import module by path dynamically
    mod_path = _module_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location('mpcap', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # _iter_video_dirs should raise for missing dir
    with pytest.raises(FileNotFoundError):
        list(module._iter_video_dirs(str(tmp_path / 'does_not_exist')))


def test_iter_video_dirs_structure(tmp_path):
    # Create structure A/video1.mp4, B/video2.mp4
    base = tmp_path / 'dataset'
    a = base / 'A'
    b = base / 'B'
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / 'v1.mp4').write_bytes(b'X')
    (b / 'v2.mp4').write_bytes(b'Y')

    # Import module and call iterator
    import importlib.util
    spec = importlib.util.spec_from_file_location('mpcap', str(_module_path()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    items = list(module._iter_video_dirs(str(base)))
    assert any(letter == 'A' for letter, files in items)
    assert any(letter == 'B' for letter, files in items)


def test_extract_hand_landmarks_from_video_missing_file():
    import importlib.util
    spec = importlib.util.spec_from_file_location('mpcap', str(_module_path()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(FileNotFoundError):
        module.extract_hand_landmarks_from_video('no_such_file.mp4')
