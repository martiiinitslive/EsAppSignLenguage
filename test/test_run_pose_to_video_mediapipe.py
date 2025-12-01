import sys
from pathlib import Path
import types
import json
import pytest

import importlib.util

# Load the module from the same directory as this test file
MODULE_PATH = Path(__file__).resolve().parent / "run_pose_to_video_mediapipe.py"
spec = importlib.util.spec_from_file_location("run_pose_to_video_mediapipe", str(MODULE_PATH))
mod = importlib.util.module_from_spec(spec)
sys.modules["run_pose_to_video_mediapipe"] = mod
spec.loader.exec_module(mod)


def test_pose_to_char_basic_cases():
    assert mod.pose_to_char(None) == ""
    assert mod.pose_to_char("SPACE") == " "
    assert mod.pose_to_char("period") == "."
    assert mod.pose_to_char("COMMA") == ","
    assert mod.pose_to_char("A") == "A"
    assert mod.pose_to_char("Hello") == "H"
    assert mod.pose_to_char(123) == "123"


def test_text_to_pose_sequence_capitalization_and_punctuation():
    seq = mod.text_to_pose_sequence("Hi.")
    assert seq == ["H", "i", "PERIOD"]

    seq2 = mod.text_to_pose_sequence("HELLO WORLD")
    # H should be uppercase, following letters lowered except first after space becomes uppercase rule in renderer
    # text_to_pose_sequence preserves letters but rules of upper/lower are applied there: first letter uppercase, others lowercase
    assert seq2[0] == "H"
    assert "SPACE" in seq2
    assert seq2[-1] == "D"

    # Diacritics removed
    seq3 = mod.text_to_pose_sequence("Áb.")
    assert seq3 == ["A", "b", "PERIOD"]

    # None input yields empty list
    assert mod.text_to_pose_sequence(None) == []


def test__format_ass_time_and_seq_from_string():
    assert mod._format_ass_time(123.456) == "0:02:03.46"
    assert mod._format_ass_time(0.0) == "0:00:00.00"
    assert mod._seq_from_poses_string("A,B,C") == ["A", "B", "C"]
    assert mod._seq_from_poses_string("  X ,  Y,,") == ["X", "Y"]


def test__convert_landmarks_to_pixels_minimum_points_and_pixels():
    # fewer than 5 points -> returns (None, None)
    small = [{"x": 0.1, "y": 0.1}] * 4
    pts, depths = mod._convert_landmarks_to_pixels(small, 100, 200)
    assert pts is None and depths is None

    # 5 valid dict landmarks should return pixel coordinates and depths
    landmarks = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.25, "y": 0.25, "z": 0.1},
        {"x": 0.5, "y": 0.5, "z": 0.2},
        {"x": 0.75, "y": 0.75, "z": 0.3},
        {"x": 1.0, "y": 1.0, "z": 0.4},
    ]
    pts2, depths2 = mod._convert_landmarks_to_pixels(landmarks, 101, 201)
    assert pts2 is not None and depths2 is not None
    assert len(pts2) == 5
    # verify conversion arithmetic roughly
    assert pts2[0][0] == 0 and pts2[0][1] == 0
    assert pts2[-1][0] == int(round(1.0 * (101 - 1)))
    assert pts2[-1][1] == int(round(1.0 * (201 - 1)))
    assert depths2[0] == 0.0 and depths2[-1] == pytest.approx(0.4)


def test__get_size_with_meta_and_args():
    meta = {"denormalization": {"image_width": 800, "image_height": 600}}
    args = types.SimpleNamespace(width=None, height=None)
    w, h = mod._get_size(meta, args)
    assert (w, h) == (800, 600)

    # explicit args override meta
    args2 = types.SimpleNamespace(width=320, height=240)
    w2, h2 = mod._get_size(meta, args2)
    assert (w2, h2) == (320, 240)

    # no meta and no args -> defaults
    w3, h3 = mod._get_size({}, types.SimpleNamespace(width=None, height=None))
    assert (w3, h3) == (1920, 1080)


def test_generate_ass_subtitles_writes_ass_and_srt(tmp_path):
    # small sequence to keep times simple
    sequence = ["H", "i", "SPACE", "PERIOD"]
    fps_out = 10
    frames_per_pose = [1, 1, 1, 1]
    ass_path = tmp_path / "out.ass"

    # ensure any attributes used by generator are set to known values
    try:
        setattr(mod.generate_ass_subtitles, "img_width", 640)
        setattr(mod.generate_ass_subtitles, "img_height", 360)
    except Exception:
        pass

    # call generator
    mod.generate_ass_subtitles(sequence, fps_out, frames_per_pose, str(ass_path), start_frame=1, write_ass=True)

    ass_file = ass_path
    srt_file = ass_path.with_suffix(".srt")
    assert ass_file.exists(), "ASS file was not created"
    assert srt_file.exists(), "SRT file was not created"

    ass_text = ass_file.read_text(encoding="utf-8")
    srt_text = srt_file.read_text(encoding="utf-8")

    # ASS should contain header and Dialogue lines equal to number of sequence entries
    assert "[Script Info]" in ass_text
    assert ass_text.count("Dialogue:") == len(sequence)

    # SRT should contain indexed blocks from 1..n
    # count occurrences of "1\n" index lines by parsing more robustly:
    srt_blocks = [b for b in srt_text.strip().split("\n\n") if b.strip()]
    assert len(srt_blocks) == len(sequence)
    # verify the first block starts with "1"
    assert srt_blocks[0].splitlines()[0].strip() == "1"


def test_cleanup_sidecars_removes_and_keeps(tmp_path):
    ass = tmp_path / "video.ass"
    srt = tmp_path / "video.srt"
    ass.write_text("dummy", encoding="utf-8")
    srt.write_text("dummy", encoding="utf-8")

    # remove sidecars
    res = mod.cleanup_sidecars(str(ass), keep_sidecars=False)
    assert res is True
    assert not ass.exists()
    assert not srt.exists()

    # recreate and call with keep_sidecars True
    ass.write_text("dummy", encoding="utf-8")
    srt.write_text("dummy", encoding="utf-8")
    res2 = mod.cleanup_sidecars(str(ass), keep_sidecars=True)
    assert res2 is True
    assert ass.exists()
    assert srt.exists()