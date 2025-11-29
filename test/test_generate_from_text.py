import os
import importlib
import importlib.util
from pathlib import Path
import time


def _make_dummy_spec(tmp_dir):
    class Loader:
        def exec_module(self, module):
            # provide required functions
            def text_to_pose_sequence(text):
                return ["A"] if text else []

            def render_sequence_from_json(json_path, seq, out_path=None, show=False, save=True, **kwargs):
                # create a dummy mp4 file at out_path
                if out_path is None:
                    out_path = str(Path(tmp_dir) / f"render_{int(time.time())}.mp4")
                Path(out_path).write_bytes(b"MP4DATA")
                return out_path

            module.text_to_pose_sequence = text_to_pose_sequence
            module.render_sequence_from_json = render_sequence_from_json

    class Spec:
        loader = Loader()

    return Spec()


def test_generate_from_text_endpoint(client, monkeypatch, tmp_output_dir, app_module, tmp_path):
    # Monkeypatch dynamic import used in generate_from_text
    tmp_dir = tmp_path
    spec_obj = _make_dummy_spec(tmp_dir)

    def fake_spec_from_file_location(name, path):
        return spec_obj

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)

    payload = {"text": "HELLO"}
    resp = client.post("/generate_from_text/", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "video_path" in data and "download_url" in data
    # file should exist
    assert Path(data["video_path"]).exists()

    # test download endpoint
    filename = Path(data["video_path"]).name
    dl = client.get(f"/download_video/{filename}")
    assert dl.status_code == 200
    assert dl.headers.get("content-type") == "video/mp4"
