from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import importlib
import importlib.util
import time
from pathlib import Path
import uuid
import datetime
from typing import Optional, Tuple, Dict, Any
from metrics_logger import log_metrics
from src.components.audio_extractor import extract_audio_from_video
from src.components.speech_to_text import speech_to_text
from src.components import format_text_for_renderer
from src.components.downloader import download_youtube
import glob
import shutil

def _get_media_duration_seconds(path: str) -> Optional[float]:
    """Try to infer media duration (audio/video) in seconds using moviepy if available.

    Returns None if duration can't be determined.
    """
    p = Path(path)
    if not p.exists():
        return None

    # Try to import moviepy dynamically to avoid static import resolution issues.
    try:
        mp = importlib.import_module("moviepy.editor")
        AudioFileClip = getattr(mp, "AudioFileClip")
        VideoFileClip = getattr(mp, "VideoFileClip")
    except Exception:
        # Fallback to ffprobe/ffmpeg if moviepy isn't available.
        try:
            from shutil import which
            if which("ffprobe") is None:
                return None
            import subprocess
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(p),
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            dur = float(out.strip())
            return dur
        except Exception:
            return None

    # Try as audio first
    try:
        clip = AudioFileClip(str(p))
    except Exception:
        clip = None
    if clip is None:
        try:
            clip = VideoFileClip(str(p))
        except Exception:
            clip = None
    if clip is None:
        return None
    try:
        dur = float(clip.duration)
    finally:
        try:
            clip.close()
        except Exception:
            pass
    return dur

app = FastAPI()

# Allow CORS for local development (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_MP_DIR = Path(BASE_DIR) / "mp" / "output_mp"


def cleanup_generated_outputs(only_render_prefix=True):
    """Remove generated render files from the output_mp folder.

    If `only_render_prefix` is True (default) only files starting with
    'render_' will be removed (MP4 and common sidecars .ass/.srt). Otherwise
    all mp4/.ass/.srt files are removed.
    """
    try:
        OUTPUT_MP_DIR.mkdir(parents=True, exist_ok=True)
        if only_render_prefix:
            # Remove render_*.mp4 and matching sidecars
            for mp4 in OUTPUT_MP_DIR.glob("render_*.mp4"):
                try:
                    mp4.unlink()
                except Exception:
                    pass
                # remove sidecars with same stem
                stem = mp4.with_suffix("")
                for ext in ('.ass', '.srt'):
                    side = stem.with_suffix(ext)
                    try:
                        if side.exists():
                            side.unlink()
                    except Exception:
                        pass
        else:
            # Conservative removal of common generated files
            for pat in ("*.mp4", "*.ass", "*.srt"):
                for p in OUTPUT_MP_DIR.glob(pat):
                    try:
                        p.unlink()
                    except Exception:
                        pass
    except Exception:
        pass


@app.on_event("shutdown")
def _on_shutdown_cleanup():
    # Remove generated outputs on server shutdown to avoid leaving large files
    # Only remove files starting with the `render_` prefix to avoid deleting
    # other artifacts (e.g. cached poses or reference videos).
    cleanup_generated_outputs(only_render_prefix=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spanish sign-language interpretation API"}

def _render_from_text(text: str, return_timings: bool = False):
    """Helper: normalize text, import renderer and run it. Returns (video_path, download_url).

    If `return_timings=True` returns a 3-tuple `(video_path, download_url, timings)` where
    `timings` is a dict with keys `t_text_normalisation`, `t_pose_sequence`, `t_render`.
    """
    # allow optional flag by checking if caller passed a tuple-like second arg
    # (to keep backward compatibility when called without keyword)
    # Prefer explicit keyword usage; callers in this file pass return_timings=True.
    BASE_DIR_LOCAL = os.path.abspath(os.path.dirname(__file__))
    mp_script = Path(BASE_DIR_LOCAL) / "mp" / "run_pose_to_video_mediapipe.py"
    json_path = Path(BASE_DIR_LOCAL) / "mp" / "poses_mediapipe_video.json"

    if not mp_script.exists() or not json_path.exists():
        raise RuntimeError("Renderer or poses JSON missing")

    # Dynamic import (robust to custom loader specs used in tests)
    spec = importlib.util.spec_from_file_location("mp_renderer", str(mp_script))
    if spec is None or getattr(spec, "loader", None) is None:
        raise RuntimeError("Could not create spec for renderer")
    try:
        mp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp_mod)
    except Exception:
        # Some test fakes provide a loader with exec_module but no
        # create_module / full spec semantics. In that case, create a
        # plain module object and ask the loader to populate it.
        import types
        name = getattr(spec, "name", "mp_renderer")
        mp_mod = types.ModuleType(name)
        spec.loader.exec_module(mp_mod)


    # Measure sub-stages inside rendering so we can report timings
    t_text_normalisation = None
    t_pose_sequence = None
    t_render = None

    # Normalize text for renderer
    start_norm = time.perf_counter()
    safe_text = format_text_for_renderer.normalize_text_for_renderer(text)
    t_text_normalisation = time.perf_counter() - start_norm

    # Pose sequence
    start_pose = time.perf_counter()
    seq = mp_mod.text_to_pose_sequence(safe_text)
    t_pose_sequence = time.perf_counter() - start_pose
    if not seq:
        raise RuntimeError("Could not convert text to pose sequence")

    out_dir = Path(BASE_DIR_LOCAL) / "mp" / "output_mp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"render_{int(time.time())}.mp4")

    # Render (this may be the heaviest part)
    start_render = time.perf_counter()
    video_path = mp_mod.render_sequence_from_json(str(json_path), seq, out_path=out_path, show=False, save=True)
    t_render = time.perf_counter() - start_render

    filename = Path(video_path).name
    download_url = f"/download_video/{filename}"

    if return_timings:
        timings = {
            "t_text_normalisation": t_text_normalisation,
            "t_pose_sequence": t_pose_sequence,
            "t_render": t_render,
            "n_poses_rendered": len(seq) if seq is not None else None,
        }
        return str(video_path), download_url, timings

    return str(video_path), download_url


@app.post("/procesar_video/")
@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    # Instrumentation: collect timing and status metrics for feasibility study
    request_id = str(uuid.uuid4())
    start_total = time.perf_counter()
    t_download = None
    t_extract_audio = None
    t_asr = None
    t_text_normalisation = None
    t_pose_sequence = None
    t_render = None
    success = False
    error_stage = ""
    error_message_short = ""
    output_video_filename = ""

    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"temp_{file.filename}")
    # Save the uploaded video to a temporary file
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    audio_path = video_path + ".wav"
    text = ""
    input_length_seconds = None
    try:
        # Extract audio
        start_extract = time.perf_counter()
        try:
            extract_audio_from_video(video_path, audio_path)
        except Exception as e:
            error_stage = "audio_extraction"
            error_message_short = str(e)
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        t_extract_audio = time.perf_counter() - start_extract

        # Infer input duration from uploaded video
        input_length_seconds = _get_media_duration_seconds(video_path)

        # Convert audio to text
        start_asr = time.perf_counter()
        text = speech_to_text(audio_path)
        t_asr = time.perf_counter() - start_asr
    except HTTPException:
        raise
    except Exception as e:
        error_stage = error_stage or "other"
        error_message_short = str(e)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        # keep uploaded video for use as background; we'll remove it after rendering
        pass

    if not text:
        raise HTTPException(status_code=500, detail="Could not transcribe uploaded video")

    # Clean up audio file as well (we'll try to remove it later too)
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    # Render the output video from the transcribed text
    try:
        start_render = time.perf_counter()
        # Request internal timings from renderer
        video_path_out, download_url, inner_timings = _render_from_text(text, return_timings=True)
        t_render = time.perf_counter() - start_render
        # prefer inner timings when present
        t_text_normalisation = inner_timings.get("t_text_normalisation") if inner_timings else t_text_normalisation
        t_pose_sequence = inner_timings.get("t_pose_sequence") if inner_timings else t_pose_sequence
        output_video_filename = Path(video_path_out).name if video_path_out else ""
        success = True
    except HTTPException:
        # attempt to remove upload and audio before raising
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        raise
    except Exception as e:
        error_stage = error_stage or "render"
        error_message_short = str(e)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")
    finally:
        # Clean up uploaded video and extracted audio now that render finished (or failed)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        t_total = time.perf_counter() - start_total
        # extra computed fields
        n_letters_rendered = sum(1 for c in text if c.isalpha()) if text else 0
        n_poses_rendered = inner_timings.get("n_poses_rendered") if ('inner_timings' in locals() and inner_timings) else None
        output_video_duration_seconds = _get_media_duration_seconds(video_path_out) if ('video_path_out' in locals() and video_path_out) else None
        # Build metrics record
        record = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "request_id": request_id,
            "endpoint": "/procesar_video/",
            "input_type": "uploaded_file",
            "input_length_seconds": input_length_seconds if 'input_length_seconds' in locals() else None,
            "text_length_chars": len(text) if text else 0,
            "n_letters_rendered": n_letters_rendered,
            "success": success,
            "error_stage": error_stage,
            "error_message_short": error_message_short.replace("\n", " ") if error_message_short else "",
            "output_video_filename": output_video_filename,
            "output_video_duration_seconds": output_video_duration_seconds,
            "n_poses_rendered": n_poses_rendered,
            "t_total": t_total,
            "t_download": None,
            "t_extract_audio": t_extract_audio,
            "t_asr": t_asr,
            "t_text_normalisation": t_text_normalisation,
            "t_pose_sequence": t_pose_sequence,
            "t_render": t_render,
        }
        try:
            log_metrics(record)
        except Exception:
            pass

    return {"video_path": video_path_out, "download_url": download_url}


@app.post("/transcribe_youtube/")
async def transcribe_youtube(payload: dict):
    url = payload.get("url") or payload.get("link")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in payload")

    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Instrumentation: collect timing metrics for YouTube transcription pipeline
    request_id = str(uuid.uuid4())
    start_total = time.perf_counter()
    t_download = None
    t_extract_audio = None
    t_asr = None
    t_text_normalisation = None
    t_pose_sequence = None
    t_render = None
    success = False
    error_stage = ""
    error_message_short = ""
    output_video_filename = ""
    input_length_seconds = None
    downloaded = None

    try:
        # Download video
        start_download = time.perf_counter()
        downloaded = download_youtube(url, temp_dir)
        t_download = time.perf_counter() - start_download

        # Extract audio
        audio_path = str(Path(downloaded).with_suffix('.wav'))
        start_extract = time.perf_counter()
        extract_audio_from_video(downloaded, audio_path)
        t_extract_audio = time.perf_counter() - start_extract

        # Infer input length from downloaded video
        input_length_seconds = _get_media_duration_seconds(downloaded)

        # Transcribe
        start_asr = time.perf_counter()
        text = speech_to_text(audio_path)
        t_asr = time.perf_counter() - start_asr
    except Exception as e:
        # Cleanup any partial files
        try:
            if downloaded and os.path.exists(downloaded):
                os.remove(downloaded)
        except Exception:
            pass
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        error_stage = error_stage or "download"
        error_message_short = str(e)
        raise HTTPException(status_code=500, detail=f"Failed processing YouTube URL: {e}")
    finally:
        # keep downloaded video until after rendering (we will clean up later)
        pass

    if not text:
        raise HTTPException(status_code=500, detail="Could not transcribe YouTube video")

    try:
        # Render using the downloaded video and request inner timings
        start_render = time.perf_counter()
        video_path_out, download_url, inner_timings = _render_from_text(text, return_timings=True)
        t_render = time.perf_counter() - start_render
        t_text_normalisation = inner_timings.get("t_text_normalisation") if inner_timings else None
        t_pose_sequence = inner_timings.get("t_pose_sequence") if inner_timings else None
        output_video_filename = Path(video_path_out).name if video_path_out else ""
        success = True
    except Exception as e:
        error_stage = error_stage or "render"
        error_message_short = str(e)
        raise HTTPException(status_code=500, detail=f"Failed processing YouTube URL: {e}")
    finally:
        # Clean up downloaded video and audio
        try:
            if downloaded and os.path.exists(downloaded):
                os.remove(downloaded)
        except Exception:
            pass
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        t_total = time.perf_counter() - start_total
        record = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "request_id": request_id,
            "endpoint": "/transcribe_youtube/",
            "input_type": "youtube_url",
            "input_length_seconds": input_length_seconds,
            "text_length_chars": len(text) if text else 0,
            "n_letters_rendered": sum(1 for c in text if c.isalpha()) if text else 0,
            "success": success,
            "error_stage": error_stage,
            "error_message_short": error_message_short.replace("\n", " ") if error_message_short else "",
            "output_video_filename": output_video_filename,
            "output_video_duration_seconds": _get_media_duration_seconds(video_path_out) if ('video_path_out' in locals() and video_path_out) else None,
            "n_poses_rendered": inner_timings.get("n_poses_rendered") if ('inner_timings' in locals() and inner_timings) else None,
            "t_total": t_total,
            "t_download": t_download,
            "t_extract_audio": t_extract_audio,
            "t_asr": t_asr,
            "t_text_normalisation": t_text_normalisation,
            "t_pose_sequence": t_pose_sequence,
            "t_render": t_render,
        }
        try:
            log_metrics(record)
        except Exception:
            pass

    if not video_path_out:
        raise HTTPException(status_code=500, detail="Rendering failed")

    return {"video_path": video_path_out, "download_url": download_url}


# Endpoint: generate a pose-video from input text and return path/download URL
@app.post("/generate_from_text/")
async def generate_from_text(payload: dict):
    text = payload.get("text") or payload.get("texto")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload")

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    mp_script = Path(BASE_DIR) / "mp" / "run_pose_to_video_mediapipe.py"
    json_path = Path(BASE_DIR) / "mp" / "poses_mediapipe_video.json"

    if not mp_script.exists():
        raise HTTPException(status_code=500, detail=f"Renderer script not found: {mp_script}")
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Poses JSON not found: {json_path}")

    # Dynamically import the renderer module so we can call its functions
    try:
        spec = importlib.util.spec_from_file_location("mp_renderer", str(mp_script))
        if spec is None or getattr(spec, "loader", None) is None:
            raise RuntimeError("Could not create spec for renderer")
        try:
            mp_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mp_mod)
        except Exception:
            import types
            name = getattr(spec, "name", "mp_renderer")
            mp_mod = types.ModuleType(name)
            spec.loader.exec_module(mp_mod)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load renderer: {e}")

    # Instrumentation: measure normalization, pose generation and render timings
    request_id = str(uuid.uuid4())
    start_total = time.perf_counter()
    t_text_normalisation = None
    t_pose_sequence = None
    t_render = None
    success = False
    error_stage = ""
    error_message_short = ""
    output_video_filename = ""

    # Build pose sequence and output path with timings
    try:
        start_norm = time.perf_counter()
        safe_text = format_text_for_renderer.normalize_text_for_renderer(text)
        t_text_normalisation = time.perf_counter() - start_norm

        start_pose = time.perf_counter()
        seq = mp_mod.text_to_pose_sequence(safe_text)
        t_pose_sequence = time.perf_counter() - start_pose
    except Exception as e:
        error_stage = "pose_generation"
        error_message_short = str(e)
        raise HTTPException(status_code=400, detail="Could not convert text to a pose sequence")

    out_dir = Path(BASE_DIR) / "mp" / "output_mp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"render_{int(time.time())}.mp4")

    try:
        start_render = time.perf_counter()
        video_path = mp_mod.render_sequence_from_json(str(json_path), seq, out_path=out_path, show=False, save=True)
        t_render = time.perf_counter() - start_render
        success = True
        output_video_filename = Path(video_path).name
    except Exception as e:
        error_stage = "render"
        error_message_short = str(e)
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")
    finally:
        t_total = time.perf_counter() - start_total
        # compute extra metrics
        n_letters_rendered = sum(1 for c in text if c.isalpha()) if text else 0
        n_poses_rendered = len(seq) if 'seq' in locals() and seq is not None else None
        output_video_duration_seconds = _get_media_duration_seconds(video_path) if ('video_path' in locals() and video_path) else None

        record = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "request_id": request_id,
            "endpoint": "/generate_from_text/",
            "input_type": "text",
            "input_length_seconds": None,
            "text_length_chars": len(text) if text else 0,
            "n_letters_rendered": n_letters_rendered,
            "success": success,
            "error_stage": error_stage,
            "error_message_short": error_message_short.replace("\n", " ") if error_message_short else "",
            "output_video_filename": output_video_filename,
            "output_video_duration_seconds": output_video_duration_seconds,
            "n_poses_rendered": n_poses_rendered,
            "t_total": t_total,
            "t_download": None,
            "t_extract_audio": None,
            "t_asr": None,
            "t_text_normalisation": t_text_normalisation,
            "t_pose_sequence": t_pose_sequence,
            "t_render": t_render,
        }
        try:
            log_metrics(record)
        except Exception:
            pass

    # Provide a download endpoint URL for the front-end to fetch the generated video
    filename = Path(video_path).name
    download_url = f"/download_video/{filename}"
    return {"video_path": str(video_path), "download_url": download_url}


@app.get("/download_video/{filename}")
def download_video(filename: str):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    p = Path(BASE_DIR) / "mp" / "output_mp" / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="video/mp4", filename=filename)