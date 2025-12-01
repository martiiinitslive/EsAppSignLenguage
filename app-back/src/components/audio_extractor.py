"""app-back/src/components/audio_extractor.py
Utility to extract audio from a media file.

This now supports both video files (mp4, mov, etc.) and audio-only files
(m4a, mp3, wav...). It uses MoviePy's ``AudioFileClip`` which reads the audio
track from either a video or an audio file. This avoids trying to read video
frames for audio-only uploads.
"""

import shutil
import os
import subprocess
from pathlib import Path


def extract_audio_from_video(media_path, output_audio_path):
    """
    Extract audio from a media file (video or audio) and save to
    ``output_audio_path``.

    Args:
        media_path: path to input media file (video or audio)
        output_audio_path: where the extracted audio will be saved
    """
    # If input already looks like an audio file and the output is the same
    # extension, just copy it for speed.
    _, in_ext = os.path.splitext(media_path)
    _, out_ext = os.path.splitext(output_audio_path)
    in_ext = in_ext.lower()
    out_ext = out_ext.lower()
    audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    if in_ext in audio_exts and in_ext == out_ext:
        shutil.copyfile(media_path, output_audio_path)
        try:
            print(f"[AUDIO] Copied audio file to: {output_audio_path}")
        except Exception:
            pass
        return

    # Prefer using ffmpeg CLI for robust handling of audio-only files.
    ffmpeg_bin = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    if ffmpeg_bin:
        # Ensure output dir exists
        out_dir = Path(output_audio_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build ffmpeg command: drop video (-vn), convert to WAV mono 16kHz
        cmd = [ffmpeg_bin, '-y', '-i', str(media_path), '-vn', '-ac', '1', '-ar', '16000', str(output_audio_path)]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
            try:
                print(f"[AUDIO] ffmpeg extracted audio to: {output_audio_path}")
            except Exception:
                pass
            return
        except subprocess.CalledProcessError as e:
            # ffmpeg failed; include stderr for diagnostics and fall back to MoviePy
            err = e.stderr or e.stdout or str(e)
            print(f"[AUDIO] ffmpeg failed: {err}")

    # Fallback: use MoviePy's AudioFileClip if ffmpeg CLI isn't available or failed.
    try:
        from moviepy.audio.io.AudioFileClip import AudioFileClip
    except Exception:
        # moviepy not available; raise a clear error
        raise RuntimeError("Neither ffmpeg CLI nor moviepy are available to extract audio.")

    try:
        with AudioFileClip(str(media_path)) as audio:
            if audio is None:
                raise ValueError("No audio track found in media file.")
            # Ensure output directory exists
            out_dir = Path(output_audio_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            audio.write_audiofile(str(output_audio_path))
            try:
                print(f"[AUDIO] Extracted audio to: {output_audio_path}")
            except Exception:
                pass
    except Exception:
        # Re-raise so the caller (FastAPI) can return a 500 with the traceback
        raise
