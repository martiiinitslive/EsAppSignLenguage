"""
Text normalization helper for the renderer.

Functions:
  normalize_text_for_renderer(text) -> cleaned string suitable for run_pose_to_video_mediapipe.text_to_pose_sequence

Behavior:
  - Normalize unicode (NFKD), remove diacritics
  - Convert common punctuation to ASCII equivalents
  - Keep letters, digits, spaces, periods and commas
  - Collapse multiple spaces
  - Trim
"""
import unicodedata
import re
from pathlib import Path
import time
import os

def normalize_text_for_renderer(text: str) -> str:
    if text is None:
        return ""
    # Normalize and remove diacritics
    s = unicodedata.normalize('NFKD', str(text))
    s = ''.join(c for c in s if not unicodedata.combining(c))

    # Replace Spanish opening punctuation and unusual dashes
    s = s.replace('¿', '').replace('¡', '')
    s = s.replace('—', '-').replace('–', '-')
    # Normalize ellipsis and typographic quotes
    s = s.replace('…', '...')
    s = s.replace('“', '"').replace('”', '"')
    s = s.replace("‘", "'").replace("’", "'")

    # Keep only letters, digits, space, period, comma and hyphen
    # Convert non-supported punctuation to spaces (so tokens separate)
    s = re.sub(r"[^A-Za-z0-9 .,\-]", ' ', s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", ' ', s)
    s = s.strip()
    return s


def save_original_for_subtitles(text: str, out_dir: str = None, filename: str = None) -> str:
    """Save the original (raw) text to a UTF-8 file for subtitle authoring.

    - `out_dir` defaults to `app-back/utils/output_utils/` if available, otherwise
      to a local `output_utils` folder next to `app-back`.
    - `filename` defaults to a timestamped `original_text_<ts>.txt`.

    Returns the absolute path to the written file as a string.
    """
    try:
        base = Path(__file__).resolve().parents[2]  # app-back
    except Exception:
        base = Path.cwd()

    if out_dir:
        outpath = Path(out_dir)
    else:
        outpath = base / "utils" / "output_utils"

    outpath.mkdir(parents=True, exist_ok=True)

    if not filename:
        ts = int(time.time())
        filename = f"original_text_{ts}.txt"

    fp = outpath / filename
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text or "")

    return str(fp.resolve())
