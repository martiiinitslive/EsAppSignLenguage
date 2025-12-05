from pathlib import Path
import csv
from typing import Dict, Any, Iterable


# Single source of truth for the CSV header used by the feasibility study.
# 20 columns in fixed order — do not change the order once data collection starts.
CSV_COLUMNS: Iterable[str] = [
    "timestamp_utc",
    "request_id",
    "endpoint",
    "input_type",
    "input_length_seconds",
    "text_length_chars",
    "n_letters_rendered",
    "success",
    "error_stage",
    "error_message_short",
    "output_video_filename",
    "output_video_duration_seconds",
    "n_poses_rendered",
    "t_total",
    "t_download",
    "t_extract_audio",
    "t_asr",
    "t_text_normalisation",
    "t_pose_sequence",
    "t_render",
]


_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "runtime_metrics.csv"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _clean_value(val: Any) -> Any:
    if val is None:
        return ""
    # Keep booleans as strings so CSV consumers don't misinterpret types
    if isinstance(val, bool):
        return str(val)
    # Numbers and other primitives are fine
    if isinstance(val, (int, float)):
        return val
    s = str(val)
    s = s.replace("\n", " ").replace("\r", " ")
    # Truncate long error messages to keep CSV cells manageable
    if len(s) > 1000:
        s = s[:997] + "..."
    return s


def log_metrics(record: Dict[str, Any]) -> None:
    """Append a single row to `runtime_metrics.csv` using the fixed column order.

    The CSV is created in `app-back/logs/runtime_metrics.csv`. This function is
    tolerant of missing keys in `record` and will write empty cells for them.
    """
    _ensure_log_dir()
    write_header = not _LOG_FILE.exists()

    row = [_clean_value(record.get(k, None)) for k in CSV_COLUMNS]

    with _LOG_FILE.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(list(CSV_COLUMNS))
        writer.writerow(row)
