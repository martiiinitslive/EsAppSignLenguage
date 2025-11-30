import sys
from pathlib import Path
import re
import importlib.util

# Ensure app-back/src is on path for imports
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / 'app-back'))

from src.components.format_text_for_renderer import normalize_text_for_renderer, save_original_for_subtitles


def _load_renderer_text_to_pose():
    # Load the renderer module by path (it's not a package import)
    mod_path = repo / 'app-back' / 'mp' / 'run_pose_to_video_mediapipe.py'
    spec = importlib.util.spec_from_file_location('rpp', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.text_to_pose_sequence


def test_normalized_text_works_with_renderer_sequence():
    src = "¡Hola, señor! ¿Cómo estás?  Esto  es   una prueba."
    out = normalize_text_for_renderer(src)

    # Output contains only allowed characters
    assert re.match(r'^[A-Za-z0-9 .,\-]+$', out)

    text_to_pose_sequence = _load_renderer_text_to_pose()
    seq = text_to_pose_sequence(out)

    # The renderer must see SPACE tokens for spaces, and PERIOD/COMMA for punctuation
    assert any(tok == 'SPACE' for tok in seq)
    # final character in source is a period, so last token must be PERIOD
    assert seq[-1] == 'PERIOD'


def test_save_original_for_subtitles_writes_file(tmp_path):
    txt = "Texto original: ¿Qué? ¡Hola!"
    out_path = save_original_for_subtitles(txt, out_dir=str(tmp_path), filename='orig.txt')
    p = Path(out_path)
    assert p.exists()
    assert p.read_text(encoding='utf-8') == txt
