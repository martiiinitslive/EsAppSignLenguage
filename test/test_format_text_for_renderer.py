import sys
from pathlib import Path
import re

# Ensure app-back/src is on path for imports
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / 'app-back'))

from src.components.format_text_for_renderer import normalize_text_for_renderer


def test_remove_diacritics_and_preserve_text():
    src = "¡Hola, señor! ¿Cómo estás?  Esto  es   una prueba: áéíóú ÁÉÍÓÚ ñ Ñ ü Ü"
    out = normalize_text_for_renderer(src)

    # No Spanish opening punctuation or original accented letters should remain
    assert '¡' not in out
    assert '¿' not in out
    for ch in 'áéíóúÁÉÍÓÚñÑüÜ':
        assert ch not in out

    # The corresponding unaccented forms must appear
    assert 'senor' in out
    assert 'Como' in out or 'Como' in out

    # Exclamation and question marks are not in the allowed set -> replaced/removed
    assert '!' not in out
    assert '?' not in out

    # Comma and period should be preserved if present
    assert ',' in out

    # Multiple spaces should be collapsed into single spaces
    assert '  ' not in out

    # Output should only contain allowed characters (letters, digits, space, period, comma, hyphen)
    assert re.match(r'^[A-Za-z0-9 .,\-]+$', out)


def test_preserve_order_and_words():
    src = 'Árbol árbol, árbol. ¡árbol?'
    out = normalize_text_for_renderer(src)
    # Order and words preserved (diacritics removed)
    assert out.startswith('Arbol')
    assert 'arbol' in out
    assert ',' in out and '.' in out
