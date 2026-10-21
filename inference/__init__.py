"""
Inference package initialization for Digital Heritage Restoration.
"""

import os
import importlib.util

from .ocr_engine import extract_text_from_heritage
from .translation import translate_heritage_text, VERIFIED_TRANSLATIONS

# Load HeritageRestorer from the root inference.py (which is shadowed by this package).
# We use importlib to load it directly by file path.
_root_inference_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
if os.path.exists(_root_inference_py):
    _spec = importlib.util.spec_from_file_location("_root_inference", _root_inference_py)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    HeritageRestorer = _mod.HeritageRestorer
    restore_image = _mod.restore_image
    get_restorer = _mod.get_restorer
else:
    HeritageRestorer = None
    restore_image = None
    get_restorer = None

__all__ = [
    "extract_text_from_heritage",
    "translate_heritage_text",
    "VERIFIED_TRANSLATIONS",
    "HeritageRestorer",
    "restore_image",
    "get_restorer",
]
