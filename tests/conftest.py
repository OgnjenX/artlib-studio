"""Shared import paths for running tests from the repository checkout."""
from __future__ import annotations

import sys
from pathlib import Path


STUDIO_ROOT = Path(__file__).resolve().parents[1]
ARTLIB_ROOT = STUDIO_ROOT.parent / "AdaptiveResonanceLib"

for path in (STUDIO_ROOT, ARTLIB_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
