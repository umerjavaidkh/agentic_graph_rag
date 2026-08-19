"""Shared test bootstrap.

Every test file used to open with its own copy of this:

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

98 copies, each asserting "the repo root is exactly one level above me".
That is only true while every test sits directly in tests/, which is
precisely what stopped the suite from being organised into subdirectories:
nest a file one level and parents[1] silently becomes tests/ rather than the
repo root.

pytest imports this file before collecting anything beneath tests/, at any
depth, so one copy here replaces all of them and keeps working however the
directory is arranged later.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
