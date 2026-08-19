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


# ─────────────────────────────────────────────────────────────────────────
# Importing a module with stand-ins in place, without leaking them.
#
# Most of this suite replaces heavy dependencies -- neo4j, langgraph, the
# auth chain -- with MagicMocks before importing its subject. Done at module
# scope that is unsafe: pytest imports EVERY test module during collection,
# before running a single test, so a stub installed at import time is still
# in sys.modules when a later module is imported, and decides what that
# module sees. The suite consequently passes in alphabetical order and fails
# in reverse, which is also what stops tests/ being split into directories.
#
# teardown_module cannot fix this -- it runs after a module's tests, long
# after every module has been imported. The stubbing has to move inside a
# fixture, so it exists only while a test is actually running.
# ─────────────────────────────────────────────────────────────────────────
import importlib
import types
from contextlib import contextmanager

import pytest


@contextmanager
def _patched_modules(stubs: dict, drop: tuple = ()):
    """Install stand-ins, then restore sys.modules exactly as it was."""
    saved = {name: sys.modules[name] for name in set(stubs) | set(drop) if name in sys.modules}
    touched = set(stubs) | set(drop)
    for name in drop:
        sys.modules.pop(name, None)
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for attr, value in (attrs or {}).items():
            setattr(mod, attr, value)
        sys.modules[name] = mod
    try:
        yield
    finally:
        for name in touched:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture
def stubbed_import():
    """Import a module under stand-ins; drop it again afterwards.

    The subject is removed from sys.modules on the way in and out, so each
    test builds it against this file's stubs rather than against whatever a
    previously imported test module happened to leave behind.
    """
    imported: list = []

    def _import(target: str, stubs: dict | None = None, drop: tuple = ()):
        # Drop the target's ancestor packages too, unless this file stubs them
        # deliberately. Another test module may have replaced src.shared.auth
        # at import time with a stand-in that has no __path__, and importing
        # src.shared.auth.rbac_setup through that fails. Clearing the ancestors
        # makes a converted file immune to the ones not yet converted, so each
        # conversion helps on its own instead of only once all of them land.
        parts = target.split(".")
        ancestors = tuple(
            ".".join(parts[:i]) for i in range(1, len(parts))
            if ".".join(parts[:i]) not in (stubs or {})
        )
        with _patched_modules(stubs or {}, drop=tuple(drop) + ancestors + (target,)):
            module = importlib.import_module(target)
            imported.append(target)
            return module

    yield _import
    for name in imported:
        sys.modules.pop(name, None)
