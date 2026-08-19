"""
tests/test_document_ocr_unit.py — pluggable OCR backend factory.

Run with:
    python -m pytest tests/test_document_ocr_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


import src.unstructured.document.ocr as ocr_mod
from src.unstructured.document.ocr import TesseractOcrBackend, get_ocr_backend


@pytest.fixture(autouse=True)
def _reset_singleton():
    ocr_mod._backend_singleton = None
    ocr_mod._backend_resolved = False
    yield
    ocr_mod._backend_singleton = None
    ocr_mod._backend_resolved = False


def test_factory_returns_none_when_backend_is_none(monkeypatch):
    import src.shared.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "PDF_OCR_BACKEND", "none")
    assert get_ocr_backend() is None


def test_factory_returns_none_when_pytesseract_unavailable(monkeypatch):
    import src.shared.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "PDF_OCR_BACKEND", "tesseract")
    monkeypatch.setattr(
        ocr_mod,
        "TesseractOcrBackend",
        MagicMock(side_effect=ImportError("no pytesseract")),
    )

    backend = get_ocr_backend()

    assert backend is None


def test_factory_caches_singleton(monkeypatch):
    import src.shared.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "PDF_OCR_BACKEND", "none")
    first = get_ocr_backend()
    second = get_ocr_backend()
    assert first is second is None
    assert ocr_mod._backend_resolved is True


def test_factory_builds_tesseract_backend_when_available(monkeypatch):
    import src.shared.config.settings as settings_mod

    fake_backend = MagicMock()
    monkeypatch.setattr(settings_mod, "PDF_OCR_BACKEND", "tesseract")
    monkeypatch.setattr(ocr_mod, "TesseractOcrBackend", MagicMock(return_value=fake_backend))

    backend = get_ocr_backend()

    assert backend is fake_backend
    assert get_ocr_backend() is fake_backend  # cached, constructor not called twice


def test_tesseract_backend_recognize_calls_image_to_string(monkeypatch):
    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.image_to_string = MagicMock(return_value="recognized text")
    fake_pil_image_mod = types.ModuleType("PIL.Image")
    fake_image = MagicMock()
    fake_pil_image_mod.open = MagicMock(return_value=fake_image)
    # Fresh fake "PIL" package (never mutate the real, already-imported PIL
    # module in place — monkeypatch.setitem swaps the sys.modules entry and
    # restores the original afterward, without touching any real object).
    fake_pil_pkg = types.ModuleType("PIL")
    fake_pil_pkg.Image = fake_pil_image_mod

    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image_mod)

    backend = TesseractOcrBackend()
    result = backend.recognize(b"fake-png-bytes", lang="eng")

    assert result == "recognized text"
    fake_pytesseract.image_to_string.assert_called_once_with(fake_image, lang="eng")
