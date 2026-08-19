"""
tests/test_document_parser_ocr_unit.py — OCR fallback wiring in LightPdfParser.

Run with:
    python -m pytest tests/test_document_parser_ocr_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest


import src.unstructured.document.light.parser as parser_mod
from src.unstructured.document.light.parser import LightPdfParser


def _blank_page():
    """A real PyMuPDF page with no text (simulates a scanned page)."""
    doc = fitz.open()
    doc.new_page()
    page = doc[0]
    return doc, page


class _FakeOcrBackend:
    def __init__(self, text: str = "", raises: Exception | None = None):
        self._text = text
        self._raises = raises
        self.calls: list[dict] = []

    def recognize(self, image_bytes: bytes, *, lang: str) -> str:
        self.calls.append({"lang": lang, "bytes_len": len(image_bytes)})
        if self._raises:
            raise self._raises
        return self._text


def test_try_ocr_returns_none_none_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: None)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block, error = parser._try_ocr(page, 1)
        assert block is None
        assert error is None
    finally:
        doc.close()


def test_try_ocr_returns_block_on_success(monkeypatch):
    fake_backend = _FakeOcrBackend(text="Recognized page text")
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: fake_backend)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block, error = parser._try_ocr(page, 1)
        assert error is None
        assert block is not None
        assert block.text == "Recognized page text"
        assert block.source == "ocr"
        assert fake_backend.calls[0]["lang"] == "eng"
    finally:
        doc.close()


def test_try_ocr_returns_error_on_exception(monkeypatch):
    fake_backend = _FakeOcrBackend(raises=RuntimeError("tesseract binary missing"))
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: fake_backend)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block, error = parser._try_ocr(page, 1)
        assert block is None
        assert error == "tesseract binary missing"
    finally:
        doc.close()


def test_try_ocr_returns_none_none_on_empty_ocr_text(monkeypatch):
    fake_backend = _FakeOcrBackend(text="   ")
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: fake_backend)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block, error = parser._try_ocr(page, 1)
        assert block is None
        assert error is None
    finally:
        doc.close()


def test_low_confidence_marker_reports_ocr_error():
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block = parser._low_confidence_marker(page, 1, False, "tesseract binary missing")
        assert "OCR was attempted" in block.text
        assert "tesseract binary missing" in block.text
    finally:
        doc.close()


def test_low_confidence_marker_reports_unavailable_backend(monkeypatch):
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_OCR", True)
    monkeypatch.setattr(parser_mod, "PDF_OCR_BACKEND", "tesseract")
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: None)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block = parser._low_confidence_marker(page, 1, False, None)
        assert "unavailable in this environment" in block.text
    finally:
        doc.close()


def test_low_confidence_marker_unchanged_when_ocr_disabled(monkeypatch):
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_OCR", False)
    doc, page = _blank_page()
    try:
        parser = LightPdfParser()
        block_no_text = parser._low_confidence_marker(page, 1, False, None)
        block_has_text = parser._low_confidence_marker(page, 1, True, None)
        assert "No reliable text extracted" in block_no_text.text
        assert "may be incomplete" in block_has_text.text
    finally:
        doc.close()


def test_extract_pages_uses_ocr_when_enabled_and_low_confidence(monkeypatch):
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_OCR", True)
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_PDFPLUMBER", False)
    fake_backend = _FakeOcrBackend(text="A" * 500)  # well above PDF_LOW_TEXT_CHARS
    monkeypatch.setattr(parser_mod, "get_ocr_backend", lambda: fake_backend)

    doc = fitz.open()
    doc.new_page()  # blank page -> near-zero native text -> low confidence
    try:
        parser = LightPdfParser()
        extracts = parser._extract_pages(Path("fake.pdf"), doc)
        assert len(extracts) == 1
        assert "A" * 500 in extracts[0].text
        assert extracts[0].low_confidence is False
    finally:
        doc.close()


def test_extract_pages_falls_back_to_marker_when_ocr_disabled(monkeypatch):
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_OCR", False)
    monkeypatch.setattr(parser_mod, "PDF_ENABLE_PDFPLUMBER", False)

    doc = fitz.open()
    doc.new_page()
    try:
        parser = LightPdfParser()
        extracts = parser._extract_pages(Path("fake.pdf"), doc)
        assert len(extracts) == 1
        assert extracts[0].low_confidence is True
        # The marker note is appended to blocks (not rejoined into .text
        # unless OCR succeeded — see the "uses_ocr" test above).
        assert any("Low confidence extract" in b.text for b in extracts[0].blocks)
    finally:
        doc.close()
