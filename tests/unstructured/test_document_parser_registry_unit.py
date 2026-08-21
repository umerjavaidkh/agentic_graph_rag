"""
tests/test_document_parser_registry_unit.py — DocumentParser registry dispatch.

Run with:
    python -m pytest tests/test_document_parser_registry_unit.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def parser_registry():
    """
    Fresh, un-stubbed src.unstructured.document.parser_registry.

    Other test modules in this suite (e.g. test_scalable_pipeline_unit.py)
    replace src.unstructured.document* with MagicMocks at *import* time, and pytest
    imports every test module during collection before running any test —
    so if this module were imported at collection time, a later-collected
    file's stubbing would overwrite attributes on the same shared module
    object. Re-importing fresh inside a fixture (i.e. at test-execution
    time, after all collection has finished) sidesteps that.
    """
    for name in list(sys.modules):
        if name == "src.unstructured.document" or name.startswith("src.unstructured.document."):
            del sys.modules[name]
    module = importlib.import_module("src.unstructured.document.parser_registry")
    return module


@pytest.fixture()
def light_pdf_parser_cls(parser_registry):
    from src.unstructured.document.light.parser import LightPdfParser

    return LightPdfParser


@pytest.fixture()
def rtldoc_pdf_parser_cls(parser_registry):
    from src.unstructured.document.rtldoc_backend.parser import RtldocPdfParser

    return RtldocPdfParser


def test_pdf_extension_resolves_to_rtldoc_pdf_parser_by_default(parser_registry, rtldoc_pdf_parser_cls):
    parser = parser_registry.get_parser("report.pdf")
    assert isinstance(parser, rtldoc_pdf_parser_cls)


def test_explicit_backend_qualifier_resolves(parser_registry, light_pdf_parser_cls):
    parser = parser_registry.get_parser("report.pdf", backend="light")
    assert isinstance(parser, light_pdf_parser_cls)


def test_explicit_rtldoc_backend_qualifier_resolves(parser_registry, rtldoc_pdf_parser_cls):
    parser = parser_registry.get_parser("report.pdf", backend="rtldoc")
    assert isinstance(parser, rtldoc_pdf_parser_cls)


def test_unknown_extension_raises_value_error(parser_registry):
    with pytest.raises(ValueError):
        parser_registry.get_parser("report.rtf")


def test_unregistered_backend_falls_back_to_bare_extension(parser_registry, rtldoc_pdf_parser_cls):
    # No ".pdf:other" registered — falls back to the bare ".pdf" entry
    # (rtldoc, the default backend as of this parser's introduction).
    parser = parser_registry.get_parser("report.pdf", backend="other")
    assert isinstance(parser, rtldoc_pdf_parser_cls)


def test_supported_extensions_includes_pdf(parser_registry):
    assert ".pdf" in parser_registry.supported_extensions()


def test_register_parser_adds_new_entry(parser_registry):
    calls = []

    class _FakeParser:
        def parse(self, source):
            calls.append(source)
            return [], []

    parser_registry.register_parser(".fake", _FakeParser)
    try:
        parser = parser_registry.get_parser("thing.fake")
        assert isinstance(parser, _FakeParser)
        assert ".fake" in parser_registry.supported_extensions()
    finally:
        del parser_registry._PARSER_REGISTRY[".fake"]
