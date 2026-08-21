"""The page renderer behind the chat viewer's "look at the actual page"."""
import subprocess
from pathlib import Path

import pytest

from src.unstructured.document.page_image import (
    PageOutOfRange,
    PyMuPdfRenderer,
    SofficeRenderer,
    UnsupportedPageFormat,
    can_render,
    register_renderer,
    registered_suffixes,
    render_page,
    renderer_for,
    supported_suffixes,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The module namespace `register_renderer` and friends actually use.
_MODULE = register_renderer.__globals__


@pytest.fixture(autouse=True)
def isolated_registry():
    """Leave the renderer registry exactly as it was found.

    Registration is a module-global side effect, so a test that adds a
    renderer changes what every later test sees -- including tests in other
    files that only ask "can this format be rendered?". Restoring the dict's
    *contents* rather than rebinding the name keeps this correct however the
    module was imported.
    """
    registry = _MODULE["_RENDERERS"]
    original = dict(registry)
    converted = _MODULE["_CONVERTED"]
    converted.clear()
    yield
    registry.clear()
    registry.update(original)
    converted.clear()


@pytest.fixture
def three_page_pdf() -> bytes:
    import fitz

    document = fitz.open()
    for n in (1, 2, 3):
        page = document.new_page()
        page.insert_text((72, 144), f"This is page {n}.", fontsize=24)
    return document.tobytes()


def test_a_page_renders_to_a_real_png(three_page_pdf):
    png = render_page(three_page_pdf, ".pdf", 2, 110)

    assert png.startswith(_PNG_MAGIC), "not a PNG"
    assert len(png) > 1000


def test_each_page_renders_to_something_different(three_page_pdf):
    """The point of a page image is that it shows *that* page.

    A renderer that quietly returns page 1 every time would satisfy every
    other assertion here.
    """
    pages = {n: render_page(three_page_pdf, ".pdf", n, 90) for n in (1, 2, 3)}

    assert len(set(pages.values())) == 3, "different pages rendered identically"


def test_dpi_changes_the_size(three_page_pdf):
    assert len(render_page(three_page_pdf, ".pdf", 1, 150)) > len(
        render_page(three_page_pdf, ".pdf", 1, 60)
    )


@pytest.mark.parametrize("page_number", [0, -1, 4, 999])
def test_a_page_outside_the_document_is_refused_not_clamped(three_page_pdf, page_number):
    """Clamping would render a page the caller did not ask for, and the
    viewer would show it as though it were the cited one."""
    with pytest.raises(PageOutOfRange) as exc:
        render_page(three_page_pdf, ".pdf", page_number, 110)

    assert "1-3" in str(exc.value)


def test_a_format_with_no_usable_renderer_says_so_rather_than_guessing(no_soffice):
    assert not can_render(".docx")
    with pytest.raises(UnsupportedPageFormat) as exc:
        render_page(b"", ".docx", 1, 110)

    assert ".docx" in str(exc.value)
    assert ".pdf" in str(exc.value), "should name what it can render"


def test_suffix_matching_ignores_case():
    assert renderer_for(".PDF") is renderer_for(".pdf")


def test_a_new_format_is_added_by_registering_one():
    """The seam that makes an Office backend a registration, not an edit.

    A LibreOffice-backed renderer is the only thing standing between this
    and .docx/.pptx pages; this proves it needs no change here to land.
    """
    class FakeOfficeRenderer:
        suffixes = (".docx", ".pptx")

        def available(self):
            return True

        def render(self, data, suffix, page_number, dpi):
            return _PNG_MAGIC + f"{suffix}:{page_number}@{dpi}".encode()

    register_renderer(FakeOfficeRenderer())

    assert can_render(".docx") and can_render(".pptx")
    assert render_page(b"", ".docx", 7, 120) == _PNG_MAGIC + b".docx:7@120"
    assert ".pdf" in supported_suffixes(), "registering must not displace the built-in"


def test_the_pdf_renderer_declares_the_suffix_it_handles():
    assert PyMuPdfRenderer.suffixes == (".pdf",)


@pytest.fixture
def no_soffice(monkeypatch):
    """A machine with no LibreOffice -- the default this must degrade to."""
    monkeypatch.setattr(SofficeRenderer, "binary", staticmethod(lambda: None))
    return None


@pytest.fixture
def with_soffice(monkeypatch, tmp_path):
    """A machine where LibreOffice exists, without running LibreOffice."""
    monkeypatch.setattr(SofficeRenderer, "binary", staticmethod(lambda: "/usr/bin/soffice"))
    return None


def test_the_office_backend_is_registered_even_when_it_cannot_run(no_soffice):
    """Registered-but-unavailable is the whole point of an optional backend.

    Being listed is what lets the API say "install LibreOffice and this
    starts working" instead of "unsupported format".
    """
    assert ".docx" in registered_suffixes()
    assert ".docx" not in supported_suffixes()
    assert renderer_for(".docx") is None


def test_installing_libreoffice_is_the_only_step(with_soffice, monkeypatch, three_page_pdf):
    """No code change, no configuration, no restart-time registration."""
    def fake_run(cmd, **kwargs):
        # LibreOffice writes <stem>.pdf into --outdir; stand in for that.
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "page_source.pdf").write_bytes(three_page_pdf)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(_MODULE["subprocess"], "run", fake_run)

    assert can_render(".docx")
    assert ".docx" in supported_suffixes()
    png = render_page(b"fake docx bytes", ".docx", 2, 90)
    assert png.startswith(_PNG_MAGIC)


def test_a_document_is_converted_once_not_once_per_page(with_soffice, monkeypatch, three_page_pdf):
    """Turning pages would otherwise pay a multi-second conversion each time."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "page_source.pdf").write_bytes(three_page_pdf)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(_MODULE["subprocess"], "run", fake_run)

    data = b"one document, read page by page"
    for page in (1, 2, 3):
        render_page(data, ".pptx", page, 90)

    assert len(calls) == 1, f"converted {len(calls)} times for 3 pages"


def test_a_conversion_that_produces_nothing_reports_why(with_soffice, monkeypatch):
    """A silent failure here would surface as a broken image in the panel."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, b"", b"source file could not be loaded")

    monkeypatch.setattr(_MODULE["subprocess"], "run", fake_run)

    with pytest.raises(UnsupportedPageFormat) as exc:
        render_page(b"corrupt", ".docx", 1, 110)

    assert "could not be loaded" in str(exc.value)


def test_the_office_renderer_runs_in_its_own_profile(with_soffice, monkeypatch, three_page_pdf):
    """LibreOffice's default profile is shared under $HOME; two conversions
    racing on it is a known way to make it hang."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["timeout"] = kwargs.get("timeout")
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "page_source.pdf").write_bytes(three_page_pdf)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(_MODULE["subprocess"], "run", fake_run)
    render_page(b"x", ".docx", 1, 90)

    assert any(a.startswith("-env:UserInstallation=") for a in seen["cmd"])
    assert "--headless" in seen["cmd"]
    assert seen["timeout"], "an unbounded conversion can hang the request thread"
