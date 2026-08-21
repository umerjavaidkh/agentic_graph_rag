"""The page renderer behind the chat viewer's "look at the actual page"."""
import pytest

from src.unstructured.document.page_image import (
    PageOutOfRange,
    PyMuPdfRenderer,
    UnsupportedPageFormat,
    can_render,
    register_renderer,
    render_page,
    renderer_for,
    supported_suffixes,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


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


def test_a_format_with_no_renderer_says_so_rather_than_guessing():
    assert not can_render(".docx")
    with pytest.raises(UnsupportedPageFormat) as exc:
        render_page(b"", ".docx", 1, 110)

    assert ".docx" in str(exc.value)
    assert ".pdf" in str(exc.value), "should name what it can render"


def test_suffix_matching_ignores_case():
    assert renderer_for(".PDF") is renderer_for(".pdf")


def test_a_new_format_is_added_by_registering_one(monkeypatch):
    """The seam that makes an Office backend a registration, not an edit.

    A LibreOffice-backed renderer is the only thing standing between this
    and .docx/.pptx pages; this proves it needs no change here to land.
    """
    import src.unstructured.document.page_image as module

    monkeypatch.setattr(module, "_RENDERERS", dict(module._RENDERERS))

    class FakeOfficeRenderer:
        suffixes = (".docx", ".pptx")

        def render(self, data, page_number, dpi):
            return _PNG_MAGIC + f"{page_number}@{dpi}".encode()

    register_renderer(FakeOfficeRenderer())

    assert can_render(".docx") and can_render(".pptx")
    assert render_page(b"", ".docx", 7, 120) == _PNG_MAGIC + b"7@120"
    assert ".pdf" in supported_suffixes(), "registering must not displace the built-in"


def test_the_pdf_renderer_declares_the_suffix_it_handles():
    assert PyMuPdfRenderer.suffixes == (".pdf",)
