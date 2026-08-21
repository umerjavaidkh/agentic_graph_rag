"""Render one page of a source document as an image.

The chat viewer embeds the original file in an iframe, which only works for
PDFs: a browser has no renderer for .docx/.pptx/.xlsx, so those arrive as a
download instead of a view. A picture of the page is something every browser
can show, and it pins the view to the cited page rather than trusting a PDF
plugin to honour a `#page` fragment.

Renderers are registered by file suffix, the same shape as the parser and
retrieval-strategy registries, because the formats split on what they need:
a PDF already carries its own layout and PyMuPDF draws it directly, while a
Word or PowerPoint file stores no layout at all -- deciding where its content
falls on a page is what a word processor *does*, so rendering one means
running a layout engine (LibreOffice), not parsing harder.

That backend is therefore a separate registration rather than a branch here.
Adding it is `register_renderer(SofficeRenderer())` plus the package in the
image; nothing in this module or its callers changes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


class UnsupportedPageFormat(RuntimeError):
    """No registered renderer can draw a page of this format."""


@runtime_checkable
class PageRenderer(Protocol):
    """Draws one page of a document to image bytes."""

    #: Lowercased file suffixes this renderer handles, e.g. `(".pdf",)`.
    suffixes: tuple[str, ...]

    def render(self, data: bytes, page_number: int, dpi: int) -> bytes:
        """Return PNG bytes for `page_number` (1-indexed) of `data`.

        Raises `PageOutOfRange` if the page does not exist.
        """


class PageOutOfRange(ValueError):
    """The requested page number is not in the document."""

    def __init__(self, page_number: int, page_count: int):
        self.page_number, self.page_count = page_number, page_count
        super().__init__(f"Page {page_number} is outside this document (1-{page_count}).")


class PyMuPdfRenderer:
    """PDF pages, drawn by the library that already parses them.

    PyMuPDF is a dependency either way, so this costs no image weight -- it
    is the same `get_pixmap` call the vision enrichment makes.
    """

    suffixes = (".pdf",)

    def render(self, data: bytes, page_number: int, dpi: int) -> bytes:
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as document:
            if not 1 <= page_number <= document.page_count:
                raise PageOutOfRange(page_number, document.page_count)
            return document[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")


_RENDERERS: dict[str, PageRenderer] = {}


def register_renderer(renderer: PageRenderer) -> None:
    for suffix in renderer.suffixes:
        _RENDERERS[suffix.lower()] = renderer


def renderer_for(suffix: str) -> PageRenderer | None:
    return _RENDERERS.get((suffix or "").lower())


def can_render(suffix: str) -> bool:
    return renderer_for(suffix) is not None


def supported_suffixes() -> tuple[str, ...]:
    return tuple(sorted(_RENDERERS))


def render_page(data: bytes, suffix: str, page_number: int, dpi: int) -> bytes:
    """PNG bytes for one page, or `UnsupportedPageFormat` if nothing can draw it."""
    renderer = renderer_for(suffix)
    if renderer is None:
        raise UnsupportedPageFormat(
            f"No renderer for {suffix or 'this format'}; "
            f"registered: {', '.join(supported_suffixes()) or 'none'}."
        )
    return renderer.render(data, page_number, dpi)


register_renderer(PyMuPdfRenderer())
