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

import functools
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable


class UnsupportedPageFormat(RuntimeError):
    """No registered renderer can draw a page of this format."""


@runtime_checkable
class PageRenderer(Protocol):
    """Draws one page of a document to image bytes."""

    #: Lowercased file suffixes this renderer handles, e.g. `(".pdf",)`.
    suffixes: tuple[str, ...]

    def render(self, data: bytes, suffix: str, page_number: int, dpi: int) -> bytes:
        """Return PNG bytes for `page_number` (1-indexed) of `data`.

        `suffix` is passed through because a converter dispatches on it --
        LibreOffice decides how to read a file from its extension.

        Raises `PageOutOfRange` if the page does not exist.
        """

    def available(self) -> bool:
        """Whether this renderer can run right now.

        Separate from being registered: an optional backend is registered
        unconditionally but only usable once its binary is installed, and a
        caller needs to know which before offering the user a page image.
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

    def available(self) -> bool:
        return True

    def render(self, data: bytes, suffix: str = ".pdf", page_number: int = 1, dpi: int = 110) -> bytes:
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
    """The renderer for `suffix`, or None if none is registered *or usable*.

    An optional backend registers whether or not its binary is installed --
    that is what makes it optional rather than conditional -- so being
    registered is not the same as being able to run.
    """
    renderer = _RENDERERS.get((suffix or "").lower())
    if renderer is None or not renderer.available():
        return None
    return renderer


def can_render(suffix: str) -> bool:
    return renderer_for(suffix) is not None


def supported_suffixes() -> tuple[str, ...]:
    """Suffixes that can actually be rendered on this machine, right now."""
    return tuple(sorted(s for s, r in _RENDERERS.items() if r.available()))


def registered_suffixes() -> tuple[str, ...]:
    """Every suffix some renderer claims -- installed or not.

    The difference between this and `supported_suffixes` is exactly the set
    of formats that would start working if the optional backend were
    installed, which is what the API tells the user when it declines.
    """
    return tuple(sorted(_RENDERERS))


def render_page(data: bytes, suffix: str, page_number: int, dpi: int) -> bytes:
    """PNG bytes for one page, or `UnsupportedPageFormat` if nothing can draw it."""
    renderer = renderer_for(suffix)
    if renderer is None:
        raise UnsupportedPageFormat(
            f"No renderer for {suffix or 'this format'}; "
            f"available here: {', '.join(supported_suffixes()) or 'none'}."
        )
    return renderer.render(data, suffix, page_number, dpi)


# Converting a document costs seconds; a reader turning pages would pay it
# per page. Keyed by content hash, and deliberately tiny -- this holds whole
# PDFs, and the case worth serving is one reader moving through one document.
_CONVERTED: "OrderedDict[str, bytes]" = OrderedDict()
_CONVERTED_MAX = 2


class SofficeRenderer:
    """Office pages, by way of LibreOffice.

    Optional on purpose. Word, PowerPoint and Excel files store no layout --
    where content falls on a page is decided when the document is opened --
    so drawing one means running a word processor, and the only real option
    weighs roughly 500MB. That is a lot to impose on someone who only ever
    ingests PDFs.

    So it registers unconditionally and reports itself unavailable until the
    binary exists. Install LibreOffice and Office pages start rendering with
    no code change and no configuration; skip it and those documents fall
    back to their text and a download link. The probe is cached because
    `which` on every page view is a syscall for an answer that does not
    change within a process.
    """

    suffixes = (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                ".odt", ".odp", ".ods", ".rtf")

    #: Generous: a large deck genuinely takes a while, and the alternative to
    #: waiting is a failed render.
    timeout_seconds = int(os.environ.get("SOFFICE_TIMEOUT_SECONDS", "120"))

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def binary() -> str | None:
        return shutil.which("soffice") or shutil.which("libreoffice")

    def available(self) -> bool:
        return self.binary() is not None

    def render(self, data: bytes, suffix: str, page_number: int, dpi: int) -> bytes:
        return PyMuPdfRenderer().render(
            self._as_pdf(data, suffix), ".pdf", page_number, dpi
        )

    def _as_pdf(self, data: bytes, suffix: str) -> bytes:
        key = hashlib.sha256(data).hexdigest()
        if key in _CONVERTED:
            _CONVERTED.move_to_end(key)
            return _CONVERTED[key]

        binary = self.binary()
        if binary is None:
            raise UnsupportedPageFormat(
                "LibreOffice is not installed, so Office pages cannot be rendered."
            )

        with tempfile.TemporaryDirectory() as workdir:
            source = Path(workdir) / f"page_source{suffix}"
            source.write_bytes(data)
            result = subprocess.run(
                [
                    binary,
                    # Its own throwaway profile: the default is a shared
                    # directory under $HOME, and two conversions racing on it
                    # is a known way to make LibreOffice hang.
                    f"-env:UserInstallation=file://{workdir}/profile",
                    "--headless", "--norestore", "--nolockcheck",
                    "--convert-to", "pdf", "--outdir", workdir, str(source),
                ],
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            converted = Path(workdir) / "page_source.pdf"
            if not converted.exists():
                raise UnsupportedPageFormat(
                    "LibreOffice could not convert this document: "
                    + (result.stderr.decode("utf-8", "replace").strip()[:300]
                       or f"exit {result.returncode}")
                )
            pdf = converted.read_bytes()

        _CONVERTED[key] = pdf
        while len(_CONVERTED) > _CONVERTED_MAX:
            _CONVERTED.popitem(last=False)
        return pdf


register_renderer(PyMuPdfRenderer())
register_renderer(SofficeRenderer())
