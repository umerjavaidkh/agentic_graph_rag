"""
ocr.py — pluggable OCR backend for low-confidence PDF pages.

Mirrors src/storage/blob/factory.py's shape: lazy settings import inside
the factory function, process-level singleton cache, backend selected by
a string setting (PDF_OCR_BACKEND).
"""
from __future__ import annotations

import io
from typing import Optional, Protocol


class OcrBackend(Protocol):
    def recognize(self, image_bytes: bytes, *, lang: str) -> str:
        """Return recognized text from a rasterized page image (PNG bytes).

        May raise (missing binary, decode failure, etc.) — callers must
        catch and degrade gracefully; this method does not swallow errors.
        """
        ...


class TesseractOcrBackend:
    """Wraps pytesseract. Constructor raises ImportError if the pip package
    isn't installed; callers (get_ocr_backend) catch that."""

    def __init__(self) -> None:
        import pytesseract
        from PIL import Image

        self._pytesseract = pytesseract
        self._Image = Image

    def recognize(self, image_bytes: bytes, *, lang: str) -> str:
        image = self._Image.open(io.BytesIO(image_bytes))
        return self._pytesseract.image_to_string(image, lang=lang)


_backend_singleton: Optional[OcrBackend] = None
_backend_resolved: bool = False


def get_ocr_backend() -> Optional[OcrBackend]:
    """
    Return the process-level OCR backend singleton, or None if disabled or
    unavailable.

    Never raises — construction failures (missing pytesseract package or
    missing tesseract binary, the latter surfaced at the first
    .recognize() call) are logged once and cached as "no backend".
    """
    global _backend_singleton, _backend_resolved
    if _backend_resolved:
        return _backend_singleton

    from ...shared.config.settings import PDF_OCR_BACKEND

    _backend_resolved = True
    if PDF_OCR_BACKEND == "tesseract":
        try:
            _backend_singleton = TesseractOcrBackend()
        except ImportError as exc:
            print(
                f"   ⚠ OCR backend 'tesseract' unavailable (pytesseract not installed): {exc}"
            )
            _backend_singleton = None
    else:
        _backend_singleton = None
    return _backend_singleton
