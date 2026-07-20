"""
triage.py — cheap, deterministic pre-filters for bulk ingestion.

No LLM calls. check_structural_sanity runs entirely offline; check_duplicate
needs a Neo4j session but is exception-transparent — callers decide whether
a failure should fail loud (single upload) or best-effort-skip (bulk scan).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def check_structural_sanity(
    path: Path, *, supported_extensions: set[str], max_pdf_pages: int
) -> Optional[str]:
    """Return a rejection reason, or None if the file is structurally acceptable."""
    if not path.exists() or not path.is_file():
        return "file does not exist"

    ext = path.suffix.lower()
    if ext not in supported_extensions:
        return f"unsupported extension {ext!r}"

    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"unreadable: {exc}"
    if size == 0:
        return "zero-byte file"

    if ext == ".pdf":
        try:
            import fitz

            with fitz.open(str(path)) as doc:
                if doc.page_count == 0:
                    return "PDF has zero pages"
                if doc.page_count > max_pdf_pages:
                    return f"PDF has {doc.page_count} pages (exceeds cap of {max_pdf_pages})"
        except Exception as exc:
            return f"corrupt or unreadable PDF: {exc}"

    return None


def check_duplicate(path: Path, *, logical_id: str, exporter, driver) -> Optional[str]:
    """
    Return a skip reason if identical content is already the ACTIVE revision
    for logical_id, else None.

    Exception-transparent: raises on driver/session errors rather than
    swallowing them — the caller decides fail-loud vs. best-effort-skip.
    """
    from ..document.versioning import file_content_sha256

    content_hash = file_content_sha256(path)
    with driver.session() as session:
        if exporter.active_revision_has_hash(session, logical_id, content_hash):
            return "duplicate of already-ACTIVE revision"
    return None
