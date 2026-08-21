"""Office documents as DocumentIR — Word, PowerPoint, Excel.

The pipeline accepted `.pdf` and nothing else, which is the wrong shape for a
real corpus: a company drive is mostly Word and PowerPoint, and a PDF export
of either loses the structure this module gets for free.

That structure is the reason these are worth having as first-class parsers
rather than converting to PDF first. A PDF forces every backend to INFER
headings from font size and boldness -- the heuristic that currently produces
"Preamble", "Smith Street" and "Dry" as section titles on a 10-K. Word states
its headings outright in the paragraph style, PowerPoint has a title
placeholder per slide, and Excel has sheet names. Nothing is guessed.

Each library is imported inside the parser that needs it, so a deployment
handling only PDFs does not carry three Office readers it never calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from ...models import DKGEdge, DKGNode
from ..ir import Block, DocumentIR, PageBlock

# A heading is announced rather than inferred, so Axis-1 is told directly.
_HEADING = {"heading_hint": "heading"}


def _build(ir: DocumentIR) -> tuple[list[DKGNode], list[DKGEdge]]:
    """Shared phase 2. Imported here so the construction service is not a
    module-level dependency of a parser that may only ever be asked for IR."""
    from ...graph.construction_service import GraphConstructionService

    nodes, edges, _chunks = GraphConstructionService().build_structure(ir)
    return nodes, edges


def _page(number: int, blocks: list[Block]) -> PageBlock:
    text = "\n".join(b.text for b in blocks if b.text.strip())
    return PageBlock(
        page=number,
        text=text,
        blocks=blocks,
        regions=[b for b in blocks if b.kind in ("table", "figure")],
        confidence=1.0,        # extracted from markup, not recovered from a render
    )


def _toc_from(blocks_by_page: list[tuple[int, list[Block]]]) -> list[tuple[int, str, int]] | None:
    """The document's own outline, from headings it declared.

    Axis-1 prefers a real outline over its heading heuristics, and these
    formats can supply one truthfully. Gated at five entries with at least one
    top level, matching what the PDF path treats as a usable outline -- fewer
    than that is not a structure worth trusting.
    """
    toc = [
        (int(b.extra.get("heading_level", 1)), b.text.strip(), page)
        for page, blocks in blocks_by_page
        for b in blocks
        if b.extra.get("heading_hint") == "heading" and b.text.strip()
    ]
    if len(toc) < 5 or not any(level <= 1 for level, _, _ in toc):
        return None
    return toc


class DocxParser:
    """Word documents.

    Word has no stable page concept -- pagination is decided by the renderer,
    not stored -- so an explicit page break is the only honest divider. A
    document with none becomes a single page, which is accurate rather than a
    guess at where pages would fall.
    """

    def parse_ir(self, source: str | Path) -> DocumentIR:
        import docx  # python-docx

        path = Path(source)
        document = docx.Document(str(path))
        pages: list[PageBlock] = []
        current: list[Block] = []
        page_no = 1

        for para in document.paragraphs:
            text = (para.text or "").strip()
            style = (getattr(para.style, "name", "") or "")
            if "\f" in (para.text or "") and current:      # explicit page break
                pages.append(_page(page_no, current))
                page_no, current = page_no + 1, []
            if not text:
                continue
            extra: dict[str, Any] = {}
            if style.lower().startswith("heading"):
                extra = dict(_HEADING)
                tail = style.split()[-1]
                extra["heading_level"] = int(tail) if tail.isdigit() else 1
            elif style.lower() in ("title", "subtitle"):
                extra = dict(_HEADING, heading_level=1)
            current.append(Block(text=text, page=page_no, source="docx", extra=extra))

        for table in document.tables:
            rows = ["\t".join(c.text.strip() for c in r.cells) for r in table.rows]
            body = "\n".join(r for r in rows if r.strip())
            if body:
                current.append(Block(text=body, page=page_no, source="docx", kind="table"))

        pages.append(_page(page_no, current))
        ir = DocumentIR(
            source_name=path.stem,
            page_count=len(pages),
            pages=pages,
            toc=_toc_from([(p.page, p.blocks) for p in pages]),
        )
        return ir.finalize()

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        return _build(self.parse_ir(source))


class PptxParser:
    """Presentations. A slide IS a page, so the mapping needs no invention,
    and the title placeholder names the section without inference."""

    def parse_ir(self, source: str | Path) -> DocumentIR:
        from pptx import Presentation

        path = Path(source)
        deck = Presentation(str(path))
        pages: list[PageBlock] = []

        for index, slide in enumerate(deck.slides, start=1):
            blocks: list[Block] = []
            title_shape = getattr(slide.shapes, "title", None)
            title_text = (getattr(title_shape, "text", "") or "").strip()
            if title_text:
                blocks.append(Block(text=title_text, page=index, source="pptx",
                                    extra=dict(_HEADING, heading_level=1)))
            for shape in slide.shapes:
                if shape is title_shape:
                    continue
                if getattr(shape, "has_table", False):
                    rows = ["\t".join(c.text.strip() for c in r.cells) for r in shape.table.rows]
                    body = "\n".join(r for r in rows if r.strip())
                    if body:
                        blocks.append(Block(text=body, page=index, source="pptx", kind="table"))
                    continue
                text = (getattr(shape, "text", "") or "").strip()
                if text:
                    blocks.append(Block(text=text, page=index, source="pptx"))
            # Speaker notes are content the slide does not show, and are often
            # where the actual argument lives.
            notes = getattr(slide, "notes_slide", None) if slide.has_notes_slide else None
            note_text = (getattr(getattr(notes, "notes_text_frame", None), "text", "") or "").strip()
            if note_text:
                blocks.append(Block(text=note_text, page=index, source="pptx",
                                    extra={"speaker_notes": True}))
            pages.append(_page(index, blocks))

        ir = DocumentIR(
            source_name=path.stem,
            page_count=len(pages),
            pages=pages,
            toc=_toc_from([(p.page, p.blocks) for p in pages]),
        )
        return ir.finalize()

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        return _build(self.parse_ir(source))


class XlsxParser:
    """Workbooks as documents -- one page per sheet, the sheet name its heading.

    This is the DOCUMENT reader, not the tabular loader. A workbook whose
    sheets are really tables belongs in structured/ingestion/tabular.py, which
    turns them into a property graph with keys and relationships. This path is
    for a workbook that is prose in a grid: a checklist, a policy matrix, a
    register -- text to search, not rows to join.
    """

    def parse_ir(self, source: str | Path) -> DocumentIR:
        from openpyxl import load_workbook

        path = Path(source)
        # read_only streams rather than building the whole workbook in memory;
        # data_only takes cached formula RESULTS, since a formula string is not
        # what a reader is searching for.
        book = load_workbook(str(path), read_only=True, data_only=True)
        pages: list[PageBlock] = []

        for index, sheet in enumerate(book.worksheets, start=1):
            blocks = [Block(text=sheet.title, page=index, source="xlsx",
                            extra=dict(_HEADING, heading_level=1))]
            rows: list[str] = []
            for values in sheet.iter_rows(values_only=True):
                line = "\t".join("" if v is None else str(v) for v in values).rstrip("\t")
                if line.strip():
                    rows.append(line)
            if rows:
                blocks.append(Block(text="\n".join(rows), page=index,
                                    source="xlsx", kind="table"))
            pages.append(_page(index, blocks))

        book.close()
        ir = DocumentIR(
            source_name=path.stem,
            page_count=len(pages),
            pages=pages,
            toc=_toc_from([(p.page, p.blocks) for p in pages]),
        )
        return ir.finalize()

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        return _build(self.parse_ir(source))
