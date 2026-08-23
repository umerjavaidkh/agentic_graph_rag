"""Lone UTF-16 surrogates in extracted text must never reach a UTF-8 write.

Regression: a 13-page arXiv paper carried 219 lone `\ud835` (the high half
of a Mathematical Alphanumeric character, low half lost during extraction).
The parse and every enrichment step succeeded; the Neo4j write then failed
with "surrogates not allowed" and the document was dropped after all that
work. Two of 355 documents in one ingestion run were lost this way.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared.text_sanitize import sanitize_text
from src.unstructured.document.ir import Block, PageBlock
from src.unstructured.models import DKGNode


LONE_HIGH = "\ud835"
REAL_MATH_X = "\U0001D465"  # the character a valid surrogate pair encodes


def test_lone_surrogate_is_dropped():
    assert sanitize_text(f"a{LONE_HIGH}b") == "ab"


def test_valid_pair_is_recombined_not_discarded():
    # The pair is a real character that merely survived extraction in UTF-16
    # form -- dropping it would silently delete math variables from the text
    # a reader searches.
    pair = REAL_MATH_X.encode("utf-16-le").decode("utf-16-le")
    assert sanitize_text(f"a{pair}b") == f"a{REAL_MATH_X}b"


def test_clean_text_is_returned_unchanged():
    clean = "ordinary text, no surrogates"
    assert sanitize_text(clean) is clean


def test_empty_and_falsy_are_safe():
    assert sanitize_text("") == ""


def test_output_is_always_utf8_encodable():
    for raw in (f"x{LONE_HIGH}", LONE_HIGH * 5, f"{LONE_HIGH}1{LONE_HIGH}2"):
        sanitize_text(raw).encode("utf-8")  # must not raise


def test_dkgnode_sanitizes_title_and_text():
    node = DKGNode(
        id="n1", type="page", title=f"T{LONE_HIGH}",
        text=f"body {LONE_HIGH}here", order=0,
    )
    assert node.title == "T"
    assert node.text == "body here"
    node.text.encode("utf-8")
    node.title.encode("utf-8")


def test_ir_blocks_sanitize_text():
    assert Block(text=f"b{LONE_HIGH}x", page=1).text == "bx"
    assert PageBlock(page=1, text=f"p{LONE_HIGH}y").text == "py"


def test_content_hash_is_computed_from_sanitized_text():
    # finalize() hashes page text as the re-ingestion idempotency key. If
    # sanitizing happened after hashing, the same document would hash
    # differently on either side of this fix and re-ingest needlessly.
    from src.unstructured.document.ir import DocumentIR

    dirty = DocumentIR(source_name="d", page_count=1,
                       pages=[PageBlock(page=1, text=f"same{LONE_HIGH}text")]).finalize()
    clean = DocumentIR(source_name="d", page_count=1,
                       pages=[PageBlock(page=1, text="sametext")]).finalize()
    assert dirty.content_hash == clean.content_hash
