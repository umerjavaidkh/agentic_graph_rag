"""Shared regex patterns for document structure parsing."""
import re

STANDALONE_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s*$")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")

TABLE_REF_PATTERN = re.compile(
    r"\btable\s+([a-z]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

REFERENCE_PATTERN = re.compile(
    r"(?:see|refer(?:s)? to|as (?:discussed|described|shown) in|"
    r"described in|mentioned in|in)\s+"
    r"(?:Chapter|Section|Appendix|Figure|Table|Part)\s+[\w\d\.]+",
    re.IGNORECASE,
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def number_depth(num_str: str) -> int:
    return len(num_str.rstrip(".").split("."))


def parse_numbered_title(text: str) -> tuple[str | None, str]:
    """Return (section_number, title) from '4.5 ENVIRONMENTAL PROTECTION' or plain title."""
    m = NUMBERED_HEADING.match(text.strip())
    if m:
        return m.group(1).rstrip("."), m.group(2).strip()
    return None, text.strip()


def is_standalone_number(text: str) -> bool:
    return bool(STANDALONE_NUMBER.match(text.strip()))
