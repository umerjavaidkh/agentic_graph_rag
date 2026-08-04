"""Shared regex patterns for document structure parsing."""
import re

STANDALONE_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s*$")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")

# SEC filings (10-K/10-Q) number their major headings "Item 1.", "Item 1A.",
# "Item 7A." rather than the dot-numbered scheme NUMBERED_HEADING covers —
# a completely different scheme (letter suffix, not a dot level), and one
# NUMBERED_HEADING can't match at all since these headings don't start with
# a digit ("Item " comes first). Without this, "Item 1A" never gets a
# section_number extracted, so it never nests under "Item 1" -- it's parsed
# as an unnumbered heading and only nests (or doesn't) based on font-size
# heuristics alone.
ITEM_HEADING = re.compile(r"^item\s+(\d+[a-z]?)\.?\s+(.+)$", re.IGNORECASE)
_ALPHA_SUFFIX = re.compile(r"^(\d+(?:\.\d+)*)([A-Za-z])$")

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


def _strip_item_prefix(num_str: str) -> tuple[str, str]:
    """Split "Item 1A" into (prefix "Item ", core "1A"); ("", num_str) if no prefix."""
    if num_str.lower().startswith("item "):
        return num_str[:5], num_str[5:].strip()
    return "", num_str


def number_depth(num_str: str) -> int:
    _, core = _strip_item_prefix(num_str)
    m = _ALPHA_SUFFIX.match(core)
    if m:
        # "1A" is one level deeper than "1" -- same nesting relationship a
        # dot level expresses, just via a letter suffix instead of a ".N".
        return number_depth(m.group(1)) + 1
    return len(core.rstrip(".").split("."))


def parent_number(num_str: str) -> str | None:
    """The number one level up from num_str ("Item 1A" -> "Item 1", "4.5.1"
    -> "4.5"), or None when num_str is already top-level."""
    prefix, core = _strip_item_prefix(num_str)
    m = _ALPHA_SUFFIX.match(core)
    if m:
        return f"{prefix}{m.group(1)}"
    parts = core.rstrip(".").split(".")
    if len(parts) < 2:
        return None
    return f"{prefix}{'.'.join(parts[:-1])}"


def parse_numbered_title(text: str) -> tuple[str | None, str]:
    """Return (section_number, title) from '4.5 ENVIRONMENTAL PROTECTION',
    'Item 1A. Risk Factors', or plain title. section_number is normalized
    ("Item 1a." / "ITEM 1A" both become "Item 1A") so number_map lookups
    are stable regardless of the source document's own casing/spacing."""
    m = ITEM_HEADING.match(text.strip())
    if m:
        return f"Item {m.group(1).upper()}", m.group(2).strip()
    m = NUMBERED_HEADING.match(text.strip())
    if m:
        return m.group(1).rstrip("."), m.group(2).strip()
    return None, text.strip()


def is_standalone_number(text: str) -> bool:
    return bool(STANDALONE_NUMBER.match(text.strip()))
