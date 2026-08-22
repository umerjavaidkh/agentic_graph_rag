"""text_sanitize.py — make extracted text safe to encode as UTF-8.

PDF text extraction can emit *lone UTF-16 surrogates*: code points in
U+D800-U+DFFF that are only meaningful as a high/low pair. Python's `str`
holds them happily, so nothing fails at extraction time, but they cannot be
encoded to UTF-8 -- so the failure surfaces much later, at the storage
boundary, as

    'utf-8' codec can't encode character '\\ud835' in position 73:
    surrogates not allowed

and takes the whole document down with it. Observed with the rtldoc backend
on scientific PDFs: one 13-page arXiv paper carried 219 lone `\\ud835` --
the high half of the Mathematical Alphanumeric block (the italic variables
in `x_1`, `A_ij`), whose low half was lost during extraction -- plus 8
surrogate pairs that were still intact.

Those two cases want different handling, which is why this is not a plain
strip:

  * a *valid pair* is a real character that merely survived extraction in
    its UTF-16 form. Encoding to UTF-16 and decoding back recombines it, so
    `x` stays `x` instead of vanishing from the text a reader searches.
  * a *lone* surrogate has no recoverable meaning -- its other half does not
    exist -- so it is dropped.

Applied at the boundaries every document crosses (`DKGNode`, and the
`Block`/`PageBlock` IR), so no individual parser backend, exporter, or
embedding call has to remember to do it.
"""
from __future__ import annotations

import re

_SURROGATE = re.compile(r"[\ud800-\udfff]")


def sanitize_text(value: str) -> str:
    """Return `value` with surrogate pairs recombined and orphans removed.

    A no-op (returning the identical object) for text that has no
    surrogates at all, which is the overwhelmingly common case -- the scan
    is a single pass in C and costs nothing worth measuring, but building a
    new string for every block would.
    """
    if not value or not _SURROGATE.search(value):
        return value
    # Round-tripping through UTF-16 rejoins any high/low pair into the
    # single character it encodes; `errors="ignore"` on the decode drops
    # halves that have no partner.
    recombined = value.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    # A pair split across a chunk boundary can still leave one half behind.
    return _SURROGATE.sub("", recombined)
