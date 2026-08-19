"""A printed page label must fit the sequence the document establishes.

detect_document_page_label reads one page at a time, so anything sitting
alone on a line near an edge can pass for a page number. On the 52-page
Go.Data report that produced three wrong labels: a decorative drop-cap "A"
on PDF 9 and "N" on PDF 37 (both matching the bare-letter pattern), and the
cover's "2021" on PDF 2 (a year matching the bare-digit pattern).

"Page A" cannot be navigated to and cannot be checked by a reader, so a
citation to it is worth no more than no citation at all.
"""
from src.document.page_numbers import label_to_number, reconcile_page_labels


def _godata_pages():
    """The labels per-page detection actually produced for that report."""
    pages = [(1, None), (2, "2021"), (3, None)]
    pages += [(4, "iv"), (5, "v"), (6, "vi"), (7, "vii")]     # roman, offset 0
    pages += [(8, "1"), (9, "A")]                              # arabic restarts
    pages += [(p, str(p - 7)) for p in range(10, 37)]          # offset -7
    pages += [(37, "N")]
    pages += [(p, str(p - 7)) for p in range(38, 52)]
    pages += [(52, None)]
    return pages


def test_label_to_number_keeps_the_numbering_system():
    """A document restarts at 1 when it moves from roman front matter to the
    arabic body, so "iv" and "4" must not vouch for each other."""
    assert label_to_number("12") == (12, "arabic")
    assert label_to_number("iv") == (4, "roman")
    assert label_to_number("A") == (None, None)
    assert label_to_number(None) == (None, None)


def test_only_the_corrupt_labels_change():
    resolved = reconcile_page_labels(_godata_pages())
    changed = {pdf: (was, resolved[pdf]) for pdf, was in _godata_pages() if was != resolved[pdf]}
    assert set(changed) == {2, 9, 37}, changed


def test_a_drop_cap_letter_is_replaced_by_the_interpolated_number():
    """PDF 9 sits between PDF 8 and PDF 10, which agree on offset -7."""
    resolved = reconcile_page_labels(_godata_pages())
    assert resolved[9] == "2"
    assert resolved[37] == "30"


def test_a_cover_year_with_no_supporting_run_is_dropped_not_guessed():
    """"2021" implies an offset of 2019 that no neighbour shares, and no run
    brackets PDF 2 -- so report no printed label rather than a wrong one."""
    assert reconcile_page_labels(_godata_pages())[2] is None


def test_good_labels_survive_including_the_ends_of_runs():
    resolved = reconcile_page_labels(_godata_pages())
    assert resolved[4] == "iv" and resolved[7] == "vii"   # roman run intact
    assert resolved[8] == "1"                             # first of the arabic run
    assert resolved[51] == "44"                           # last of the arabic run


def test_nothing_is_invented_past_the_end_of_a_run():
    """A final page is often deliberately unnumbered. Inventing a label there
    would be a fabricated citation of the exact kind this work prevents."""
    resolved = reconcile_page_labels(_godata_pages())
    assert resolved[52] is None
    assert resolved[1] is None


def test_roman_run_is_not_vouched_for_by_the_arabic_one():
    """Offsets that coincide across numbering systems must not merge: roman
    iv at PDF 4 is offset 0, and an arabic 4 at PDF 4 would be too."""
    resolved = reconcile_page_labels([(1, "i"), (2, "ii"), (3, "3"), (4, "4"), (5, "5")])
    assert resolved[1] == "i" and resolved[2] == "ii"
    assert resolved[3] == "3" and resolved[4] == "4"


def test_a_document_with_no_detectable_labels_is_left_alone():
    resolved = reconcile_page_labels([(1, None), (2, None), (3, None)])
    assert set(resolved.values()) == {None}


def test_an_isolated_page_keeps_nothing_it_cannot_corroborate():
    """One page, one label, nothing to check it against."""
    assert reconcile_page_labels([(5, "99")])[5] is None
