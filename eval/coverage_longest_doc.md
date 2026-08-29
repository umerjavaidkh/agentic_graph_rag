# Coverage — the longest document in the last 100 ingestions

`doc_arxiv_2608_19689`, 34 pages, the largest of the last 100 documents
ingested. 68 nodes carry usable text, spread across every page.

Every other harness here samples a document's nodes in `order` and stops
at the first 40-60, which on a 34-page paper means the questions all come
from the opening pages: a document could be answerable at page 3 and
invisible at page 30 and still score perfectly. This draws questions
evenly across the page range and reports the score by where in the
document the answer lives.

Questions are cloze over a stated fact, so the expected answer is
verifiable without a model. 18 questions landed on pages 1, 6, 10, 14, 18
and 30. Run 2026-08-29.

## The result depends entirely on whether the document is named

| Condition | Correct | Right document | Source node retrieved |
|---|---|---|---|
| **Document named** | **18/18** | **18/18** | **18/18** |
| Document not named | 10/18, of which 5 unproven | 5/18 | 9/18 |

### Named: complete, and flat with depth

| Slice | Page | Correct | Source node retrieved |
|---|---|---|---|
| 1/8 | 1 | 3/3 | 3/3 |
| 2/8 | 6 | 3/3 | 3/3 |
| 3/8 | 10 | 3/3 | 3/3 |
| 4/8 | 14 | 3/3 | 3/3 |
| 5/8 | 18 | 3/3 | 3/3 |
| 8/8 | 30 | 3/3 | 3/3 |

Page 30 answers exactly as well as page 1, and in every case the node the
question was built from is the node that came back. Within a document
that is named, coverage is complete -- there is no depth falloff to fix.

### Not named: the failure is placing the question, not reaching the page

| Slice | Page | Correct | Source node retrieved |
|---|---|---|---|
| 1/8 | 1 | 1/3 | 1/3 |
| 2/8 | 6 | 3/3 | 3/3 |
| 3/8 | 10 | 3/3 | 3/3 |
| 4/8 | 14 | 1/3 | 1/3 |
| 5/8 | 18 | 1/3 | 0/3 |
| 8/8 | 30 | 1/3 | 1/3 |

This is worth reading carefully rather than as a coverage number. A cloze
question has the distinctive term **removed** -- it is the blank. Asked
without naming the document, the query is the one sentence in the corpus
with its identifying word deleted, so document selection has less to work
with than any real question would. The same 18 questions, with the
document named, score 18/18.

So this measures how well an unscoped question can be placed, and it is a
deliberately hostile case. It is not evidence that page 30 is hard to
reach.

## Two corrections made while building this

**Number-only spans left three of eight slices empty.** `_FACT` matches
decimals, percentages and 3+ digit numbers, and the middle of this paper
has almost none, so three slices produced no question at all -- which
would have been reported as coverage of pages that were never asked
about.

**Document-local rarity is not distinctiveness.** The first fix filled
those gaps with words occurring once in this document, and drew
"captured", "separate", "according", "primarily" -- common words that
happen not to repeat here. Seven answers contained the expected word
while retrieval had resolved a different document: the question was
measuring the string, not the retrieval. Selection now uses corpus
document frequency, the same table built for the underspecified gate, and
draws terms like `SubjSim`, `personas`, `situational`, `precomputed`.

Both would have inflated the result. The 18/18 above is after them.

Harness: `scripts/eval_coverage.py` (`--named` for the scoped condition).
