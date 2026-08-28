# Shape eval — second-last 5 documents of 1,001

Run 2026-08-29 against master (PRs #115, #116, #117 merged), 20 question
categories per document, 103 questions. Documents selected by ingestion
recency, stepping over the most recent five so this is a fresh sample
rather than the batch already reported on (`--docs 5 --skip 5`).

Questions are generated from each document's own text, so the expected
answer is verifiable and no LLM is spent producing them.

| Document | Chapters in graph | of those, unusable |
|---|---|---|
| arxiv_2608_20246 | 8 | 3 (38%) |
| arxiv_2608_20256 | 3 | 2 (67%) |
| arxiv_2608_20271 | 4 | 0 |
| arxiv_2608_20220 | 11 | 5 (45%) |
| arxiv_2608_20231 | 10 | 2 (20%) |

## Overall

| Metric | Result |
|---|---|
| Questions | 103 |
| Right document | **93/93 (100%)** |
| Precision@k (mean, scoped questions) | **0.95** |
| Recall@k (source node) | 14/18 |
| Deterministically scored | **22/30 (73%)** |
| Median latency | **2.0s** |
| Retrieval-only shapes | 73 (answer not machine-checkable) |

## Scored shapes

| Shape | Score | Note |
|---|---|---|
| 7 aggregation | **5/5** | was 1/5 before #116 |
| 28 unanswerable | 5/5 | |
| 27 ambiguous | **5/5** | after fixing the grader, see below |
| 1 fact | 4/5 | |
| 18 numeric | 3/5 | |
| 5 enumerative | 0/5 | ground truth is corrupt, see below |

## Two corrections to the measurement, not the system

**Ambiguous first scored 0/5 and is actually 5/5.** The grader decided a
question had been declined by looking for phrases like "does not cover"
in the answer. Retrieval now declines an unplaceable question with "does
not say which document to look in", which matches none of them, so five
correct declines scored as five failures. The grader reads the
`underspecified` flag off the response now and falls back to wording only
when there is no flag. Grading prose for a signal the response carries
explicitly is how a working fix reads as a regression.

**Precision excluded REFUSE questions.** Precision here means "share of
sources from the expected document", and a question meant to be declined
has no expected document, so those scored 0.00 and dragged the mean down.
They are excluded rather than counted as zero; the 0.95 above is over the
93 questions that actually have an expected document.

## Enumerative 0/5 is an ingestion defect, not a retrieval one

Asked to list every chapter heading, the system answered "This document
does not cover chapter headings." It was right to. The headings stored
for that document are:

    'XXℓi,t, (3) LTM ='
    '1000 2000 3000 4000 5000 6000 7000 8000 Toke'
    '6000 8000 10000 12000 Mean response length ('

Chart axis tick labels and math fragments, detected as chapter headings
at ingest. The eval expects them to be recited; declining is the better
answer, and no retrieval change can fix it.

Corpus-wide, sampling 8,000 chapter titles:

| Group | Sampled | Unusable |
|---|---|---|
| All | 8,000 | **3,106 (39%)** |
| arXiv | 7,106 | 2,857 (40%) |
| non-arXiv | 894 | 249 (28%) |

A title counts as unusable when it has fewer than two real words or is
more than 30% digits.

This also qualifies the aggregation 5/5 above: those counts are correct
*against the graph*, and for these documents the graph is counting
chart labels as chapters. Heading detection is the next thing worth
fixing, and it is upstream of both shapes.
