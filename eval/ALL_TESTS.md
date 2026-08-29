# Every test run, in one place

One row per measurement actually taken, with what it covered, what it
found, and where the detail lives. Numbers here are copied from the runs
that produced them, not recalled.

Where a figure was later found to be wrong, both figures appear and the
reason is stated. Several of the corrections are to the measuring, not to
the system, and those are the ones worth reading twice: on this project a
bad metric has cost more time than a bad fix.

Last updated 2026-08-29, against master with #115, #116, #117 merged and
#118 open.

---

## 1. Shape eval — 20 question categories per document

The broadest measurement. Questions are generated from each document's own
text, so the expected answer is verifiable and no model is spent producing
them. Categories cover factoid, numeric, structural, enumerative,
aggregation, thematic, multi-hop, causal, comparison, verification,
unanswerable and ambiguous.

| Run | Documents | Questions | Right doc | Precision@k | Scored | Median latency |
|---|---|---|---|---|---|---|
| Second-last 5 of 1,001 | 5 (mixed parse quality) | 103 | 93/93 (100%) | 0.95 | **22/30 (73%)** | 2.0s |
| 5 cleanly-parsed | 5 | 102 | 92/92 (100%) | 0.95 | **26/30 (87%)** | 2.4s |
| 5 cleanly-parsed, after #118 | 5 | 102 | 92/92 (100%) | 0.95 | **30/30 (100%)** | 2.5s |

Per shape, most recent run: fact 5/5, numeric 5/5, enumerative 5/5,
aggregation 5/5, ambiguous 5/5, unanswerable 5/5.

Choosing cleanly-parsed documents was worth 14 points on its own (73% ->
87%) with no code change. That gap is heading corruption, not retrieval.

Detail: `shapes_second_last5_qa_log.md`. Harness: `scripts/eval_shapes.py`.

### Arabic, same harness, same shape count

37-document Arabic corpus (BilArabi teacher's guide + 36 Arabic Wikipedia
articles). Run as `eval_shapes.py --docs 5 --language ar`.

| Scope | Docs | Qs | Right document | Precision@k | Deterministic | Median |
|---|---|---|---|---|---|---|
| English baseline, 5 cleanly-parsed | 5 | 102 | 92/92 (100%) | 0.95 | 30/30 (100%) | 2.5s |
| **Arabic, 5 documents** | 5 | 94 | **84/84 (100%)** | **0.29** | **17/24 (71%)** | 0.4s |

Per shape (Arabic): fact 4/4, numeric 3/4, ambiguous 5/5,
unanswerable 5/5, enumerative 0/3, aggregation 0/3.

**Right-document is the number the language work was for, and it is
100%.** Every question resolved to the document it was generated from,
across a shared database holding 998 English documents alongside the 37
Arabic ones. Scoping, detection and normalization all hold at corpus
scale.

**The unanswerable score was 0/5 until the harness was fixed, and the
system was never wrong.** `REFUSAL` here was a tuple of English phrases.
Once the system began refusing in the language it was asked in, a correct
Arabic refusal matched none of them and scored zero -- an exam graded
against the wrong key, and indistinguishable in the summary from a system
that had started fabricating. Checked directly: 5/5 refusals, every one
of the form "هذا المستند لا يغطي ..." ("this document does not cover
..."). Refusal phrases now live in the language profiles.

**Precision 0.29 against English's 0.95 is real, and it is not
retrieval.** A precision of 0.00 here means no chunks came back at all,
and those rows return in 0.4s -- too fast for a model call. The document
picker is firing: with 37 topically-spread Arabic documents, a question
generated from one document's prose and not naming it is genuinely
ambiguous more often in Arabic than in English, because Arabic keyword
extraction is weaker. Two-letter function words are dropped by a token
floor written for English, there is no stemmer (correctly -- Arabic
morphology is templatic), and `_KEYWORD_STOP` is an English list, so an
Arabic question offers fewer usable anchors and looks underspecified.

That is the half of Phase 3 deliberately not done: document frequency is
wired to the underspecified GATE, not yet to keyword extraction. The
number above is what quantifies the cost of leaving it, and enumerative
0/3 and aggregation 0/3 are the same shortage seen from another angle --
both shapes need several anchors to hit a set.

Detail: `shapes_arabic_qa_log.md`.

## 2. Deterministic cloze eval — 500-document corpus

| Scope | Answer accuracy | Right document | Median |
|---|---|---|---|
| 20 documents x 5 questions | 87/95 (92%) | 100% | — |
| earlier pass | 80/94 (85%) | 94/94 (100%) | 16s |
| after fixes | 61/62 (98%) | 62/62 (100%) | 9s |

Detail: `corpus500_qa_log.md`.

## 3. Multi-turn scoping — 8 conversations, 3 turns each

T1 names the document, T2 names nothing and must stay, T3 names a
different document and must switch. T2 and T3 fail in opposite ways, so
neither always-trust-the-thread nor always-ignore-it can pass both.

**24/24 right document** (T1 8/8, T2 8/8, T3 8/8), 23/24 answered.

Detail: `multiturn_questions_seed5150.md`.

## 4. Category coverage — NIST SP 800-161r1

28 categories against one 887-node document, each answer checked against
the PDF rather than scored by reading it.

| Status | Count |
|---|---|
| Verified correct | 7 |
| Correct refusal | 5 |
| Verified wrong | 1 |
| Not verified | 14 |
| Unverifiable | 1 |

The honest headline is 12 confirmed good, 1 confirmed bad, and 15 not
actually checked. An earlier version of this table scored by reading the
answers and was wrong four times -- one question was about a document not
in the corpus, one was graded on keyword presence, one rested on an
unchecked premise.

Detail: `category_coverage_nist_161_verified.md`.

## 5. Fix verifications, end to end against the live stack

| Fix | Question set | Before | After |
|---|---|---|---|
| #115 underspecified gate | 10 generic + 8 answerable | ambiguous 0/5 | **18/18** |
| #116 counting from the graph | 7 counting questions | **0/6** | **7/7** |
| #117 region descriptions | 355 regions / 12 documents | titles were grid rows | 355/355 well-formed, 0 broken |
| #118 heading listing | 5 documents | 2/5 | **5/5**, every heading matched |

#116 before-figures, with the graph's true value:

| Question | Answered | Actual |
|---|---|---|
| How many tables in NIST SP 800-161? | 23 | 88 |
| How many chapters in NIST SP 800-161? | "3 main chapters" | 15 |
| How many tables in IRS Pub 225? | "does not cover" | 25 |
| How many pages in IRS Pub 225? | "does not cover" | 99 |

## 6. Ingestion quality — headings, whole corpus

| | |
|---|---|
| Documents scanned | 998 |
| Headings scanned | 30,077 |
| Unusable | **1,869 (6%)** |
| Documents >= 30% unusable | 68 |
| Documents fully clean | 717 of 986 |
| Documents needing re-parse | 270, all located on disk |

Failure modes: no words at all 749, line-break fragment 394, mostly
digits 348, body sentence 294, equation fragment 263, mid-sentence
fragment 202, table-grid row 31.

Detail: `heading_quality_report.md`, worklist in
`bad_heading_documents.md` / `.txt`. Harness: `scripts/audit_headings.py`,
`scripts/list_bad_heading_docs.py`.

## 7. Ingestion quality — tables and figures

12,818 Region nodes, 100% embedded and effectively unsearchable: the
title was the pipe grid's first row and the embedded text was punctuation
and `<br>` markers. Measured over 12 documents / 355 regions after #117:
well-formed titles 355/355, descriptions 355/355, broken titles 0/355,
real captions recovered 81/355 (23%).

Applies to documents ingested from #117 onward. The 12,818 existing
regions keep their old titles until re-ingested.

## 8. Unit tests

836 passing in `tests/unstructured`. Three failures are pre-existing
`rtldoc` `ModuleNotFoundError` and fail identically on master. A combined
run with `tests/interface` shows 9 failures on this branch and 9 on
master -- cross-suite pollution that predates this work.

---

## Measurement errors found, and what they cost

Every one of these made a working system look broken or a broken one look
fine. They are listed because the pattern repeats: the metric is the thing
most likely to be wrong.

| What was wrong | What it did |
|---|---|
| Precision counted REFUSE questions as 0.00 | Read as "no context retrieved" when sources were 6 and 4. Sent a whole round of work after a defect that did not exist. |
| Decline graded by string-matching prose | New wording "does not say which document to look in" matched no phrase in the list, so 5 correct declines scored 0/5 and read as a regression. |
| Graph chunks scored as wrong-document | Marker ids (`graph_outline`, `graph_count`) do not start with the doc id, so precision fell 0.95 -> 0.89 the moment answers came from the hierarchy. |
| "Fewer than two words" as a heading test | Flagged "Introduction", "Abstract", "Contents" -- 90% of all hits -- and reported 25% of the corpus broken. Real figure is 6%. |
| Math symbol alone as equation debris | Flagged "4.1 Global Optimality of pg = pdata", a real heading. |
| `"25" in "IRS Publication 225"` | Scored a "does not cover" answer as a hit. Baseline was 0/6, not 1/6. |
| Thread ids built from `time.time()` | A dot collapses a thread id to `default`, so "unique" threads shared one polluted conversation and its document hint. A working gate read as broken. |
| Sampling the most recent 500 | Recency correlates with document type -- that slice is almost all arXiv -- so it measured one family and read as a corpus number. |

## What is measured, and what is not

Scored deterministically: fact, numeric, enumerative, aggregation,
ambiguous, unanswerable -- 30 of 102 questions in a shape run. The other
72 are retrieval-only shapes whose answers are not machine-checkable;
they contribute right-document and precision figures but no correctness
score. Any claim about those is an impression, not a measurement.

Not measured anywhere yet: whether #117's region descriptions improve
retrieval recall. Verified what ingestion writes, not what retrieval then
finds, because the existing 12,818 regions need re-ingesting first.
