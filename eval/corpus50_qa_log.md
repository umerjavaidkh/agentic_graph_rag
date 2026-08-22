# Retrieval spot-checks — 50-document corpus

Questions asked against the ingested corpus, with the answer verified in the
PDF **before** asking. Ground truth is the document, not a model, so every
row here is checkable by opening the file at the page named.

Pattern: the document is named in the first turn, the question asked in the
second without naming it. Thread memory carries the scoping — this is the
difference between 1/5 and 4/5 measured on 2026-08-22.

## Round 1 — 2026-08-22

| # | Document | Page | Expected | Question | Result |
|---|---|---|---|---|---|
| 1 | irs_p502 | 11 | `$4,810` | Most includable as a medical expense for long-term care premiums, age 61–70 | ✅ |
| 2 | irs_p559 | 10 | `$108.28` | Church-employee wages triggering self-employment tax on a decedent's final return | ✅ |
| 3 | nist_ir_8286 | 44 | `$750,000` | Estimated cost in the stolen-laptop example | ✅ |
| 4 | irs_p575 | 6 | `$7,500` | Length-of-service award threshold for volunteer firefighters | ✅ |
| 5 | irs_p575 | 10 | `$150,000` | AGI above which 110% replaces 100% for estimated tax | ❌ page retrieved, ranked 5th, synthesis missed it |

**4/5 correct · 5/5 cited the right page.**
Unscoped (no document named in turn 1) the same five scored **1/5**.

## Round 2 — 2026-08-22

| # | Document | Page | Expected | Question | Result |
|---|---|---|---|---|---|
| 6 | arxiv_alexnet_ref | 7 | `42.63` | Top-1 error for 2 GPUs at batch size (256, 256) | ✅ |
| 7 | arxiv_attention | 1 | `28.4` | BLEU on WMT English-to-German | ❌ resolved to `doc_arxiv_t5` — wrong document |
| 8 | irs_p517 | 10 | `$125` | Nondeductible portion of wedding and baptism income | ✅ |
| 9 | nist_ir_8228 | 33 | `CP-9` | Control cited for missing secure backup/restore | ⚠️ answered `PR.IP-4` — right page, but the page cites several controls; the question was underspecified |
| 10 | irs_p926_household | 2 | employment taxes | Federal employment taxes a household employer may owe | ✅ |

**3/5 correct · 4/5 right document · 3/5 right page.**

## Failures worth acting on

**Wrong document (row 7).** "the arXiv paper Attention Is All You Need" resolved to
`doc_arxiv_t5`. Four transformer papers share most of their vocabulary, and the
title carries no number for the identifier path to latch onto — the case the
document picker exists for.

**Ranked but not used (row 5).** The right page *was* retrieved, at position 5,
and synthesis still answered "not specified". A ranking and synthesis problem,
not a retrieval one.

**Underspecified question (row 9).** Page 33 cites several controls; asking
"which control" has more than one defensible answer. Recorded as a flaw in the
question, not the system.

## Method notes

- Facts chosen to appear on exactly one page, so a citation is checkable.
- Amounts and years are avoided as identifiers — `$559` must not be read as
  Publication 559.
- Each question runs in its own thread, so no scoping leaks between rows.

## Follow-up on row 7 — the picker is bypassed

The strict resolver handles this case correctly. Asked for
"the arXiv paper Attention Is All You Need" it declines, and the candidates
it computes rank the right paper first:

```
strict resolver : None            (correctly declines)
  arxiv_attention   rel = 1.000   <- correct
  arxiv_t5          rel = 0.972   <- what the pipeline chose
  nist_sp800_161    rel = 0.727
```

A 2.8% gap between the top two: a real near-tie, and precisely what the
picker exists to resolve.

The pipeline never gets there. `resolve_document_for_query` has fallback
tiers below the strict pass that pick something rather than nothing, so a
near-tie is silently resolved to the runner-up instead of being offered as
a choice.

**Next fix:** when the strict pass declines *and* the candidates are within
`AMBIGUITY_LEAD` of each other, prefer asking over guessing. Four documents
sharing a vocabulary is not a case more heuristics will separate — it is a
case where the user knows and the system does not.

## Was it the model change? Partly — and it is now split by job

The synthesis model was swapped from `gpt-4o-mini` to `gpt-4.1-nano` earlier
the same day, to escape a per-day request cap. Re-running the failures with
only that setting changed back, against the identical graph and retrieval:

| question | gpt-4.1-nano | gpt-4o-mini | cause |
|---|---|---|---|
| NIST `CP-9` | answered `PR.IP-4` | **correct** | **the model** |
| IRS `$150,000` | wrong once, right on re-run | correct | non-determinism |
| arXiv `28.4` | wrong document | wrong document | resolver, not the model |

Page 33 of NIST IR 8228 lists both `CP-9, System Backup` (an SP 800-53
control) and `PR.IP-4` (a Cybersecurity Framework subcategory) against the
same risk. Asked specifically for the SP 800-53 control, nano returned the
CSF subcategory. Both were on the page; only one answered the question.

The two models were compared on entity extraction before the swap, where
nano looked better. Synthesis was never tested, and nano's coarseness --
visible then as typing almost everything `MISC` -- shows up here as failing
to discriminate between two similar labels.

Settings now differ by job, which they always could:

- `AXIS2_MODEL = gpt-4.1-nano` — ingestion, thousands of calls, rate-limited,
  and the work is coarse extraction
- `CHAT_MODEL = gpt-4o-mini` — synthesis, one call per question, where the
  distinction between two labels on a page is the whole answer

Also worth recording: **the same question failed and then passed on the same
model**, so a single pass over five questions carries real noise. Treat these
numbers as directional until a suite runs them repeatedly.

## Re-run after the fixes — 10/10

All ten questions, current configuration (`CHAT_MODEL=gpt-4o-mini`,
`AXIS2_MODEL=gpt-4.1-nano`, ambiguous references declining to the picker):

```
10/10 correct · 9/10 resolved a single document · 8/10 cited the page I expected
```

The two rows where the document or page column reads ✗ are measurement
artifacts, not failures:

- **arxiv_attention** — `document_id` is null because the resolver correctly
  declined between four transformer papers and offered the picker. Retrieval
  still found the answer, and the cited sections (`Results`,
  `Machine Translation`, `Model Variations`) all belong to
  `doc_arxiv_attention`. My expected page was 1, the abstract; page 8 restates
  the same figure, so both are ground truth.
- **irs_p926_household** — the expected page was my guess for a general
  question with no single home.

## What each fix was worth

| fix | rows recovered |
|---|---|
| Naming the document in turn 1 (thread scoping) | 3 |
| `CHAT_MODEL` back to gpt-4o-mini for synthesis | 1 |
| Document numbers parsed and matched (`Publication 559`) | resolution 0/5 → 4/5 |
| Ambiguous references decline instead of guessing | 1 |

## What this number is and is not

Ten questions, one pass. During this work the same question failed and then
passed with nothing changed, so a single pass carries real noise and 10/10
should be read as "no known failures" rather than as a score.

They are also all single-hop lookups: find a page, read a value. Nothing here
tests multi-hop or cross-document reasoning, which is where a graph should
beat flat retrieval. That suite does not exist yet, and it is the honest next
thing to build.

## Why some questions take minutes — a composite index missed by half

Measured on the Go.Data report, same thread, document already grounded:

```
"What is Box 9 about in this report?"          0.5s
"What is the table of contents?"               4.3s
"What does Figure 1 show in this report?"    143.1s
"What does the list of abbreviations...?"    183.2s
```

Sampling Neo4j during a slow one catches a single query running 21+ seconds,
scoped by document. The index exists:

```
Page(logical_doc_id, lifecycle_status)      composite
Section(logical_doc_id, lifecycle_status)   composite
```

Composite indexes are only seekable when the predicate names the leading
properties. Measured on Page:

| predicate | DbHits |
|---|---|
| `logical_doc_id = $doc_id` | **1,110,037** |
| `logical_doc_id = $doc_id AND lifecycle_status = 'ACTIVE'` | **53** |

A factor of ~21,000. Any retrieval path that scopes to a document without
also filtering `lifecycle_status` scans the whole database -- which now holds
the structured business graph as well, so the scan is over half a million
nodes rather than a few thousand pages.

Fast paths (Box, TOC) filter lifecycle. Slow paths do not. That is the whole
difference between 0.5s and 183s, and it explains the streaming client
appearing to hang: the answer is correct and arrives, minutes later.

**Fix, not yet applied:** make the shared scope fragments in
`retrieval/cypher_scope.py` always emit the lifecycle predicate alongside the
document predicate, so a scoped query cannot accidentally omit half of its
own index.
