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
