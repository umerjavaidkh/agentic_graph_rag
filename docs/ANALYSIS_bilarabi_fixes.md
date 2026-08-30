# BilArabi — what needs fixing, by layer

Evidence from re-ingesting `BilArabi_TG07.pdf` (239 pages) under
rtldoc **1.2.0** and asking 10 grounded Arabic questions: **7/10** answered
from the document. All three failures trace to two defects, in two
different layers.

---

## Layer 1 — Parser (rtldoc). Open in 1.2.0.

### 1.1 Lam-alef ligature stored with two characters transposed — HIGH

- **Scale:** 2,719 reversed pairs in 1.2.0's own JSON output. All four
  ligature variants, not just one:
  `األ` 1,319 · `اإل` 544 · `اال` 469 · `اآل` 387
- **Effect on words:** `الاصطناعي` occurs **0** times correctly and 13
  times as `االصطناعي`. Same for `الأول`, `الآخرين`, `الإنترنت`.
- **Effect on retrieval:** question 5 — *"ما هو الذكاء الاصطناعي **في هذا
  الدليل**؟"* — is answered from `doc_arwiki_ai`, a different document,
  because BilArabi's copy of the word cannot match what a reader types.
- **Effect on answers:** visible in returned text — `اآلخرِ` for `الآخر`,
  `واالج…` for `والاج…`.
- **Root cause:** lam-alef is a *mandatory* ligature drawn as one glyph
  `ﻻ` (U+FEFB). The PDF's `ToUnicode` maps it back to two components in
  **visual** order; RTL puts the alef left of the lam, so reading
  left-to-right yields alef-then-lam. Not bidi reversal (lines read
  correctly in logical order) and not presentation forms (44,799 base
  letters vs 101 presentation forms).
- **Not rtldoc-specific:** PyMuPDF reproduces it at 92% against rtldoc's
  79% — a PDF-level defect that the Arabic-aware parser is the right place
  to repair.
- **The fix signal — structural, not linguistic:**

  | | count |
  |---|---|
  | zero-width alef-variant followed by lam | **4,218** ← the ligature |
  | normal-width alef followed by lam | 16,893 ← real definite article |
  | zero-width alef *not* before a lam | **0** ← perfectly specific |

  The lam after a zero-width alef measures **6.08** wide against **3.05**
  for a normal lam: it is the ligature glyph carrying both letters.
  Rule: *zero-width alef-variant + lam → emit lam, then alef-variant.*
- **Validated:** `الاجتماعي` 0→41, `الاستماع` 0→30, `الاصطناعي` 0→10,
  `الآخرين` 0→29, `الأسئلة` 0→59; invalid `األ` 1085→0, `اإل` 442→0,
  `اآل` 262→0, `اال` 331→4.
- **Why it must be fixed in the parser:** the corruption is character
  *order*. A normalizer cannot reorder, and a text-level regex is
  ambiguous for a bare lam-alef (`طلاب` corrupts to `طالب`, itself a real
  word). At glyph level it is unambiguous — and the glyph signal also
  catches those bare cases (4,218 vs 2,118 findable in text).
- **Caveat to check first:** these measurements come from
  `get_text("rawdict")` char bboxes. If rtldoc's own pipeline normalises
  widths before the point a fix would sit, the zero-width signal has to be
  captured earlier, at primitive extraction.

### 1.2 Diacritics detached from their base letter — MEDIUM

- 2.9% of combining marks (1,159 of 40,214) are separated by a space:
  `ال َّتطبيق` for `التطبيق`.
- Splits one word into several tokens, so neither the raw nor the
  normalized form matches.
- Normalization cannot repair it — `ال َّتطبيق` normalizes to `ال تطبيق`,
  still two tokens.

### 1.3 Column interleaving on overlapping-text pages — MEDIUM

- One title came out as `الحقيقأيةُن؟َ الحقيقاةلُ؟هويّاتُ` — two columns
  merged character by character.
- Matches the known rtldoc overlapping-text behaviour.

---

## Layer 2 — Heading consumption (this repo). The parser fix already
## landed and is being discarded downstream.

### 2.1 rtldoc's heading work does not reach the graph — HIGH

- **Standalone**, rtldoc 1.2.0 parses this document into **42 headings, 0
  unusable, longest 163 chars**.
- **Through the pipeline**, the graph gets **512 Chapter nodes for 239
  pages**, **187** of them titled literally `"Preamble"` — byte-identical
  to what 1.1.0 produced. The 17.1% → 2.3% corpus improvement is
  **invisible downstream**.
- **Effect on retrieval:** question 9 — *"ما هي عناوين الوحدات في
  الدليل؟"* — returns `1. الوحدة 2 الوحد / 2. الوحدة 2 الوحد`.
- **Root cause:** `Axis1StructuralBuilder._is_heading`
  (`src/unstructured/graph/axis1_structural.py:823`) trusts rtldoc's
  *positive* call outright:

  ```python
  if block.source == "rtldoc" and block.extra.get("heading_hint") == "heading":
      return True
  ```

  but a *negative* falls through to a font/regex "rescue". Only
  `{figure, table, activity_marker}` are vetoed. BilArabi's block roles
  are **paragraph 1,389 · activity_marker 1,022 · passage 294 ·
  page_furniture 250 · table 213 · heading 42 · figure 42** — so
  paragraph, passage and page_furniture (1,933 blocks) all reach the
  rescue, which promotes roughly 470 of them.
- **The rescue is load-bearing — do not simply delete it.** Tried a
  blanket veto on body roles; it broke
  `test_rtldoc_missed_heading_rescued_by_font_geometry`, which pins a real
  live regression: rtldoc role `passage`, bold, 11pt against a 10.5
  threshold, correctly rescued as a heading.
- **What is still needed before changing it:** measure *which* of the five
  rescue branches promotes the ~470 blocks —

  | branch | line |
  |---|---|
  | `parse_numbered_title` | 915 |
  | `_COMMON_HEADING` regex | 918 |
  | bold ≥ threshold | 920 |
  | font ≥ threshold + 1 | 922 |
  | uppercase ratio > 0.72 | 924 |

  A teacher's guide is full of `1.` `2.` numbering, so
  `parse_numbered_title` is the first suspect — but this must be measured,
  not assumed. The likely shape of the fix is *narrowing* the rescue for
  rtldoc body roles to the typographic branches only, keeping the
  behaviour the test pins.
- Note `uppercase_ratio` is meaningless for Arabic (no case), so that
  branch cannot fire here — one fewer suspect.

---

## Layer 3 — Retrieval. No defect of its own.

- 7/10 answered correctly from the document; the unanswerable question was
  correctly refused in Arabic.
- Question 2 (`من مؤلفة دليل المعلم؟`) returns the **correct** answer
  (`أ.د. هنادا طه تامير`) but mixes in `doc_arwiki_education` — "دليل
  المعلم" is generic enough that an education article also matches. Loose
  scoping, low priority.
- Questions 5 and 9 fail for the parser/pipeline reasons above, not
  because retrieval is wrong.

---

## Priority

| # | fix | layer | impact |
|---|---|---|---|
| 1 | lam-alef transposition at glyph level | rtldoc | wrong-document answers; ~2,700 words/doc |
| 2 | stop discarding rtldoc's heading roles | this repo | 512 → ~42 chapters; fixes structure questions |
| 3 | detached diacritics | rtldoc | token splitting |
| 4 | column interleaving | rtldoc | corrupt titles |
| 5 | tighter document scoping | this repo | cosmetic |

Items 1 and 2 are independent and can be done in parallel — 2 needs no
parser change and is entirely inside this repository.
