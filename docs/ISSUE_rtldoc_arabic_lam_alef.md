# Arabic lam-alef ligatures come out of the PDF in the wrong order

Open issue for the rtldoc parser. Written from a diagnosis on
`BilArabi_TG07.pdf` (239-page Arabic teacher's guide), where roughly
2,100 words are affected in a single document.

## Symptom

Words containing a lam-alef ligature are stored with two characters
transposed, so they never match what a reader types:

| reader types | stored as | occurrences in this document |
|---|---|---|
| `الاصطناعي` (artificial) | `االصطناعي` | 13 stored wrong, **0** right |
| `الاجتماعي` (social) | `االجتماعي` | 41 |
| `الاستماع` (listening) | `االستماع` | 30 |
| `الآخرين` (the others) | `اآلخرين` | — |
| `الإنترنت` (the internet) | `اإلنترنت` | — |

It is visible in answers, not just in storage: retrieval returns
`والعالقات` where the document says `والعلاقات`, and `اإلنترنت` where it
says `الإنترنت`.

**It changes which document is retrieved.** Asked
`ما هو الذكاء الاصطناعي في هذا الدليل؟` ("in this guide"), retrieval
answered from a different document entirely -- the guide's own copy of
the word is misspelled, so it could not match, while another document
that spells it correctly could.

## Root cause

Not visual-order (bidi) reversal, and not presentation forms. Checked:

- Arabic base letters U+0620-064A: 44,799. Presentation forms U+FB50+: 101.
  The PDF emits proper base letters, not a shaped-glyph dump.
- A stored line reads correctly in logical order
  (`تأليف أ.د. هنادا طه تامير`); reversing it produces gibberish. So the
  text is not stored right-to-left.

The defect is confined to the lam-alef ligature. Code points for one word:

```
stored   ALEF  ALEF  LAM   SAD TAH NOON ALEF AIN YEH     ا ا ل ص ط ن ا ع ي
correct  ALEF  LAM   ALEF  SAD TAH NOON ALEF AIN YEH     ا ل ا ص ط ن ا ع ي
               ^^^^^^^^^^  transposed
```

Lam-alef is a *mandatory* ligature in Arabic: `ل` + `ا` is drawn as the
single glyph `ﻻ` (U+FEFB). The PDF's `ToUnicode` CMap maps that one glyph
back to its two component characters -- and lists them in **visual**
order. Rendered right-to-left the alef sits to the left of the lam, so
reading the pair left-to-right yields alef-then-lam, which is the
reverse of the logical order.

**Any extractor that trusts `ToUnicode` reproduces this.** Confirmed on
the same file:

| extractor | share of lam-alef pairs reversed |
|---|---|
| rtldoc | 79% |
| PyMuPDF | 92% |

So it is not an rtldoc bug in the sense of rtldoc doing something the
others do not -- it is a PDF-level defect that rtldoc, being the Arabic-
aware parser, is the right place to repair.

## All four ligatures are affected

| ligature | stored | correct | count in this document |
|---|---|---|---|
| `ﻷ` lam + alef hamza above | `اأ` | `لأ` | 1,086 |
| `ﻹ` lam + alef hamza below | `اإ` | `لإ` | 442 |
| `ﻵ` lam + alef madda | `اآ` | `لآ` | 262 |
| `ﻻ` lam + alef | `اا` | `لا` | 332 |

## The repair

Swap the alef-variant back after the lam:

    ALEF  <alef-variant>  LAM   ->   ALEF  LAM  <alef-variant>

where `<alef-variant>` is one of `ا آ أ إ`. Verified on this document --
`اال` -> `الا` alone:

```
الاصطناعي   0 -> 10        الاجتماعي   0 -> 41
الاستكشاف   0 ->  4        الاستماع    0 -> 30
invalid 'اا' inside words: 332 -> 3
```

### Where it is safe, and where it is not

The rule above is written to fire only after a **preceding alef**, which
in practice is the definite article: `ال` + a word beginning with an
alef-variant. That case is unambiguous, and it is the overwhelming
majority here.

A bare lam-alef *not* preceded by the article is genuinely ambiguous
after the fact. `طلاب` ("students") corrupts to `طالب`, which is itself a
real word ("student"). Nothing downstream can tell those apart, which is
why this has to be fixed at extraction, where the ligature glyph is still
identifiable -- not by a regex over stored text.

### Do not fix it in the normalizer

`src/shared/language.py`'s Arabic normalizer folds alef variants and
strips diacritics for matching. It cannot repair this: the corruption is
character *order*, and the normalizer would happily fold both the right
and the wrong spelling to different results. Repairing it downstream also
leaves the stored text wrong, and citations are quoted verbatim.

## How to verify a fix

```
python - <<'PY'
import fitz, re
txt = "".join(p.get_text() for p in fitz.open("BilArabi_TG07.pdf"))
print("invalid intra-word 'اا':", len(re.findall("اا", txt)))
for w in ["الاصطناعي","الاجتماعي","الاستماع","الآخرين","الإنترنت"]:
    print(w, txt.count(w))
PY
```

A correct extraction has near-zero intra-word `اا`, `اأ`, `اإ`, `اآ`, and
non-zero counts for the words above.

## Separate issues found in the same document

Not this bug, tracked here only so they are not conflated with it:

- **512 Chapter nodes for a 239-page document**, and 187 of 984
  Chapter/Section titles are literally `"Preamble"` (19%). This is the
  chronic heading-detection gap, not Arabic-specific.
- **Column interleaving** on overlapping-text pages -- one title came out
  as `الحقيقأيةُن؟َ الحقيقاةلُ؟هويّاتُ`, two columns merged character by
  character. Matches the known rtldoc overlapping-text behaviour.
- **Detached diacritics**: 2.9% of combining marks are separated from
  their base letter by a space (`ال َّتطبيق` for `التطبيق`), which splits
  one word into several tokens.
