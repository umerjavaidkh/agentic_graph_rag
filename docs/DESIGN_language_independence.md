# English + Arabic: the plan

Supersedes the earlier draft of this file, which was written for a
"support 100 languages" target before the scope was set. The decisions
below are the ones actually taken.

## Decisions

| | |
|---|---|
| Languages | English and Arabic only |
| Corpora | Two, fully separate |
| Deployment | Two instances, routed by language |
| Code | **One codebase.** Two deployments, never two forks |
| Cross-lingual (Arabic query → English document) | Out of scope for now |
| Corpus translation | Never |
| Longer term | This becomes a compliance / decision-support engine |

Separate instances were chosen because they make English regression
structurally impossible: no shared index, no shared statistics, no shared
analyzer. After an arc spent proving that changes did not regress, that
guarantee is worth its operational cost.

The cost is real and should be planned for: two Neo4j + Qdrant + MinIO +
worker sets. The Docker VM on this machine has already had to be rebuilt
once after filling up. If it becomes a problem, the fallback with almost
the same isolation is a single deployment with `language` as a filter
beside `tenant_id` -- additive, so English behaviour still cannot change.

## The one rule everything else depends on

**Language is data -- a filter value and a config key. It is never a
branch in retrieval logic.**

The repository already contains the cautionary tale. Five structural
fast-paths each resolved their own document; the document-scoping fix
landed in one of them and there were six, so TOC questions kept answering
from the wrong document until PR #120 deleted the lot. An
`if lang == "ar"` inside retrieval rebuilds exactly that, with languages
instead of question shapes.

Everything language-specific goes in a `LanguageProfile` -- normalizer,
stopword source, structural vocabulary, parser backend -- resolved once
from the request parameter and passed down. Adding a language is
registering a profile, the same plug-and-play shape as the retrieval
strategies.

## What is already language-independent

Listed because the instinct is to rewrite more than necessary.

- **The graph.** Chapter / Section / Page / Region, counting, outlines,
  address-by-number. A Section is a Section in any script.
- **Vector scoping** (`candidate_docs.py`). Reads no words.
- **`CorpusTermStats`.** Learns generic vocabulary from the corpus
  instead of listing it. On an Arabic corpus it will find Arabic generic
  terms with no code change -- and with separate corpora it is already
  correctly partitioned.
- **LLM-based NER**, which is multilingual by nature.
- **The eval harnesses.** They build questions from each document's own
  text, so they work in whatever language the corpus is in, once the
  regexes below are fixed.

## Phase 0 — Unicode correctness

Blocks everything else. Not a design question: the Arabic instance
cannot work without it, and it fixes a live silent bug.

24 regexes use ASCII-only letter classes. Measured behaviour today:

    Résultats   -> ['sultats']     the accent splits the word
    Prévisions  -> ['visions']     a real English word, out of French text
    النتائج      -> []              nothing matches at all

Arabic fails cleanly. Accented Latin fails *silently and plausibly*,
producing tokens that then pollute frequency statistics and match
unrelated English text. Both are wrong; the second is worse because
nothing looks broken.

Work: replace ASCII classes with Unicode-aware ones across the 24 sites
(`patterns.py`, `page_numbers.py`, `document_resolver.py`,
`region_description.py` and the rest); add NFKC normalization to
`text_sanitize.py` so it applies to stored text and queries alike.

Verification: the English suite must be byte-identical -- diff the
failure set against master, as PR #120 did. English text contains no
characters this changes, so any English movement is a bug in the sweep.

## Phase 1 — the language parameter and the profile seam

Work:

- `language` on the request, validated rather than trusted: use the
  parameter as primary and detected query language as a sanity check. A
  user whose locale is `ar` typing an English question should widen, not
  fail.
- `language` stored **per node**, not per document. An Arabic document
  quoting English regulation has English sections, and mislabelling them
  makes their `search_text` matching wrong.
- `LanguageProfile` registered by code, with English as the default
  profile built from today's behaviour, so the English path is unchanged
  by construction.
- Routing to the right instance in front of the API.

Verification: English numbers unchanged; a request with `language=ar`
reaches the Arabic instance and returns empty rather than wrong, since
nothing is ingested there yet.

## Phase 2 — Arabic ingestion

The part that cannot be fixed later: what the parser fails to store is
not recoverable at query time.

- Arabic normalization in the profile: alef variants (أ إ آ → ا), teh
  marbuta (ة → ه), diacritics/tashkeel, tatweel (ـ). This is also where
  dialect spelling variation collapses, which is the concern that
  motivated separate corpora in the first place.
- Arabic structural vocabulary: الفصل / القسم / الجدول / الشكل for
  headings, captions and address lookups, in the profile rather than in
  `structural.py`.
- OCR quality. Arabic OCR is materially worse than Latin, and this is
  where the parser work matters most. Expect the heading-quality figure
  to start worse than English's 6% and treat that as the baseline to
  improve, not as a failure.

Verification: run `scripts/audit_headings.py` against the Arabic corpus.
It is language-agnostic once Phase 0 lands, so the number is directly
comparable to English's 1,869 / 30,077.

## Phase 3 — Arabic retrieval quality

- Stopwords from corpus document frequency rather than the English
  `_KEYWORD_STOP` list. The mechanism exists and is measured; with
  separate corpora the table is already per-language. This is what fixes
  the underspecified gate, which currently stands down in every language
  but English -- measured at 1 content keyword for "What is the value?"
  against 4 for the French equivalent.
- Retire `_STEM_SUFFIXES` for Arabic. Nine English suffixes are wrong for
  a templatic morphology; rely on embeddings and use lexical matching for
  exact anchors only.
- Intent detection: Arabic patterns in the profile. The 77 regexes decide
  *shape*, not meaning, and English keeps its own accurate set.

Verification: `scripts/eval_shapes.py` and `scripts/eval_coverage.py`
against the Arabic corpus, reported next to the English baseline in
`eval/ALL_TESTS.md`.

## Phase 4 — answering

One prompt set, instructed to answer in the language of the question.
**Citations stay in the source language, byte-identical.** A translated
citation cannot be verified against the PDF, and every deterministic eval
in `eval/` depends on exact spans.

## What we are deliberately not doing

- **Translating the corpus.** Lossy, expensive, and it destroys
  exact-span checking.
- **Translating queries.** Only needed to make a literal lexical channel
  cross languages, and cross-lingual is out of scope. Note for later: if
  the corpora are ever merged, translating the query without fanning out
  across both language indexes biases retrieval toward the translated
  language, silently.
- **A multilingual embedding swap.** It buys cross-lingual retrieval,
  which is out of scope, and would change English behaviour. Revisit only
  if Arabic-to-English retrieval becomes a requirement.
- **Per-language forks of retrieval logic.** See the one rule above.

## Later: the compliance engine

Recorded now because it affects what is worth building.

The schema is already most of the way there. `logical_doc_id` +
revisions + `lifecycle_status` is temporal versioning, which is the hard
part -- answering "what did the rule say in March 2025", not only "what
does it say now". Two things already built that most RAG systems lack in
this setting: verbatim citations, and declining rather than guessing.

Three additions when it comes:

- **Effective date on revisions**, distinct from `ingested_at`.
  Compliance cares when a rule took effect, not when it was indexed.
- **Jurisdiction as a scoping dimension**, separate from language. Same
  shape as `tenant_id`. Language decides how you speak; jurisdiction
  decides what is true, and conflating them produces fluent, correctly
  translated, regulatorily wrong answers.
- **Diff between revisions.** "What changed" is the most-asked compliance
  question and both revisions are already stored.

## Order, and why

Phase 0 first because nothing Arabic is measurable without it and it
fixes a live bug in accented Latin. Phase 1 next because it is additive
and cannot move English. Phase 2 before Phase 3 because ingestion loss is
permanent and retrieval quality is not. Phase 4 last because it is one
line per prompt and depends on nothing.

Each phase carries the same verification: the English failure set diffed
against the previous commit, and the Arabic number reported next to the
English one rather than on its own.
