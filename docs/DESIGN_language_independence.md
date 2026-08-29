# English + Arabic: the plan

Supersedes the earlier draft of this file, which was written for a
"support 100 languages" target before the scope was set. The decisions
below are the ones actually taken.

## Decisions

| | |
|---|---|
| Languages | English and Arabic only |
| Corpora | Two, separated logically |
| Deployment | **One** — shared Neo4j, Qdrant, MinIO, workers |
| Separation | A `:Language` parent node, plus a `language` property on every node |
| Code | **One codebase**, one retrieval path |
| Cross-lingual (Arabic query → English document) | Out of scope for now |
| Scope | **Unstructured (document) retrieval only** |
| A document's language | One per document, English by default |
| A document containing both | Goes to Arabic — the non-default language wins |
| Corpus translation | Never |
| Longer term | This becomes a compliance / decision-support engine |

Separate instances were considered and rejected on cost: two full stacks
on a machine whose Docker VM has already had to be rebuilt once after
filling up, immediately after reclaiming 22GB. Logical separation gives
almost the same guarantee for less.

The guarantee still holds where it matters. The language predicate
compiles to `true` while only English is configured, so English
behaviour is byte-identical by construction rather than by testing.

### Shape

    (:Language {code:'en'}) -[:HAS_DOCUMENT]-> (:DocumentLogical)   998 edges
    (:Language {code:'ar'}) -[:HAS_DOCUMENT]-> (:DocumentLogical)

`DocumentLogical` is today's root of the document graph -- 998 nodes with
no incoming relationship. The Language node sits above it, which also
gives the document graph a single entry point it currently lacks.

### The structured path is out of scope

Language scoping applies to document retrieval and nothing else. The
structured business graph has no language dimension: its labels and
properties are schema, not prose, and an Arabic question about orders is
still answered by `MATCH (o:Order)`. `language_filter()` therefore goes
on document scope call sites only -- splicing it into a structured query
would scope a graph that has nothing to scope.

This is why the `:Language` node attaches to `DocumentLogical` and to
nothing else, and why `language` is stamped by the document ingestion
path rather than by anything shared with tabular ingest.

### How a document gets its language

English is the default: it is what a document is when no other profile
claims it, which is why every existing document backfills to `en` without
being examined. Any other language present in enough quantity wins over
the default -- an Arabic document quoting English regulation is Arabic,
and so is a document laid out in both.

"In enough quantity" is a *share* of the document's letters, not a
presence test. A presence test would move a 300-page English filing into
the Arabic corpus on one stray glyph, and OCR on scanned pages produces
stray glyphs routinely. The threshold is a setting rather than a
constant because the right value has to be measured against real
bilingual documents; it is not knowable in advance.

The rule is a precedence over registered profiles, not a test for
Arabic. Adding a third language is registering a profile with its
scripts -- the same plug-and-play shape as the retrieval strategies, and
the reason this is not written as `if lang == "ar"`.

**One language per document, stamped onto every node in it.** An earlier
draft of this file labelled each node by the language of its own text, so
the English sections of a bilingual document stayed reachable by an
English query. That was rejected: it means one document lives in two
corpora at once, and every count, every df table and every "which
document" answer has to say which half it means. The cost is real and is
recorded here rather than discovered later -- an English-scoped query
returns nothing from a bilingual document, including for content written
in English. Cross-lingual retrieval is what would fix that, and it is out
of scope by the same decision.

Cost: +2 nodes on 611,814 (0.0003%) and +998 relationships on 978,354
(0.1%). Attaching Language to *every* node instead would put 611,814
relationships on one node -- 62% of the graph on a single supernode --
which is why the edge stops at `DocumentLogical`.

Checked before committing to it: nothing traverses upward from
`DocumentLogical`; the only orphan detection is scoped to content labels
and the `:CONTAINS` type specifically, so a `:HAS_DOCUMENT` edge is
invisible to it; and purge already uses `DETACH DELETE`, so the new edge
is cleaned up with the document.

### Structure and speed are different jobs

| Purpose | Mechanism |
|---|---|
| Retrieval filtering | `n.language = $language` -- property, index-backed, no hop |
| Config lookup | traverse from `:Language`, once per request |
| Cascade, ops, reporting | traverse from `:Language` |

The edge is the authority; the property is derived from it at ingest and
stamped onto every node -- the same denormalisation `logical_doc_id` and
`lifecycle_status` already use, and for the same reason: scoped queries
must not have to traverse to find their scope.

Existing documents get `language='en'` **backfilled**, not null-guarded.
A null-guard in a scope predicate has already cost this project a
611,815-node scan on a single query; there is no reason to reintroduce
it.

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

- `language_filter(alias, param)` beside `tenant_filter` in
  `shared/neo4j/tenancy.py`, using the same idiom: it returns the string
  `"true"` while fewer than two languages are configured, so all 20
  existing scope call sites can splice it unconditionally with no
  behaviour change.
- `language` on the request, validated rather than trusted: use the
  parameter as primary and detected query language as a sanity check. A
  user whose locale is `ar` typing an English question should widen, not
  fail. Absent parameter defaults to `en`, so every existing caller keeps
  working untouched.
- `MERGE` the `:Language` node at ingest -- create-if-missing, never a
  precondition. A missing config node must not block an ingest.
- `language` stamped onto **every node of a document**, from the one
  language the document resolved to. See "How a document gets its
  language" above for why this is per document rather than per node, and
  what it costs.
- `LanguageProfile` registered by code, with English as the default
  profile built from today's behaviour, so the English path is unchanged
  by construction.

Verification: English numbers unchanged; a request with `language=ar`
reaches the Arabic instance and returns empty rather than wrong, since
nothing is ingested there yet.

## Phase 2 — Arabic ingestion

The part that cannot be fixed later: what the parser fails to store is
not recoverable at query time.

- Detection calibrated against real bilingual documents: the share
  threshold above is provisional until there is something to measure it
  on. The failure to watch for is a scanned English document whose OCR
  noise crosses the line, not a bilingual document that fails to.
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
