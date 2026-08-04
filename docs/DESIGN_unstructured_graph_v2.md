# Design — Unstructured Ingestion & Graph v2

> Status: **planning** (no implementation yet). Branch `redesign/unstructured-graph-v2`.
> Scope: **unstructured/document path only**. Structured Northwind path untouched.
> Scale target: **~100k docs, one domain**.

## 1. Why

Today the document path entangles three concerns that should be separate:

- **Parsing** — `DocumentParser.parse(source) -> (list[DKGNode], list[DKGEdge])`. The parser *is* the graph builder.
- **Graph construction** — happens inside parser + exporter, no dedicated owner.
- **Storage** — the exporter writes every node with `SET n = row`, where the row carries full `text` **and** the raw `embedding` vector as Neo4j properties. That same content is *also* dual-written to MinIO (`blob_key_text`) and Qdrant. Every Region is materialized **three times**, and Neo4j's property store carries millions of text blobs + dense vectors.

Retrieval then reads `n.text` / `n.embedding` / the Neo4j full-text and vector indexes straight off those properties.

**Goal:** decouple the three. Parser emits a neutral representation; a dedicated service builds the graph in two explicit axes; text lives in blobs, vectors in a dedicated store, and **Neo4j holds only structure + idea-graph + pointers**.

## 2. Governing principle — plug-and-play modules

Every stage is a **swappable module behind an interface + registry**, with a default implementation, so different techniques can be A/B compared without rewiring callers. Mirrors existing seams (`DocumentParser`/`parser_registry`, `VectorStore`/`BlobStore` ABCs, `register_unstructured`/`register_structured`). No technique is hardcoded inline where a second could plausibly be tried.

## 3. Target pipeline

```
Parser ──► Document IR ──► Chunker ──► GraphConstructionService ──► Loaders
(→ IR,      (storage-        (→ retrieval   ├─ Axis1StructuralBuilder   ├─ BlobLoader   → MinIO (full text, authoritative)
 no graph)   agnostic         units)         └─ Axis2IdeaBuilder         ├─ VectorLoader → Qdrant (embeddings, authoritative)
             block tree)                                                 └─ GraphLoader  → Neo4j (lean: structure + ideas + pointers + search_text)

Retrieval ──► Hydrator (pull text from blob via pointer) ──► strategies (registry)
```

## 4. Module map (interfaces + default impls)

| Stage | Interface | Default impl | Swap to try… | Status |
|---|---|---|---|---|
| Parse | `DocumentParser` (emits **IR** now) | rtldoc / light / table-aware | other parsers | exists, re-target output |
| Chunk | **`Chunker`** (new) | structural (IR-derived) | fixed-window, semantic | new |
| Embed | `ModelProvider.embeddings` | OpenAI | any provider | exists |
| Vectors | `VectorStore` ABC | **Qdrant** (authoritative) | Milvus/pgvector/… | exists, make primary |
| Lexical | **`LexicalIndex`** (new, wraps 1b) | Neo4j Lucene + IDF scan on `search_text` | Qdrant hybrid (opt 2) | new wrapper over existing |
| Blob | `BlobStore` ABC | MinIO | S3/local | exists |
| Graph x1 | **`GraphAxisBuilder`** (new) | `Axis1StructuralBuilder` (deterministic) | — | new (extracted) |
| Graph x2 | **`GraphAxisBuilder`** | `Axis2IdeaBuilder` (entities/concepts) | different linking techniques | new (extracted from Axis-2) |
| Hydrate | **`Hydrator`** (new) | blob-fetch via `blob_key` | cache-backed | new |
| Retrieve | `UnstructuredStrategy` | existing strategies | new strategies | exists |

## 5. Data contracts

**Document IR** — implemented in `src/document/ir.py` (`Block`, `PageBlock`, `DocumentIR`). Storage-agnostic: no Neo4j/blob/vector imports.

Grounded finding that de-risks phase 1 significantly: **all three registered parser backends already split cleanly into exactly this boundary.** `LightPdfParser`, `RtldocPdfParser`, and `TableAwarePdfParser` each only override `_extract_pages()` (page/block extraction — this *is* the IR-producing phase); the graph-construction phase (`_usable_toc()` + `_build_from_toc()`/`_build_from_extracts()`, in `src/document/light/parser.py`) is **never overridden** — it lives in exactly one place, inherited unchanged by the other two backends. So there is only **one** graph-construction implementation to extract into `Axis1StructuralBuilder`, not three — the per-backend work in phase 1 is genuinely just "return `DocumentIR` instead of calling into the inherited graph-builder."

`Block` mirrors the existing (parser-private) `_PdfBlock` field-for-field, including `max_font_size`/`avg_font_size`/`bold` — the non-TOC heading heuristic that becomes part of `Axis1StructuralBuilder` reads these, so they can't be dropped in the promotion. `PageBlock` mirrors `_PageExtract`. `DocumentIR.toc` carries the PDF's own embedded outline (same usability gate as `_usable_toc`, resolved once at extraction time) so both `GraphConstructionService` and `scripts/validate_ontology_accuracy.py` read identical ground truth instead of each re-deriving it. `DocumentIR.finalize()` computes `content_hash` from extracted text — the idempotency key, computed one phase earlier than today's versioning.py use.

**Lean Neo4j node** (v2): `{id, type, order, page_start, page_end, depth, content_hash, blob_key, vector_id, entities, cluster_id, summary, tenant_id, …lineage, search_text}`.
- **Removed** from Neo4j: `text` (aggregated bodies → blob), `embedding` (→ vector store; drop `section_embedding` index).
- **Added**: `blob_key`, `vector_id`, `search_text` (chunk-sized, feeds Lucene + IDF lexical).

**Vector store payload:** `{doc_id, node_id, page, tags, tenant_id}` + embedding.

**Blob key scheme:** `{tenant_id}/{logical_id}/{revision_id}/{node_id}/text` (already used).

## 6. Lexical (decision = 1b)

Two lexical mechanisms exist today, both on `n.text`:
1. Neo4j Lucene full-text index `node_text_index` (`graph_seeds.fulltext_seed`).
2. IDF-weighted `CONTAINS` keyword scan (`LexicalService.structural_keyword_retrieve`).

**1b:** keep both, but run them over a lean **chunk-sized `search_text`** property instead of aggregated bodies. When the retrieval unit is chunk/Region-sized this is not truncation — it's the real unit; the heavy aggregated bodies (Chapter/Section/Document) go to blobs for hydration/display. Preserves tuned lexical behavior and the 95/101 eval pass rate; no rewrite of the two mechanisms mid-migration. Option 2 (lexical fully in the vector store, zero Neo4j text) deferred — at 100k the Neo4j index size won't bite.

## 7. The two hard parts

**A. Retrieval read-path (biggest cost).** Strategies read `n.text` directly today. Post-split they must **hydrate from blob via `blob_key`** through a single `Hydrator` seam — strategies never fetch blobs ad hoc. Touches the retrieval strategy surface incl. the lexical-scoping files. First-class phase 4.

**B. Lexical relocation.** Handled by 1b above.

## 8. Phasing (each independently shippable; regression + eval stay green)

1. **IR + parser refactor** — introduce Document IR; adapt 3 parsers; IR→legacy-node shim so downstream is unaffected. Behavior-neutral.
2. **Chunker + GraphConstructionService** — extract graph building out of parser/exporter into the service; split `Axis1StructuralBuilder` / `Axis2IdeaBuilder`; add `Chunker`. Still writing fat nodes. **Gate: ontology-accuracy report (§8a) ≥ 90% on both axes before phase 3 starts.**
3. **Storage split** — blob+vector authoritative; strip `text`/`embedding` from Neo4j writes; add `blob_key`/`vector_id`/`search_text`; drop `section_embedding` index. Introduce `LexicalIndex`. Re-run the ontology-accuracy report — the storage split must not move the score.
4. **Retrieval hydration** — `Hydrator`; rewrite read path to hydrate via pointers (part A). De-risks/blocks everything — gate here.
5. **Backfill + cutover** — migrate existing docs (idempotent via `content_hash` + versioning), dual-read during transition, then flip.

### 8a. Ontology-accuracy gate (≥ 90%, both axes)

"Ontology accuracy" is only a real gate if it's a measured number, not a claim. `src/document/ontology_validation.py` + `scripts/validate_ontology_accuracy.py` define and compute it today, ahead of any v2 code, so there's a **baseline from the current pipeline** to compare v2 against — the number must hold or improve, not just clear 90% in isolation.

**Axis-1 (structural) — scored two ways, picked automatically per document:**
- **TOC ground truth** (`score_axis1_against_toc`): when the source PDF carries a real embedded outline (`fitz.Document.get_toc()`, same usability gate as `light/parser.py`'s `_usable_toc` — ≥5 entries, at least one top-level entry), score = F1 of constructed Chapter/Section nodes against outline entries. A match requires: same tier (outline level 1 ↔ depth 1, level ≥2 ↔ depth ≥2), fuzzy title match (exact-after-normalization, or `difflib.SequenceMatcher` ratio ≥ 0.80 as fallback — looser than Axis-2's 0.92 entity-dedup threshold since outline↔node titles have more legitimate surface variance), and page match within ±1 (off-by-one tolerance). This is **real ground truth** — written by whatever tool produced the PDF — not a self-check.
- **Structural invariants** (`score_axis1_structural_invariants`): when no usable outline exists (e.g. HTML-sourced SEC filings carry none), score = fraction of pass/fail checks over the constructed tree — every child's page range nests inside its parent's, siblings don't have overlapping page ranges, no orphaned non-root nodes. Weaker signal (can't catch a whole document collapsed into one giant node, since that's internally "consistent") but catches the corruption classes already seen live (Item-7/7A prefix collisions, the physics-textbook broken-bookmark blowup).

**Axis-2 (idea-linking) — sampled LLM-judge precision** (`score_axis2_idea_linking`): no ground truth exists for semantic edges, so this uses the same class of measurement RAGAS-style faithfulness/context-precision metrics use industry-wide. For a random sample (default 25) of `SHARES_ENTITY`/`SAME_CATEGORY`/`SEMANTICALLY_SIMILAR` edges, an LLM judge scores whether the claimed shared concept genuinely connects the two passages; separately, a sample of extracted entities is checked for grounding (does the entity actually appear/paraphrase in its source text, catching hallucinated extraction). Fails an item **closed** (counts as invalid) on any judge error or malformed response — an accuracy *audit* must undercount rather than overcount, unlike the fail-open pattern used for a live user-facing answer in `retrieval/unstructured/verification.py`. Sampled, not exhaustive — cheap enough to run per phase-gate, not free enough to run on every ingest (see `feedback_eval_suite_cost`).

**Running it:**
```bash
python scripts/validate_ontology_accuracy.py --doc <logical_doc_id>
python scripts/validate_ontology_accuracy.py --all --sample-size 25
python scripts/validate_ontology_accuracy.py --all --axis1-only   # free, no LLM calls
```
Reports per-document Axis-1/Axis-2 scores + a corpus-average pass rate against the 90% target. Axis-1 is free (pure Cypher + local PDF parsing); Axis-2 costs one small LLM call per sampled edge/entity — use `--axis1-only`/`--axis2-only` to avoid re-paying for one axis when only checking the other.

### 8b. Baseline result (today's pipeline, 15 ingested documents, 2026-08-02)

```
corpus average: axis1=91.64%  axis2=56.45%  (axis1: 13/15 pass; axis2: 0/15 pass)
```

**Axis-1 (structural) is already close to the target** — 13/15 documents individually clear 90%, most in the high-90s/100%. Two genuine, isolated failures, not a systemic problem:
- `godata-annual-report-2021` (11.76%, TOC ground truth) — the constructed graph barely matches the PDF's own embedded outline. Real defect, worth its own investigation (separate from this redesign's scope).
- `table-aware-aapl-10k-2024` (67.86%, structural invariants) — cascading page-range nesting violations specific to the `table-aware` parser backend (e.g. a section's page range extends beyond its parent's). Real defect in that backend specifically, not `light`/`rtldoc`.

(First scan of this baseline showed axis1=81.56% with every document flagged for "orphan" Sections — that was a bug in the scoring script's own Cypher, not the pipeline: it fetched only Chapter/Section nodes, so a Section parented directly by the Document root (no intervening Chapter — common for SEC filings) had an unresolvable parent and was misread as orphaned. Fixed by including the Document root in the fetched set; see `_fetch_structural_nodes` in `scripts/validate_ontology_accuracy.py` and the regression test `test_axis1_invariants_section_parented_directly_by_document_not_orphan`.)

**Axis-2 (idea-linking) is the real, confirmed gap** — every single document fails, averaging 56%, and this number is unaffected by the axis-1 scoring bug above. The judge's `invalid_examples` show a consistent pattern across every document: `SHARES_ENTITY`/`SAME_CATEGORY` edges are built on **generic, low-information terms** — `"company"`, `"2023"`, `"u.s."`, `"chevron"`, `"apple inc."`, `"chief financial officer"`. These are technically true (both passages do contain the word), but not a meaningfully informative connection — two unrelated sections of a 10-K both mentioning "the Company" or the fiscal year shouldn't be linked as sharing an idea. This is the same *class* of problem already fixed once for lexical ranking (`repo_lexical_scoping_scaling_fix`'s IDF weighting: don't let generic terms count the same as distinctive ones) and for entity canonicalization (`repo_axis2_scaling_fix`'s near-duplicate merging) — but neither existing fix filters *generic* entities out of edge-building in the first place; they only dedupe/weight what's already there.

**Consequence for phase 2:** `Axis2IdeaBuilder` isn't just an extraction of the existing Axis-2 code — it needs an **informativeness filter** on which entities are allowed to anchor a `SHARES_ENTITY`/`SAME_CATEGORY` edge (e.g. IDF-style rarity within the document/corpus, excluding entity strings below a specificity threshold) before this clears 90%. Re-running `--axis1-only`/`--axis2-only` after that change, against this same 15-document baseline, is the acceptance check.

## 9. Scope discipline (what we are NOT doing)

- Not touching the structured Northwind path.
- Not adopting entity-centric/community GraphRAG as the primary retrieval model.
- No fixed `entity_types` ontology — keep discovered entities.
- No `neo4j-admin` bulk-import / sharding — unnecessary at 100k; `apoc.periodic.iterate` batching suffices.

## 10. Open questions

- Chunk granularity + overlap for the default `Chunker` (drives how lossless 1b's `search_text` is).
- Whether Axis-2 idea linking stays intra-document only, or gains cross-document idea edges (the reference project's strength) — deferred, but the IR/loader shape should not preclude it.
