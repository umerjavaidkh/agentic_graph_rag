"""
axis2.py — Semantic relationship discovery (Axis 2).

Builds:
    SEMANTICALLY_SIMILAR  — embedding cosine similarity
    SHARES_ENTITY         — shared NER entities, IDF-weighted by document
                             rarity (a shared entity mentioned in only 2
                             nodes counts far more than one mentioned in 40)
    SAME_CATEGORY         — KMeans over entity co-occurrence vectors when
                             enough nodes carry entities (a signal distinct
                             from SEMANTICALLY_SIMILAR's raw embeddings,
                             not a coarser copy of it); falls back to the
                             embeddings themselves when too few nodes have
                             entities for that signal to mean anything
    CONTRADICTS           — LLM reasoning pass (expensive, optional)
    ELABORATES            — LLM reasoning pass (expensive, optional)
    PREREQUISITE_OF       — LLM reasoning pass (expensive, optional)

Design principles:
  - Cheap relationships (SIMILAR, SHARES_ENTITY, SAME_CATEGORY) run always
  - Expensive LLM relationships run on top-k embedding-similar pairs PLUS a
    reserved slice of "entity-bridge" pairs (tied by a rare shared entity
    but not already embedding-similar) — pure similarity-ranked sampling
    misses cross-topic logical relationships (two sections that discuss the
    same named thing in different phrasing) that entity co-occurrence
    surfaces instead.
  - NER and LLM-pair calls are parallelised with bounded ThreadPoolExecutors
  - All relationships are Axis 2 flagged
"""
import difflib
import json
import itertools
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from ..config.settings import (
    AXIS2_GROUND_LLM_EDGES,
    AXIS2_LLM_PAIR_CONCURRENCY,
    AXIS2_MAX_LLM_PAIRS,
    AXIS2_MAX_SIMILARITY_EDGES_PER_NODE,
    AXIS2_MODEL,
    AXIS2_NER_BATCH_SIZE,
    AXIS2_NER_CONCURRENCY,
    AXIS2_NER_MAX_TOKENS,
    AXIS2_RELATION_MAX_TOKENS,
    EMBEDDING_MODEL,
)
from ..model_providers.factory import get_chat_provider, get_embedding_provider
from ..models import DKGNode, DKGEdge, EdgeConfidenceTier, NodeType, RelType


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD   = 0.75   # cosine sim for SEMANTICALLY_SIMILAR
CONTRADICTION_THRESH   = 0.85   # only run LLM on very similar pairs
# SAME_CATEGORY has no per-pair score (cluster co-membership alone doesn't
# confirm two specific members are strongly related) — AMBIGUOUS, flat score.
SAME_CATEGORY_CONFIDENCE = 0.5
# Node types to include in semantic analysis (skip PAGE for perf)
SEMANTIC_NODE_TYPES    = {NodeType.CHAPTER, NodeType.SECTION}
CONCEPT_NODE_TYPES     = {NodeType.SECTION, NodeType.PAGE}


def _cap_edges_by_degree(
    candidates: list[Tuple[int, int, float]], k: int
) -> list[Tuple[int, int, float]]:
    """
    Greedily select from `candidates` (deduped (i, j, score) triples, i<j)
    highest-score first, skipping any pair where either endpoint has
    already reached degree k. This is the only approach that actually
    bounds *every* node's final degree at k.

    An earlier version had each node independently pick its own top-k
    candidates, which does NOT bound degree: a node that happens to be
    "everyone's favorite neighbor" (e.g. near-duplicate/tied embeddings, or
    an entity that appears in most of the corpus) gets chosen by many other
    nodes' independent top-k picks regardless of its own choices — verified
    directly: 100 nodes all sharing 2 identical entities produced 20 nodes
    with degree 99 under the per-node-picks-its-own-top-k approach, because
    ties meant every node's "top 20" was a different arbitrary subset whose
    union covered nearly the whole graph. Global greedy selection with a
    live degree check on both endpoints closes that gap.
    """
    if k <= 0:
        return []
    candidates = sorted(candidates, key=lambda c: c[2], reverse=True)
    degree: dict[int, int] = {}
    out: list[Tuple[int, int, float]] = []
    for i, j, score in candidates:
        if degree.get(i, 0) >= k or degree.get(j, 0) >= k:
            continue
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
        out.append((i, j, score))
    return out


def _topk_edge_pairs(
    sim: np.ndarray, k: int, threshold: Optional[float] = None
) -> list[Tuple[int, int, float]]:
    """
    Bound edges to a per-node degree of k instead of every pair above a
    flat threshold — a plain threshold filter over a full similarity
    matrix is O(n^2) in edge count with no per-node cap at all, which is
    exactly what let a 7,165-node document produce 2.17M edges. `sim` must
    already have its diagonal excluded (e.g. set to -1) so a node never
    picks itself.
    """
    n = sim.shape[0]
    if n < 2 or k <= 0:
        return []
    iu, ju = np.triu_indices(n, k=1)
    scores = sim[iu, ju]
    if threshold is not None:
        mask = scores >= threshold
        iu, ju, scores = iu[mask], ju[mask], scores[mask]
    candidates = [(int(i), int(j), float(s)) for i, j, s in zip(iu, ju, scores)]
    return _cap_edges_by_degree(candidates, k)


_POSSESSIVE_RE = re.compile(r"[’']s\b")
_WS_RE_ENTITY = re.compile(r"\s+")
# Strips a trailing corporate-entity suffix so "Pfizer" and "Pfizer Inc."
# (or "Apple Inc.", "Chevron Corporation", ...) normalize to the same key
# and canonicalize together -- found live: entity-frequency analysis of an
# ingested Pfizer 10-Q showed "pfizer inc." (40.4% of entity-bearing
# nodes) and "pfizer" (19.2%) as two SEPARATE canonical entities, each
# individually staying just under the genericity-filter threshold despite
# jointly referring to the filer's own name in ~60% of nodes -- neither
# _canonicalize_entities' possessive-stripping nor its 0.92 fuzzy-ratio
# threshold catches this (a short root word plus a several-character
# suffix pushes the similarity ratio well below 0.92). General pattern
# (any all-caps-agnostic company-suffix word), not any specific company
# name, so it generalizes across the corpus rather than being tuned to
# this one filer.
_CORP_SUFFIX_RE = re.compile(
    r"\s*[,.]?\s*\b(?:incorporated|inc|corporation|corp|company|co|llc|ltd|"
    r"limited|plc|l\.?p\.?|group|holdings?)\.?\s*$"
)
_ENTITY_SIMILARITY_THRESHOLD = 0.92
_ENTITY_MAX_LEN_DIFF = 6
_ENTITY_PREFIX_LEN = 4
# Prefix bucketing alone doesn't bound worst-case cost: a vocabulary with
# many entities sharing a common prefix (repetitive technical terminology,
# e.g. a physics textbook's small set of core concepts recombined into many
# phrases) still lands most strings in a few large buckets, each still
# O(bucket_size^2). Measured directly: 3,500 unique entity strings under an
# adversarial 20-word vocabulary took ~2.5s; 10,000 took 23s+. Above this
# ceiling, skip the fuzzy-similarity pass and keep only the O(n)
# possessive-stripped exact-match normalization -- bounded cost at any
# corpus size, same spirit as this file's other degree/count caps.
_ENTITY_FUZZY_CLUSTER_MAX_VOCAB = 3000

# Deterministic substring check used by _is_entity_grounded -- entity text
# is normalized the same whitespace-collapsing way as canonicalization
# before comparing against the source text.
_WS_RE_GROUND = _WS_RE_ENTITY


def _is_entity_grounded(entity: str, source_text: str) -> bool:
    """A NER-extracted entity must actually occur in the text it was
    extracted from -- a deterministic substring check, not another model
    call, so it costs nothing and catches outright hallucination (an
    entity invented from the model's own training-data knowledge rather
    than read off the page) with certainty instead of another probabilistic
    estimate stacked on top of the first one.

    Corporate-suffix-stripped fallback reuses the exact normalization
    _canonicalize_entities applies for merging, so a legitimate surface
    variant ("Pfizer Inc." extracted from text that only spells out
    "Pfizer") isn't punished here for a difference canonicalization would
    merge away anyway -- grounding and canonicalization should agree on
    what counts as "the same entity", not apply two different notions of
    equivalence.
    """
    norm_entity = _WS_RE_GROUND.sub(" ", (entity or "")).strip().lower()
    norm_text = _WS_RE_GROUND.sub(" ", (source_text or "")).strip().lower()
    if not norm_entity:
        return False
    if norm_entity in norm_text:
        return True
    stripped = _CORP_SUFFIX_RE.sub("", norm_entity).strip()
    return bool(stripped) and stripped in norm_text


def _canonicalize_entities(
    all_entities: set[str], entity_types: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """
    Map each raw (already-lowercased) entity string to a single canonical
    form, so SHARES_ENTITY edge-building isn't fragmented by surface-text
    variance of the same real-world entity ("newton" vs "newton's" vs
    "kinematics"/"kinematic" would otherwise be separate dict keys in
    _build_entity_edges, none of which "share" an entity with each other).

    General string-similarity clustering (possessive-stripping +
    difflib.SequenceMatcher ratio), not any per-document vocabulary list —
    validated by threshold/length-gap tuning, not by hardcoding entity
    names, so it generalizes to any corpus.

    `entity_types` (optional): entity string -> NER type (e.g. "ORG",
    "CONCEPT"), when the caller has it. Entities are only ever clustered
    *within* the same type -- an identical surface string tagged with two
    different types ("Apple" as ORG in one node, "apple" as CONCEPT in
    another) is never merged, however similar the strings look, since
    conflating those is exactly the "fruit vs. company" failure mode
    typed entities exist to prevent. Entities missing from `entity_types`
    (or when the whole argument is omitted, e.g. from an older ingestion
    run before typed NER existed) fall into one shared "untyped" bucket,
    identical to this function's behavior before this parameter existed.

    Only compares strings within the same (type, first-N-character-prefix)
    bucket. A plain length-sorted scan is still O(k^2) in practice for
    natural-language vocabulary: word lengths cluster too tightly for a
    length gap to prune much (verified: 1,200 physics-textbook-scale
    entity strings took 5+ seconds, and this only gets worse per document
    since entity vocabulary grows with corpus size — the same class of
    blowup this file already fixes elsewhere via degree-capping).
    Near-duplicates (typos, possessive/plural variants) overwhelmingly
    share a prefix, so bucketing by it keeps same-cluster comparisons
    intact while cutting cross-bucket ones entirely — trades recall on
    prefix-diverging aliases (e.g. "isaac newton" vs "newton") for
    bounded, near-linear cost at any corpus size.
    """
    entity_types = entity_types or {}
    if len(all_entities) < 2:
        return {e: e for e in all_entities}

    # namespaced key = (type, normalized_base_text) -- type is part of
    # cluster identity from here on, so bucketing/fuzzy-matching below
    # naturally never crosses a type boundary (different types produce
    # different bucket keys even for the identical base text).
    namespaced_of: dict[str, tuple[str, str]] = {}
    for e in all_entities:
        key = _WS_RE_ENTITY.sub(" ", _POSSESSIVE_RE.sub("", e)).strip()
        key = _CORP_SUFFIX_RE.sub("", key).strip()
        base = key or e
        namespaced_of[e] = (entity_types.get(e, ""), base)

    keys = sorted(set(namespaced_of.values()))
    parent = {k: k for k in keys}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if len(keys) <= _ENTITY_FUZZY_CLUSTER_MAX_VOCAB:
        buckets: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for etype, base in keys:
            buckets.setdefault((etype, base[:_ENTITY_PREFIX_LEN]), []).append((etype, base))

        for bucket_keys in buckets.values():
            bucket_keys.sort(key=lambda k: len(k[1]))
            for i, a in enumerate(bucket_keys):
                for b in bucket_keys[i + 1:]:
                    if len(b[1]) - len(a[1]) > _ENTITY_MAX_LEN_DIFF:
                        break
                    if difflib.SequenceMatcher(None, a[1], b[1]).ratio() >= _ENTITY_SIMILARITY_THRESHOLD:
                        union(a, b)

    cluster_members: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for k in keys:
        cluster_members.setdefault(find(k), []).append(k)

    def _display(k: tuple[str, str]) -> str:
        # Canonical representative = the longest surface form in the
        # cluster (a fuller name is more informative to read back in
        # edge.properties than an abbreviation it absorbed). Type suffix
        # keeps two same-text-different-type canonical forms distinct as
        # dict values (both display as "apple" otherwise, which would
        # silently re-merge them the moment a caller groups edges by this
        # string) and doubles as a transparency win in shared_entities.
        etype, base = k
        return f"{base} ({etype})" if etype else base

    key_to_canonical: dict[tuple[str, str], str] = {
        member: _display(max(members, key=lambda k: len(k[1])))
        for members in cluster_members.values()
        for member in members
    }
    return {e: key_to_canonical[namespaced_of[e]] for e in all_entities}


# Below this many entity-bearing nodes, a document-frequency ratio isn't a
# meaningful signal (a term shared by 2 of 4 nodes is 50% "generic" purely
# by chance) -- skip the genericity filter entirely rather than let small
# documents' entities get spuriously excluded.
_ENTITY_GENERICITY_MIN_NODES = 5
# An entity mentioned in more than this fraction of a document's
# entity-bearing nodes doesn't distinguish one section from another within
# THIS document -- every node "sharing" it makes a SHARES_ENTITY edge built
# on it meaningless as a link, not just weak (verified live: sampled
# LLM-judge audits of ingested SEC filings flagged exactly this pattern --
# edges anchored on "company", "2023", the filer's own name, appearing in
# most sections of the filing). Same generic-vs-distinctive reasoning
# already used for lexical ranking (document_resolver.py's IDF-weighted
# term scoring, LexicalService's IDF keyword ranking) -- applied here to
# which entities may anchor an edge, not to retrieval scoring.
#
# The ratio itself is NOT flat -- it's the FLOOR of an adaptive curve (see
# _adaptive_genericity_ratio below). A flat 40% cutoff was tuned against a
# large, multi-topic SEC filing (hundreds of entity-bearing nodes), where a
# term above 40% document frequency really is generic filler ("the
# Company", the filer's own name) -- there's enough unrelated content
# diluting the corpus that a term touching almost everything can't be the
# real subject of any specific pair. Verified live on the opposite case: a
# short, single-topic tutorial (~20 entity-bearing nodes) has its own core
# subject ("sample mean", 70% document frequency) hit the SAME flat cutoff
# and get excluded from anchoring any edge -- the summary section
# discussing that subject then failed to connect to the sections it was
# actually summarizing. On a small corpus there's much less unrelated
# content to dilute a real subject's frequency, so high document frequency
# is far weaker evidence of genericity than on a large one.
_ENTITY_GENERICITY_DF_RATIO = 0.40  # floor: converges here as corpus size grows
_ENTITY_GENERICITY_DF_RATIO_CEILING = 0.85  # cap: small-corpus permissiveness
# Controls how fast the ratio decays from the ceiling toward the floor as
# total_entity_nodes grows -- chosen so a ~20-node corpus (this file's
# motivating case) stays comfortably permissive (~75%, safely above the
# 70% document frequency that triggered this).
_ENTITY_GENERICITY_DECAY_NODES = 80
# An exponential decay asymptotically APPROACHES the floor but never
# exactly reaches it -- at real SEC-filing scale (~222 entity-bearing
# nodes) it was still sitting at ~42.8%, a permissive excess of ~3 points
# over the validated 40%. Verified live: "Chevron" (the filer's own name)
# landed at 95/222 = 42.8% document frequency in one ingestion run --
# comfortably above the intended 40% floor, but just inside this excess
# margin, so it anchored 683 of 1,833 SHARES_ENTITY edges (37.3%) despite
# canonicalization correctly merging all its spelling variants into one
# key (confirmed: this was NOT the type-fragmentation bug fixed above --
# the curve itself just hadn't fully converged). At and beyond this many
# entity-bearing nodes, use the exact validated floor with no residual
# excess, rather than only asymptotically approaching it -- this is where
# the original flat-40% threshold was actually measured and trusted, so
# it should be applied exactly, not approximately.
_ENTITY_GENERICITY_LARGE_CORPUS_NODES = 150

# Fixed, small taxonomy -- enough to separate "Apple the company" from
# "apple the fruit"-shaped ambiguity (ORG vs CONCEPT/PRODUCT) without asking
# the model to invent an open-ended type vocabulary. Module-level (not just
# an Axis2Builder attribute) so _entity_base_text below can recognize a
# type suffix without importing the class.
_ENTITY_TYPES = ("PERSON", "ORG", "LOCATION", "PRODUCT", "CONCEPT", "METRIC", "DATE", "OTHER")


def _entity_base_text(entity_key: str) -> str:
    """Strips a trailing " (TYPE)" suffix added by _resolve_canonical_entities's
    _display(), so genericity can be judged on the entity's real-world
    identity's TOTAL document frequency across every type variant it was
    tagged with -- not on one type-suffixed key's own count in isolation.

    Necessary because the LLM's type tag for the same entity isn't
    perfectly consistent across separate NER batch calls (each covering a
    handful of nodes) -- a large document runs many such calls. Verified
    live on a real 264-page 10-K: "Chevron" (the filer's own name, 57% raw
    document frequency) was tagged ORG in most batches but not all,
    fragmenting its true frequency across type buckets so that its
    dominant "Chevron (ORG)" bucket landed at exactly 95/223 nodes --
    just under the adaptive threshold's cutoff of 95 -- while the real,
    combined entity anchored 716 of 1,846 SHARES_ENTITY edges (38.8%) in
    the ingested graph. Type-awareness must only affect which mentions are
    treated as the same real-world entity (Apple ORG vs apple CONCEPT
    genuinely are different things and must stay separate) -- it must
    never be allowed to affect how OFTEN that entity is judged to appear,
    which is what genericity filtering depends on.
    """
    if entity_key.endswith(")") and " (" in entity_key:
        base, _, rest = entity_key.rpartition(" (")
        if rest[:-1] in _ENTITY_TYPES:
            return base
    return entity_key


def _entity_type(entity_key: str) -> Optional[str]:
    """Inverse of _entity_base_text: the tagged TYPE, or None if untyped/legacy."""
    if entity_key.endswith(")") and " (" in entity_key:
        _, _, rest = entity_key.rpartition(" (")
        type_str = rest[:-1]
        if type_str in _ENTITY_TYPES:
            return type_str
    return None


# DATE entities pass the document-frequency genericity filter by raw count
# (they're rarely 40%+ of a document's nodes) but are structurally
# meaningless as a topical-similarity anchor regardless of frequency: two
# pages both mentioning "2025" says nothing about whether their content is
# related, in any document. Verified live: DATE-anchored SHARES_ENTITY edges
# were the dominant failure mode in a 10-K's sampled ontology score (all 10
# flagged edges DATE-anchored, score 50%) even after the document-frequency
# fixes above resolved the entity-name (self-referential-term) failure mode.
# Deliberately scoped to DATE only, not LOCATION -- unlike a bare date, many
# document types (supply-chain, regulatory, geological) rely on genuine
# location-based relations, so excluding the whole type would be overreach
# rather than a root-cause fix.
_NON_TOPICAL_ENTITY_TYPES = frozenset({"DATE"})


def _adaptive_genericity_ratio(total_entity_nodes: int) -> float:
    """Document-frequency cutoff above which an entity is "too generic" to
    anchor an edge, scaled by corpus size: permissive (near the ceiling)
    for a small corpus, decaying toward the floor as the corpus grows.
    Exponential decay, not linear -- most of the drop happens over the
    first ~1-2 decay constants (small/medium documents, where the true
    corpus-size-vs-genericity relationship is least certain and this
    function's shape matters most).

    At and beyond _ENTITY_GENERICITY_LARGE_CORPUS_NODES, returns the exact
    floor rather than the exponential's asymptotic approximation of it --
    an exponential decay gets arbitrarily close to the floor but never
    exactly reaches it, and at real SEC-filing scale that residual excess
    (~3 points, verified live) was enough for a document's own dominant
    self-referential term to slip through. The original flat 40% was
    validated at that scale, so it's applied exactly there, not
    approximately.
    """
    if total_entity_nodes >= _ENTITY_GENERICITY_LARGE_CORPUS_NODES:
        return _ENTITY_GENERICITY_DF_RATIO
    decay = math.exp(-total_entity_nodes / _ENTITY_GENERICITY_DECAY_NODES)
    return _ENTITY_GENERICITY_DF_RATIO + (_ENTITY_GENERICITY_DF_RATIO_CEILING - _ENTITY_GENERICITY_DF_RATIO) * decay


def _informative_entities(entity_to_nodes: dict[str, list[int]], total_entity_nodes: int) -> set[str]:
    """Entities NOT too generic (by document-frequency ratio) to anchor a
    SHARES_ENTITY edge within this document. See constants above for why.

    Document frequency is computed per BASE TEXT (see _entity_base_text),
    aggregating node coverage across every type-suffixed variant that base
    text appears under, not per type-suffixed key in isolation -- a type
    tag is occasionally inconsistent across separate NER batch calls, and
    judging genericity on the fragmented per-type count instead of the
    real combined one lets a genuinely dominant/generic entity slip
    through underneath the cutoff in whichever type bucket happens to
    hold most of its mentions.

    DATE entities are excluded unconditionally (see _NON_TOPICAL_ENTITY_TYPES)
    regardless of corpus size or frequency -- that's a type judgment, not a
    frequency one, so it applies even below the min-nodes bypass below.
    """
    topical = {e for e in entity_to_nodes if _entity_type(e) not in _NON_TOPICAL_ENTITY_TYPES}

    if total_entity_nodes < _ENTITY_GENERICITY_MIN_NODES:
        return topical
    ratio = _adaptive_genericity_ratio(total_entity_nodes)
    max_df = max(1, int(total_entity_nodes * ratio))

    base_nodes: dict[str, set[int]] = {}
    for e, idxs in entity_to_nodes.items():
        base_nodes.setdefault(_entity_base_text(e), set()).update(idxs)

    return {e for e in topical if len(base_nodes[_entity_base_text(e)]) <= max_df}


def _resolve_canonical_entities(nodes: list[DKGNode]) -> dict[Tuple[str, str], str]:
    """Canonical entity text for each (node_id, lowercased_entity_text)
    occurrence, resolved using THAT occurrence's own NER type
    (node.entity_types) -- not a single corpus-wide type per string, which
    cannot represent "the same spelling means two different things in two
    different nodes" at all (a plain text->type dict has exactly one type
    per key; a real conflict would just silently pick whichever node's
    type was written last, `_canonicalize_entities` would then place the
    other node's mentions under the wrong type, and the whole point of
    typing entities -- catching "Apple" the ORG merging with "apple" the
    CONCEPT -- would quietly not work).

    Fixed by splitting the corpus's entity vocabulary into one bucket per
    type actually seen (untyped mentions share one "" bucket, same as
    before per-occurrence typing existed) and canonicalizing each bucket
    with a separate, independent call to _canonicalize_entities. Because
    each call only ever sees one type's vocabulary, there is no code path
    by which an ORG-typed and a CONCEPT-typed mention of the identical
    string can be compared to each other, let alone merged -- not "less
    likely to merge", structurally incapable of it.
    """
    by_type: dict[str, set[str]] = {}
    for node in nodes:
        for e in node.entities:
            text = e.lower()
            etype = (node.entity_types or {}).get(text, "")
            by_type.setdefault(etype, set()).add(text)

    canonical_by_type = {etype: _canonicalize_entities(texts) for etype, texts in by_type.items()}

    result: dict[Tuple[str, str], str] = {}
    for node in nodes:
        for e in node.entities:
            text = e.lower()
            etype = (node.entity_types or {}).get(text, "")
            base = canonical_by_type[etype][text]
            # Type suffix keeps two same-text-different-type canonical
            # forms distinct as dict values -- both would otherwise
            # display as the same bare string, which would silently
            # re-merge them the moment a caller groups edges by it.
            result[(node.id, text)] = f"{base} ({etype})" if etype else base
    return result


# ─────────────────────────────────────────
# AXIS 2 BUILDER
# ─────────────────────────────────────────
class Axis2Builder:
    """
    Takes the node list from document ingestion and enriches it with
    all Axis 2 semantic edges.

    Usage:
        builder = Axis2Builder()
        nodes, new_edges = builder.build(nodes)
    """

    def __init__(self):
        self.client = get_chat_provider()
        # Embeddings always go through OpenAI regardless of MODEL_PROVIDER —
        # see model_providers.factory.get_embedding_provider().
        self.embedding_client = get_embedding_provider()

    def build(
        self,
        nodes: list[DKGNode],
        run_llm_pass: bool = False,  # set True only when you want CONTRADICTS/ELABORATES
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        """
        Returns updated nodes (with embeddings + entities) and new Axis 2 edges.
        """
        edges: list[DKGEdge] = []

        # 1. Embed nodes
        nodes = self._embed_nodes(nodes)

        # 2. NER — extract entities per node (parallel)
        nodes = self._extract_entities(nodes)

        # 3. SEMANTICALLY_SIMILAR
        edges += self._build_similarity_edges(nodes)

        # 4. SHARES_ENTITY
        entity_edges = self._build_entity_edges(nodes)
        edges += entity_edges

        # 5. SAME_CATEGORY (clustering)
        nodes, edges_cat = self._build_category_edges(nodes)
        edges += edges_cat

        # 6. LLM pass — CONTRADICTS / ELABORATES / PREREQUISITE_OF (parallel).
        # entity_edges is passed through so the candidate sampler can draw
        # part of its budget from entity-bridge pairs, not just top-cosine
        # ones (see module docstring).
        if run_llm_pass and self.client:
            edges += self._build_llm_edges(nodes, entity_edges=entity_edges)

        return nodes, edges

    # ─────────────────────────────────────────
    # 1. EMBEDDINGS
    # ─────────────────────────────────────────
    def _embed_nodes(self, nodes: list[DKGNode]) -> list[DKGNode]:
        targets = [n for n in nodes if n.type in SEMANTIC_NODE_TYPES]
        if not targets or not self.client:
            return nodes

        texts = [f"{n.title}\n\n{n.text[:2000]}" for n in targets]
        # Batch in groups of 100 (OpenAI limit)
        for batch_start in range(0, len(texts), 100):
            batch = texts[batch_start:batch_start + 100]
            response = self.embedding_client.embeddings(
                model=EMBEDDING_MODEL, input=batch
            )
            for i, emb_obj in enumerate(response.data):
                targets[batch_start + i].embedding = emb_obj.embedding

        return nodes

    # ─────────────────────────────────────────
    # 2. ENTITY EXTRACTION — parallel NER
    # ─────────────────────────────────────────
    # See module-level _ENTITY_TYPES for why this is a fixed, small
    # taxonomy rather than an open-ended one the model invents per call.
    _ENTITY_TYPES = _ENTITY_TYPES

    _NER_CHUNK_CHARS = 1200
    # Each chunk's own NER call already asks for its top 10; a node spanning
    # several chunks can validly surface more real entities than one call
    # would -- capped here (not just "10 per call, unbounded per node") so a
    # very long node can't produce an unbounded entity list.
    _NER_MAX_ENTITIES_PER_NODE = 30

    @staticmethod
    def _chunk_node_text(text: str, size: int) -> list[str]:
        """Split `text` into <=`size`-char pieces, breaking on whitespace
        where possible so a word isn't split mid-token. Whole-text passthrough
        (a single chunk) when text already fits."""
        if len(text) <= size:
            return [text]
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            if end < n:
                split_at = text.rfind(" ", start, end)
                if split_at > start:
                    end = split_at
            chunks.append(text[start:end])
            start = end
        return chunks

    def _extract_entities(self, nodes: list[DKGNode]) -> list[DKGNode]:
        """
        Uses LLM for NER in parallel (bounded by AXIS2_NER_CONCURRENCY),
        batching AXIS2_NER_BATCH_SIZE nodes per call instead of one call per
        node — a large document has thousands of Section/Page nodes, and
        one-call-per-node burns API request quota proportional to node
        count with no way to bound it. Returns top-10 entities per node to
        keep it manageable.

        Two additional layers beyond the raw LLM response:
          - grounding: an entity that doesn't actually occur (verbatim or
            corp-suffix-normalized) in the excerpt it was supposedly found
            in is dropped -- a deterministic check, not another model call,
            so it catches outright hallucination for free.
          - typing: each entity is tagged with a coarse type (see
            _ENTITY_TYPES), stored on node.entity_types (additive, keyed by
            lowercased entity text) so downstream canonicalization can tell
            "Apple" the ORG apart from "apple" the CONCEPT instead of
            merging same-looking strings from different real-world things.

        Long node text is split into _NER_CHUNK_CHARS-sized chunks rather
        than truncated to one -- verified live on a real 264-page 10-K: a
        4,074-char page lost every entity past its first ~1,200 chars (OPEC,
        Russia, "Chevron's Strategic Direction") under a flat per-node
        truncation, since only that first slice was ever sent to the model.
        A node's chunks can land in different batches (processed and merged
        independently via as_completed), so results are merged additively
        below, not assigned -- an assignment would let a later-completing
        batch silently overwrite an earlier chunk's entities for the same
        node.
        """
        if not self.client:
            return nodes

        targets = [n for n in nodes if n.type in CONCEPT_NODE_TYPES]
        if not targets:
            return nodes

        batch_size = max(1, AXIS2_NER_BATCH_SIZE)
        type_choices = " | ".join(self._ENTITY_TYPES)

        units: list[tuple[DKGNode, str]] = [
            (node, chunk)
            for node in targets
            for chunk in self._chunk_node_text(node.text, self._NER_CHUNK_CHARS)
        ]
        batches = [units[i:i + batch_size] for i in range(0, len(units), batch_size)]

        def _ner_batch(batch: list[tuple[DKGNode, str]]) -> tuple[dict[str, list], dict[str, dict[str, str]]]:
            # Excerpts are keyed by a short local index ("0", "1", ...), not
            # the node's own (long) id -- cheaper in tokens and avoids the
            # model mangling a complex id string as a JSON key. Mapped back
            # to node.id below, after parsing.
            parts = [f"[{i}]\n{chunk}" for i, (_, chunk) in enumerate(batch)]
            user_content = "\n\n---\n\n".join(parts)
            try:
                resp = self.client.chat_completion(
                    model=AXIS2_MODEL,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You will be given several numbered text excerpts, "
                                "each marked [N]. For EACH excerpt, extract its top "
                                "10 named entities (people, organizations, concepts, "
                                "theories, technical terms) THAT LITERALLY APPEAR IN "
                                "THE TEXT -- do not use outside knowledge to add "
                                f"entities not present in the excerpt. Type each "
                                f"entity as one of: {type_choices}. Return ONLY a "
                                "JSON object mapping each excerpt's number (as a "
                                'string) to an array of {"text": ..., "type": ...} '
                                'objects, e.g. {"0": [{"text": "Isaac Newton", '
                                '"type": "PERSON"}], "1": [...]}. Include every '
                                "excerpt number, even if its array is empty. Respond "
                                "with COMPACT JSON only -- no extra whitespace, "
                                "newlines, or indentation between entries (pretty-"
                                "printed JSON for a 10-entity list can run 3-4x more "
                                "tokens than compact, which was silently truncating "
                                "responses mid-object under the token budget below). "
                                "No explanation."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=AXIS2_NER_MAX_TOKENS * len(batch),
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
            except Exception:
                # No fixed max_tokens budget is safe against an arbitrary
                # combination of entity-dense chunks landing in the same
                # batch (verified live: raising the per-unit budget just
                # moved the failure to a slightly larger combination) --
                # was silent before too, and a truncated/malformed response
                # here zeroed out entities for every node/chunk in the
                # batch, indistinguishable from "the model found nothing."
                # Self-healing instead of a bigger constant: split the
                # batch in half and retry each half independently, so only
                # the actually-too-dense sub-batch keeps splitting, down to
                # single units in the worst case, rather than losing an
                # entire batch's entities to one overflow.
                if len(batch) > 1:
                    logger.warning(
                        "axis2 NER batch failed to parse (%d units); retrying as two smaller batches.",
                        len(batch),
                        exc_info=True,
                    )
                    mid = len(batch) // 2
                    left_entities, left_types = _ner_batch(batch[:mid])
                    right_entities, right_types = _ner_batch(batch[mid:])
                    entities_result: dict[str, list] = dict(left_entities)
                    for node_id, texts in right_entities.items():
                        entities_result.setdefault(node_id, []).extend(texts)
                    types_result: dict[str, dict[str, str]] = dict(left_types)
                    for node_id, etypes in right_types.items():
                        types_result.setdefault(node_id, {}).update(etypes)
                    return entities_result, types_result
                logger.warning(
                    "axis2 NER unit failed to parse even alone (node=%s); leaving its entities empty.",
                    batch[0][0].id,
                    exc_info=True,
                )
                parsed = {}

            # Additive (setdefault+extend/update), not assignment: a node
            # with multiple chunks can have more than one entry in this same
            # batch, and results from other batches for the same node.id get
            # merged in by the caller too.
            entities_result: dict[str, list] = {}
            types_result: dict[str, dict[str, str]] = {}
            for i, (node, chunk) in enumerate(batch):
                raw_items = parsed.get(str(i)) if isinstance(parsed, dict) else None
                raw_items = raw_items if isinstance(raw_items, list) else []

                texts: list[str] = []
                node_types: dict[str, str] = {}
                for item in raw_items:
                    if isinstance(item, dict):
                        text, etype = item.get("text"), item.get("type")
                    elif isinstance(item, str):
                        # Tolerate a model that ignores the typed-format
                        # instruction and returns a flat string -- still
                        # grounded and kept, just without a type.
                        text, etype = item, None
                    else:
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if not _is_entity_grounded(text, node.text):
                        continue
                    texts.append(text)
                    if isinstance(etype, str) and etype.strip():
                        node_types[text.lower()] = etype.strip().upper()

                entities_result.setdefault(node.id, []).extend(texts)
                types_result.setdefault(node.id, {}).update(node_types)
            return entities_result, types_result

        id_to_node = {n.id: n for n in targets}
        merged_entities: dict[str, list[str]] = {}
        merged_types: dict[str, dict[str, str]] = {}

        with ThreadPoolExecutor(max_workers=AXIS2_NER_CONCURRENCY, thread_name_prefix="axis2_ner") as pool:
            futures = [pool.submit(_ner_batch, batch) for batch in batches]
            for fut in as_completed(futures):
                try:
                    batch_entities, batch_types = fut.result()
                except Exception:
                    continue
                for node_id, entities in batch_entities.items():
                    merged_entities.setdefault(node_id, []).extend(entities)
                for node_id, etypes in batch_types.items():
                    merged_types.setdefault(node_id, {}).update(etypes)

        for node_id, entities in merged_entities.items():
            node = id_to_node.get(node_id)
            if node is None:
                continue
            seen: set[str] = set()
            deduped: list[str] = []
            for e in entities:
                key = e.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(e)
            node.entities = deduped[: self._NER_MAX_ENTITIES_PER_NODE]
        for node_id, etypes in merged_types.items():
            node = id_to_node.get(node_id)
            if node is not None:
                node.entity_types = etypes

        return nodes

    # ─────────────────────────────────────────
    # 3. SEMANTICALLY_SIMILAR
    # ─────────────────────────────────────────
    def _build_similarity_edges(self, nodes: list[DKGNode]) -> list[DKGEdge]:
        embedded = [n for n in nodes if n.embedding is not None]
        edges: list[DKGEdge] = []

        if len(embedded) < 2:
            return edges

        vecs  = np.array([n.embedding for n in embedded], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = vecs / (norms + 1e-10)
        sim   = vecs @ vecs.T  # cosine similarity matrix
        np.fill_diagonal(sim, -1.0)  # a node is never its own neighbor

        for i, j, score in _topk_edge_pairs(
            sim, AXIS2_MAX_SIMILARITY_EDGES_PER_NODE, SIMILARITY_THRESHOLD
        ):
            a, b = embedded[i], embedded[j]
            edges.append(DKGEdge(
                source_id  = a.id,
                target_id  = b.id,
                rel_type   = RelType.SEMANTICALLY_SIMILAR,
                weight     = round(score, 4),
                axis       = 2,
                properties = {"score": round(score, 4)},
                confidence = round(score, 4),
                confidence_tier = EdgeConfidenceTier.INFERRED,
            ))

        return edges

    # ─────────────────────────────────────────
    # 4. SHARES_ENTITY
    # ─────────────────────────────────────────
    def _build_entity_edges(self, nodes: list[DKGNode]) -> list[DKGEdge]:
        """
        Same unbounded-pairwise bug as SEMANTICALLY_SIMILAR had (every pair
        with ANY shared entity got an edge, no per-node cap) — a document
        whose entities repeat often (e.g. "Newton", "force", "energy"
        throughout a physics textbook) hits the identical O(n^2) blowup.
        Fixed the same way: a global degree-capped greedy selection (see
        _cap_edges_by_degree) over candidate pairs ranked by shared-entity
        count. Uses an inverted index (entity -> node indices) to build
        candidates rather than scanning every pair, so nodes that share
        nothing are never compared at all — cheaper AND bounded, instead of
        just bounded.
        """
        edges: list[DKGEdge] = []
        entity_nodes = [n for n in nodes if n.entities]
        if len(entity_nodes) < 2:
            return edges

        # Canonicalize surface-text variants of the same entity ("newton"
        # vs "newton's laws" vs "sir isaac newton") before grouping, so they
        # count as one shared entity instead of silently never matching each
        # other. Resolved per-occurrence (see _resolve_canonical_entities)
        # so a string tagged with two different types across nodes ("Apple"
        # ORG vs "apple" CONCEPT) is never merged just because the spelling
        # matches.
        canonical = _resolve_canonical_entities(entity_nodes)

        entity_to_nodes: dict[str, list[int]] = {}
        node_entity_sets: list[set] = []
        for idx, node in enumerate(entity_nodes):
            ents = {canonical[(node.id, e.lower())] for e in node.entities}
            node_entity_sets.append(ents)
            for e in ents:
                entity_to_nodes.setdefault(e, []).append(idx)

        # Entities mentioned in most of this document's entity-bearing
        # nodes ("the Company", the filer's own name, a bare fiscal year)
        # don't distinguish one section from another -- excluded from
        # anchoring an edge (and from the reported shared_entities) so a
        # SHARES_ENTITY link means something more specific than "both
        # sections are part of the same filing". See _informative_entities.
        informative = _informative_entities(entity_to_nodes, len(entity_nodes))

        # IDF-style rarity weight per entity, not a flat "1 per shared
        # entity" count: two sections sharing an entity mentioned in only 2
        # nodes document-wide is a far stronger signal than sharing one
        # mentioned in 40, even after the genericity filter above has
        # already dropped the most extreme (>40%-of-nodes) offenders --
        # among the entities that DO pass that filter, rarity still varies
        # a lot and should be reflected in edge weight, not just presence.
        # Smoothed idf(e) = log((N+1)/(freq(e)+1)) + 1 is always >= 1 (a
        # single shared entity never drags an edge below the previous
        # flat-count floor), rising as the entity gets rarer.
        idf = {
            e: float(np.log((len(entity_nodes) + 1) / (len(idxs) + 1)) + 1)
            for e, idxs in entity_to_nodes.items()
            if e in informative
        }

        pair_scores: dict[Tuple[int, int], float] = {}
        for entity, idx_list in entity_to_nodes.items():
            if entity not in informative:
                continue
            weight = idf[entity]
            for a, b in itertools.combinations(sorted(idx_list), 2):
                pair_scores[(a, b)] = pair_scores.get((a, b), 0.0) + weight

        candidates = [(i, j, score) for (i, j), score in pair_scores.items()]
        cap = AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
        for i, j, score in _cap_edges_by_degree(candidates, cap):
            a, b = entity_nodes[i], entity_nodes[j]
            shared = (node_entity_sets[i] & node_entity_sets[j]) & informative
            if not shared:
                continue
            edges.append(DKGEdge(
                source_id  = a.id,
                target_id  = b.id,
                rel_type   = RelType.SHARES_ENTITY,
                weight     = round(score, 4),
                axis       = 2,
                properties = {"shared_entities": list(shared), "rarity_score": round(score, 4)},
            ))

        return edges

    # ─────────────────────────────────────────
    # 5. SAME_CATEGORY (KMeans)
    # ─────────────────────────────────────────
    def _build_category_edges(
        self, nodes: list[DKGNode]
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        import hdbscan

        edges: list[DKGEdge] = []

        # Cluster on entity co-occurrence when enough nodes carry entities,
        # not the same raw embeddings SEMANTICALLY_SIMILAR already uses --
        # clustering the identical vector space just produces a coarser
        # copy of that edge type (same geometry, fewer/blurrier edges),
        # not an independent signal. Entity co-occurrence is a genuinely
        # different axis: two sections can be embedding-dissimilar (very
        # different phrasing/length) yet co-cluster here because they both
        # revolve around the same named things. Falls back to embeddings
        # when too few nodes have entities for a co-occurrence signal to
        # mean anything (small documents, or a chat provider unavailable
        # so NER never ran) -- same fallback shape as before this change,
        # not a behavior regression for that case.
        #
        # Canonicalized the same way _build_entity_edges is (_canonicalize_
        # entities), so "Newton"/"newton's"/"Sir Isaac Newton" count as one
        # clustering dimension instead of three separate near-duplicate
        # ones. Deliberately NOT filtered through _build_entity_edges's
        # _informative_entities genericity cutoff, though -- that threshold
        # is calibrated for PAIRWISE edge-anchoring (a term touching >40% of
        # nodes makes a poor excuse to link any two of them), not for
        # clustering, where a term split roughly 50/50 across the corpus is
        # exactly the kind of feature that makes two real clusters
        # separable. Reusing that cutoff here was tried and breaks that
        # legitimate case (verified via
        # test_same_category_uses_entity_signal_when_entities_available).
        target_nodes = [n for n in nodes if n.entities]
        canonical = _resolve_canonical_entities(target_nodes)
        vocab = sorted({canonical[(n.id, e.lower())] for n in target_nodes for e in n.entities})
        signal = "entity_cooccurrence"
        if len(target_nodes) < 3 or len(vocab) < 2:
            target_nodes = [n for n in nodes if n.embedding is not None]
            signal = "embedding"

        if len(target_nodes) < 3:
            return nodes, edges

        if signal == "entity_cooccurrence":
            vocab_index = {term: i for i, term in enumerate(vocab)}
            raw = np.zeros((len(target_nodes), len(vocab)), dtype=np.float32)
            doc_freq = np.zeros(len(vocab), dtype=np.float32)
            for row, node in enumerate(target_nodes):
                touched = {vocab_index[canonical[(node.id, e.lower())]] for e in node.entities}
                for idx in touched:
                    raw[row, idx] += 1
                    doc_freq[idx] += 1
            idf = np.log((len(target_nodes) + 1) / (doc_freq + 1)) + 1
            vecs = raw * idf

            # Drop nodes whose entity vector is negligible relative to the
            # rest of the corpus, instead of letting them into clustering at
            # all -- verified live on a real SEC filing: a cover-page node
            # whose only entities were boilerplate ("United States",
            # "Securities and Exchange Commission", both near-100% document
            # frequency, so idf collapses toward its floor of 1) ended up
            # with a near-zero raw*idf vector. `vecs / (norms + 1e-10)`
            # below turns a near-zero vector into an unstable, arbitrary-
            # direction unit vector, which then scored spuriously high
            # cosine similarity against unrelated sections -- one low-
            # content node fanning out into dozens of bad SAME_CATEGORY
            # edges. A RELATIVE floor (fraction of the corpus median), not
            # an absolute one, since a node's total vector magnitude scales
            # with how many distinct entities it has and how rare they are,
            # which varies a lot document to document.
            norms_pre = np.linalg.norm(vecs, axis=1)
            nonzero = norms_pre[norms_pre > 0]
            median_norm = float(np.median(nonzero)) if nonzero.size else 0.0
            keep_mask = norms_pre >= (median_norm * 0.1) if median_norm > 0 else norms_pre > 0
            if not bool(np.all(keep_mask)):
                target_nodes = [n for n, keep in zip(target_nodes, keep_mask) if keep]
                vecs = vecs[keep_mask]
                if len(target_nodes) < 3:
                    return nodes, edges
        else:
            vecs = np.array([n.embedding for n in target_nodes], dtype=np.float32)

        # Density-based, not a fixed k: a fixed cluster COUNT (the previous
        # KMeans approach) forces every node into some cluster regardless
        # of whether the data actually separates that way. Verified live
        # on a real ~20-entity-bearing-node document: KMeans dumped 19 of
        # them into one dominant cluster (78 of the resulting 84
        # SAME_CATEGORY edges), and the sampled LLM-judge audit then
        # flagged exactly those edges as "not meaningfully connected" --
        # measured evidence, not a hypothetical, that forcing weakly-
        # related nodes together produces edges the rest of the pipeline
        # already independently identifies as low-quality. HDBSCAN leaves
        # a node with no real category home labeled -1 ("noise") instead
        # of nearest-clustering it anyway, so a genuine gap in the data
        # produces no edge rather than a bad one. Euclidean distance over
        # L2-normalized vectors is monotonic with cosine distance, so
        # clustering runs on the same geometry the top-k intra-cluster
        # step below already uses (computed once, here, and reused).
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit_vecs = vecs / (norms + 1e-10)
        min_cluster_size = max(2, min(5, len(target_nodes) // 4))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(unit_vecs)

        for node, label in zip(target_nodes, labels):
            node.cluster_id = int(label)

        # Same degree-capping reasoning as SEMANTICALLY_SIMILAR: connecting
        # every pair within a cluster grows quadratically with cluster
        # size (this is what produced 2.17M edges for a 7,165-node
        # document under the old fixed-cluster-count approach -- capped
        # cluster count meant cluster SIZE grew with corpus size instead
        # of staying flat). Cap each node to its top-k most-similar
        # members of its OWN cluster, on a per-cluster similarity
        # submatrix (cheap -- clusters are far smaller than the full
        # corpus). label -1 (HDBSCAN's "noise", not a real cluster) is
        # excluded entirely -- those nodes get no SAME_CATEGORY edges,
        # which is the correct outcome for a node with no confident
        # category, not a gap to paper over.
        clusters: dict[int, list[int]] = {}  # cluster_id -> indices into target_nodes
        for idx, node in enumerate(target_nodes):
            if node.cluster_id == -1:
                continue
            clusters.setdefault(node.cluster_id, []).append(idx)

        cap = AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
        seen: set[Tuple[int, int]] = set()

        for cluster_id, member_idx in clusters.items():
            if len(member_idx) < 2:
                continue
            sub = unit_vecs[member_idx]
            sub_sim = sub @ sub.T
            np.fill_diagonal(sub_sim, -1.0)
            for li, lj, _score in _topk_edge_pairs(sub_sim, cap):
                gi, gj = member_idx[li], member_idx[lj]
                pair = (gi, gj) if gi < gj else (gj, gi)
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = target_nodes[pair[0]], target_nodes[pair[1]]
                edges.append(DKGEdge(
                    source_id  = a.id,
                    target_id  = b.id,
                    rel_type   = RelType.SAME_CATEGORY,
                    axis       = 2,
                    properties = {"cluster_id": cluster_id, "signal": signal},
                    confidence = SAME_CATEGORY_CONFIDENCE,
                    confidence_tier = EdgeConfidenceTier.AMBIGUOUS,
                ))

        return nodes, edges

    # ─────────────────────────────────────────
    # 5b. INDEPENDENT GROUNDING for LLM-judged edges
    # ─────────────────────────────────────────
    _REL_GROUNDING_DESCRIPTION = {
        RelType.ELABORATES: "Passage A elaborates on / adds meaningful detail to something in Passage B",
        RelType.CONTRADICTS: "Passage A states something that directly contradicts / is inconsistent with Passage B",
        RelType.PREREQUISITE_OF: "understanding Passage A is a genuine prerequisite for understanding Passage B",
    }

    _GROUNDING_PROMPT = """You are independently auditing a claimed relationship between two passages. \
Do not assume the claim is correct -- your job is to catch it if it is NOT.

Passage A ({id_a}):
{text_a}

Passage B ({id_b}):
{text_b}

Claim being audited: {claim}

Based ONLY on the text above (ignore any framing in this prompt beyond the \
passages themselves), is this relationship specifically and genuinely \
supported? A vague topical overlap between the two passages is NOT enough \
on its own. Answer strictly as JSON: {{"grounded": true/false, "confidence": 0.0-1.0}}"""

    def _ground_edge(self, rel_type: RelType, src_node: DKGNode, tgt_node: DKGNode) -> tuple[bool, float]:
        """Second, independent LLM call verifying a CONTRADICTS/ELABORATES/
        PREREQUISITE_OF claim against the two passages alone -- deliberately
        does NOT see the first call's stated "reason", so it can't just
        agree with reasoning it was handed rather than checking the text
        itself. Fails closed (not grounded) on any parse/provider error,
        same posture as ontology_validation.score_axis2_idea_linking's
        audit -- an ungrounded-by-default edge is the safe failure mode,
        not a silently-kept one."""
        claim = self._REL_GROUNDING_DESCRIPTION.get(rel_type, str(rel_type))
        try:
            resp = self.client.chat_completion(
                model=AXIS2_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": self._GROUNDING_PROMPT.format(
                    id_a=src_node.id, text_a=src_node.text[:1500],
                    id_b=tgt_node.id, text_b=tgt_node.text[:1500],
                    claim=claim,
                )}],
                max_tokens=AXIS2_RELATION_MAX_TOKENS,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return bool(data.get("grounded", False)), float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            return False, 0.0

    # ─────────────────────────────────────────
    # 6. LLM PASS — CONTRADICTS / ELABORATES / PREREQUISITE_OF (parallel)
    # ─────────────────────────────────────────
    def _build_llm_edges(
        self, nodes: list[DKGNode], entity_edges: Optional[list[DKGEdge]] = None
    ) -> list[DKGEdge]:
        """
        Runs on two candidate pools, both capped by AXIS2_MAX_LLM_PAIRS total,
        with bounded parallel LLM calls (AXIS2_LLM_PAIR_CONCURRENCY):
          - top-k highest embedding-similarity pairs (surface-similar
            phrasing of the same idea)
          - "entity-bridge" pairs from entity_edges (already computed,
            rarity-weighted SHARES_ENTITY edges) that are NOT already
            embedding-similar -- two sections can share a rare named
            entity while phrasing it differently enough that cosine
            similarity alone misses the relationship entirely. Without
            this, pure similarity-ranked sampling only ever sees
            paraphrase-level pairs and never cross-topic logical ones.
        """
        edges: list[DKGEdge] = []
        embedded = [n for n in nodes if n.embedding is not None]
        if len(embedded) < 2:
            return edges

        vecs  = np.array([n.embedding for n in embedded], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = vecs / (norms + 1e-10)
        sim   = vecs @ vecs.T

        sim_candidates: list[Tuple[float, int, int]] = []
        for i, j in itertools.combinations(range(len(embedded)), 2):
            score = float(sim[i, j])
            if score >= CONTRADICTION_THRESH:
                sim_candidates.append((score, i, j))
        sim_candidates.sort(reverse=True)

        bridge_candidates: list[Tuple[float, int, int]] = []
        if entity_edges:
            id_to_idx = {n.id: i for i, n in enumerate(embedded)}
            seen_pairs = {(min(i, j), max(i, j)) for _, i, j in sim_candidates}
            for e in entity_edges:
                i, j = id_to_idx.get(e.source_id), id_to_idx.get(e.target_id)
                if i is None or j is None:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs or float(sim[i, j]) >= CONTRADICTION_THRESH:
                    continue  # already covered by the similarity pool
                bridge_candidates.append((float(e.weight or 0.0), pair[0], pair[1]))
            bridge_candidates.sort(reverse=True)

        # Reserve up to a third of the fixed LLM-call budget for bridge
        # pairs -- a single ranked-by-score list would always favor the
        # similarity pool (it's the higher-scoring one by construction),
        # crowding entity bridges out entirely regardless of how many exist.
        budget = AXIS2_MAX_LLM_PAIRS
        bridge_budget = min(len(bridge_candidates), budget // 3)
        sim_budget = budget - bridge_budget
        candidates = sim_candidates[:sim_budget] + bridge_candidates[:bridge_budget]

        if not candidates:
            return edges

        PROMPT = """You are analyzing two sections of a document.

Section A ({id_a}): {text_a}

Section B ({id_b}): {text_b}

Determine the relationship. Return ONLY valid JSON:
{{
  "relationship": "ELABORATES" | "CONTRADICTS" | "PREREQUISITE_OF" | "NONE",
  "direction": "A_TO_B" | "B_TO_A" | "SYMMETRIC",
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}"""

        def _llm_pair(score: float, i: int, j: int) -> Optional[DKGEdge]:
            a, b = embedded[i], embedded[j]
            try:
                resp = self.client.chat_completion(
                    model=AXIS2_MODEL,
                    temperature=0,
                    messages=[{"role": "user", "content": PROMPT.format(
                        id_a=a.id, text_a=a.text[:1500],
                        id_b=b.id, text_b=b.text[:1500],
                    )}],
                    max_tokens=AXIS2_RELATION_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw)

                rel_map = {
                    "ELABORATES":      RelType.ELABORATES,
                    "CONTRADICTS":     RelType.CONTRADICTS,
                    "PREREQUISITE_OF": RelType.PREREQUISITE_OF,
                }
                rel = rel_map.get(data.get("relationship", "NONE"))
                if rel and data.get("confidence", 0) >= 0.7:
                    src, tgt = (
                        (a.id, b.id)
                        if data["direction"] in ("A_TO_B", "SYMMETRIC")
                        else (b.id, a.id)
                    )
                    edge = DKGEdge(
                        source_id  = src,
                        target_id  = tgt,
                        rel_type   = rel,
                        weight     = data["confidence"],
                        axis       = 2,
                        properties = {"reason": data.get("reason", "")},
                        confidence = data["confidence"],
                        confidence_tier = EdgeConfidenceTier.INFERRED,
                    )
                    if AXIS2_GROUND_LLM_EDGES:
                        src_node, tgt_node = (a, b) if src == a.id else (b, a)
                        grounded, grounding_confidence = self._ground_edge(rel, src_node, tgt_node)
                        edge.properties["grounding_checked"] = True
                        edge.properties["grounding_confidence"] = round(grounding_confidence, 4)
                        if not grounded:
                            return None
                    return edge
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(
            max_workers=AXIS2_LLM_PAIR_CONCURRENCY,
            thread_name_prefix="axis2_llm",
        ) as pool:
            futures = {
                pool.submit(_llm_pair, score, i, j): (i, j)
                for score, i, j in candidates
            }
            for fut in as_completed(futures):
                try:
                    edge = fut.result()
                    if edge is not None:
                        edges.append(edge)
                except Exception:
                    pass

        return edges


# Alias for GraphConstructionService (docs/DESIGN_unstructured_graph_v2.md
# §4/§8): Axis2Builder already implements GraphAxisBuilder's `build(nodes,
# run_llm_pass=...)` shape exactly, so this is a name, not a rewrite --
# renaming/moving the class outright would touch the 9 existing call sites
# for no behavioral benefit.
Axis2IdeaBuilder = Axis2Builder
