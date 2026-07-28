"""
axis2.py — Semantic relationship discovery (Axis 2).

Builds:
    SEMANTICALLY_SIMILAR  — embedding cosine similarity
    SHARES_ENTITY         — shared NER entities between nodes
    SAME_CATEGORY         — KMeans cluster membership
    CONTRADICTS           — LLM reasoning pass (expensive, optional)
    ELABORATES            — LLM reasoning pass (expensive, optional)
    PREREQUISITE_OF       — LLM reasoning pass (expensive, optional)

Design principles:
  - Cheap relationships (SIMILAR, SHARES_ENTITY, SAME_CATEGORY) run always
  - Expensive LLM relationships run only on top-k candidate pairs
  - NER and LLM-pair calls are parallelised with bounded ThreadPoolExecutors
  - All relationships are Axis 2 flagged
"""
import difflib
import json
import itertools
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from ..config.settings import (
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
N_CLUSTERS             = None   # None = auto (sqrt of chapter count)
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


def _canonicalize_entities(all_entities: set[str]) -> dict[str, str]:
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

    Only compares strings within the same first-N-character prefix bucket.
    A plain length-sorted scan is still O(k^2) in practice for natural-
    language vocabulary: word lengths cluster too tightly for a length gap
    to prune much (verified: 1,200 physics-textbook-scale entity strings
    took 5+ seconds, and this only gets worse per document since entity
    vocabulary grows with corpus size — the same class of blowup this file
    already fixes elsewhere via degree-capping). Near-duplicates (typos,
    possessive/plural variants) overwhelmingly share a prefix, so bucketing
    by it keeps same-cluster comparisons intact while cutting cross-bucket
    ones entirely — trades recall on prefix-diverging aliases (e.g. "isaac
    newton" vs "newton") for bounded, near-linear cost at any corpus size.
    """
    if len(all_entities) < 2:
        return {e: e for e in all_entities}

    normalized: dict[str, str] = {}
    for e in all_entities:
        key = _WS_RE_ENTITY.sub(" ", _POSSESSIVE_RE.sub("", e)).strip()
        normalized[e] = key or e

    keys = sorted(set(normalized.values()))
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if len(keys) <= _ENTITY_FUZZY_CLUSTER_MAX_VOCAB:
        buckets: dict[str, list[str]] = {}
        for k in keys:
            buckets.setdefault(k[:_ENTITY_PREFIX_LEN], []).append(k)

        for bucket_keys in buckets.values():
            bucket_keys.sort(key=len)
            for i, a in enumerate(bucket_keys):
                for b in bucket_keys[i + 1:]:
                    if len(b) - len(a) > _ENTITY_MAX_LEN_DIFF:
                        break
                    if difflib.SequenceMatcher(None, a, b).ratio() >= _ENTITY_SIMILARITY_THRESHOLD:
                        union(a, b)

    cluster_members: dict[str, list[str]] = {}
    for k in keys:
        cluster_members.setdefault(find(k), []).append(k)

    # Canonical representative = the longest surface form in the cluster
    # (a fuller name is more informative to read back in edge.properties
    # than an abbreviation it absorbed).
    key_to_canonical = {
        member: max(members, key=len)
        for members in cluster_members.values()
        for member in members
    }
    return {e: key_to_canonical[normalized[e]] for e in all_entities}


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
        edges += self._build_entity_edges(nodes)

        # 5. SAME_CATEGORY (clustering)
        nodes, edges_cat = self._build_category_edges(nodes)
        edges += edges_cat

        # 6. LLM pass — CONTRADICTS / ELABORATES / PREREQUISITE_OF (parallel)
        if run_llm_pass and self.client:
            edges += self._build_llm_edges(nodes)

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
    def _extract_entities(self, nodes: list[DKGNode]) -> list[DKGNode]:
        """
        Uses LLM for NER in parallel (bounded by AXIS2_NER_CONCURRENCY),
        batching AXIS2_NER_BATCH_SIZE nodes per call instead of one call per
        node — a large document has thousands of Section/Page nodes, and
        one-call-per-node burns API request quota proportional to node
        count with no way to bound it. Returns top-10 entities per node to
        keep it manageable.
        """
        if not self.client:
            return nodes

        targets = [n for n in nodes if n.type in CONCEPT_NODE_TYPES]
        if not targets:
            return nodes

        batch_size = max(1, AXIS2_NER_BATCH_SIZE)
        batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]

        def _ner_batch(batch: list[DKGNode]) -> dict[str, list]:
            # Excerpts are keyed by a short local index ("0", "1", ...), not
            # the node's own (long) id -- cheaper in tokens and avoids the
            # model mangling a complex id string as a JSON key. Mapped back
            # to node.id below, after parsing.
            parts = [f"[{i}]\n{node.text[:1200]}" for i, node in enumerate(batch)]
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
                                "theories, technical terms). Return ONLY a JSON "
                                "object mapping each excerpt's number (as a string) "
                                "to its array of entity strings, e.g. "
                                '{"0": [...], "1": [...]}. Include every excerpt '
                                "number, even if its array is empty. No explanation."
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
                parsed = {}

            result: dict[str, list] = {}
            for i, node in enumerate(batch):
                entities = parsed.get(str(i)) if isinstance(parsed, dict) else None
                result[node.id] = entities if isinstance(entities, list) else []
            return result

        id_to_node = {n.id: n for n in targets}

        with ThreadPoolExecutor(max_workers=AXIS2_NER_CONCURRENCY, thread_name_prefix="axis2_ner") as pool:
            futures = [pool.submit(_ner_batch, batch) for batch in batches]
            for fut in as_completed(futures):
                try:
                    batch_result = fut.result()
                except Exception:
                    continue
                for node_id, entities in batch_result.items():
                    if node_id in id_to_node:
                        id_to_node[node_id].entities = entities

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
        # other. See _canonicalize_entities for why this is corpus-vocabulary
        # driven, not hardcoded to any document.
        raw_entities = {e.lower() for node in entity_nodes for e in node.entities}
        canonical = _canonicalize_entities(raw_entities)

        entity_to_nodes: dict[str, list[int]] = {}
        node_entity_sets: list[set] = []
        for idx, node in enumerate(entity_nodes):
            ents = set(canonical[e.lower()] for e in node.entities)
            node_entity_sets.append(ents)
            for e in ents:
                entity_to_nodes.setdefault(e, []).append(idx)

        pair_counts: dict[Tuple[int, int], int] = {}
        for idx_list in entity_to_nodes.values():
            for a, b in itertools.combinations(sorted(idx_list), 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        candidates = [(i, j, float(c)) for (i, j), c in pair_counts.items()]
        cap = AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
        for i, j, _count in _cap_edges_by_degree(candidates, cap):
            a, b = entity_nodes[i], entity_nodes[j]
            shared = node_entity_sets[i] & node_entity_sets[j]
            edges.append(DKGEdge(
                source_id  = a.id,
                target_id  = b.id,
                rel_type   = RelType.SHARES_ENTITY,
                weight     = len(shared),
                axis       = 2,
                properties = {"shared_entities": list(shared)},
            ))

        return edges

    # ─────────────────────────────────────────
    # 5. SAME_CATEGORY (KMeans)
    # ─────────────────────────────────────────
    def _build_category_edges(
        self, nodes: list[DKGNode]
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        from sklearn.cluster import KMeans

        embedded = [n for n in nodes if n.embedding is not None]
        edges: list[DKGEdge] = []

        if len(embedded) < 3:
            return nodes, edges

        # Auto k: sqrt of node count, min 2 max 10
        k = N_CLUSTERS or max(2, min(10, int(len(embedded) ** 0.5)))
        vecs = np.array([n.embedding for n in embedded], dtype=np.float32)
        km   = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(vecs)

        for node, label in zip(embedded, labels):
            node.cluster_id = int(label)

        # Cluster count is capped at 10 regardless of corpus size (above),
        # so cluster SIZE grows with the corpus instead of staying flat --
        # connecting every pair within a cluster (the previous behavior)
        # then grows quadratically with cluster size. This is what actually
        # produced 2.17M edges for a 7,165-node document: 10 clusters ->
        # ~716 members each -> C(716,2) * 10 ≈ 2.56M, dwarfing the other two
        # edge builders combined. Fixed the same way as SEMANTICALLY_SIMILAR:
        # cap each node to its top-k most-similar members of its OWN
        # cluster, computed on a per-cluster similarity submatrix (cheap --
        # clusters are far smaller than the full corpus).
        clusters: dict[int, list[int]] = {}  # cluster_id -> indices into embedded
        for idx, node in enumerate(embedded):
            clusters.setdefault(node.cluster_id, []).append(idx)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit_vecs = vecs / (norms + 1e-10)
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
                a, b = embedded[pair[0]], embedded[pair[1]]
                edges.append(DKGEdge(
                    source_id  = a.id,
                    target_id  = b.id,
                    rel_type   = RelType.SAME_CATEGORY,
                    axis       = 2,
                    properties = {"cluster_id": cluster_id},
                    confidence = SAME_CATEGORY_CONFIDENCE,
                    confidence_tier = EdgeConfidenceTier.AMBIGUOUS,
                ))

        return nodes, edges

    # ─────────────────────────────────────────
    # 6. LLM PASS — CONTRADICTS / ELABORATES / PREREQUISITE_OF (parallel)
    # ─────────────────────────────────────────
    def _build_llm_edges(self, nodes: list[DKGNode]) -> list[DKGEdge]:
        """
        Runs only on top-k highest-similarity pairs (capped by AXIS2_MAX_LLM_PAIRS)
        with bounded parallel LLM calls (AXIS2_LLM_PAIR_CONCURRENCY).
        """
        edges: list[DKGEdge] = []
        embedded = [n for n in nodes if n.embedding is not None]
        if len(embedded) < 2:
            return edges

        vecs  = np.array([n.embedding for n in embedded], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = vecs / (norms + 1e-10)
        sim   = vecs @ vecs.T

        # Collect all candidate pairs above the threshold, sorted by similarity
        # (highest first) then capped to AXIS2_MAX_LLM_PAIRS.
        candidates: list[Tuple[float, int, int]] = []
        for i, j in itertools.combinations(range(len(embedded)), 2):
            score = float(sim[i, j])
            if score >= CONTRADICTION_THRESH:
                candidates.append((score, i, j))

        # Sort descending by similarity and cap
        candidates.sort(reverse=True)
        candidates = candidates[:AXIS2_MAX_LLM_PAIRS]

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
                    return DKGEdge(
                        source_id  = src,
                        target_id  = tgt,
                        rel_type   = rel,
                        weight     = data["confidence"],
                        axis       = 2,
                        properties = {"reason": data.get("reason", "")},
                        confidence = data["confidence"],
                        confidence_tier = EdgeConfidenceTier.INFERRED,
                    )
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
