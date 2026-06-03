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
import json
import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from ..config.settings import (
    AXIS2_LLM_PAIR_CONCURRENCY,
    AXIS2_MAX_LLM_PAIRS,
    AXIS2_MODEL,
    AXIS2_NER_CONCURRENCY,
    AXIS2_NER_MAX_TOKENS,
    AXIS2_RELATION_MAX_TOKENS,
    EMBEDDING_MODEL,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
)
from ..model_providers.factory import get_model_provider
from ..models import DKGNode, DKGEdge, NodeType, RelType


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD   = 0.75   # cosine sim for SEMANTICALLY_SIMILAR
CONTRADICTION_THRESH   = 0.85   # only run LLM on very similar pairs
N_CLUSTERS             = None   # None = auto (sqrt of chapter count)
# Node types to include in semantic analysis (skip PAGE for perf)
SEMANTIC_NODE_TYPES    = {NodeType.CHAPTER, NodeType.SECTION}
CONCEPT_NODE_TYPES     = {NodeType.SECTION, NodeType.PAGE}


# ─────────────────────────────────────────
# AXIS 2 BUILDER
# ─────────────────────────────────────────
class Axis2Builder:
    """
    Takes the node list from document ingestion and enriches it with
    all Axis 2 semantic edges.

    Usage:
        builder = Axis2Builder(api_key="sk-...")
        nodes, new_edges = builder.build(nodes)
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or OPENAI_API_KEY
        self.client = get_model_provider(MODEL_PROVIDER, key)

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
            response = self.client.embeddings(
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
        Uses LLM for NER in parallel (bounded by AXIS2_NER_CONCURRENCY).
        Returns top-10 entities per node to keep it manageable.
        """
        if not self.client:
            return nodes

        targets = [n for n in nodes if n.type in CONCEPT_NODE_TYPES]
        if not targets:
            return nodes

        def _ner_one(node: DKGNode) -> Tuple[str, list]:
            try:
                resp = self.client.chat_completion(
                    model=AXIS2_MODEL,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract the top 10 named entities (people, organizations, "
                                "concepts, theories, technical terms) from the text. "
                                "Return ONLY a JSON array of strings. No explanation."
                            ),
                        },
                        {"role": "user", "content": node.text[:3000]},
                    ],
                    max_tokens=AXIS2_NER_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                return node.id, json.loads(raw)
            except Exception:
                return node.id, []

        # Build id→node lookup for result assignment
        id_to_node = {n.id: n for n in targets}

        with ThreadPoolExecutor(max_workers=AXIS2_NER_CONCURRENCY, thread_name_prefix="axis2_ner") as pool:
            futures = {pool.submit(_ner_one, node): node.id for node in targets}
            for fut in as_completed(futures):
                try:
                    node_id, entities = fut.result()
                    if node_id in id_to_node:
                        id_to_node[node_id].entities = entities
                except Exception:
                    pass

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

        for i, j in itertools.combinations(range(len(embedded)), 2):
            score = float(sim[i, j])
            if score >= SIMILARITY_THRESHOLD:
                a, b = embedded[i], embedded[j]
                edges.append(DKGEdge(
                    source_id  = a.id,
                    target_id  = b.id,
                    rel_type   = RelType.SEMANTICALLY_SIMILAR,
                    weight     = round(score, 4),
                    axis       = 2,
                    properties = {"score": round(score, 4)},
                ))

        return edges

    # ─────────────────────────────────────────
    # 4. SHARES_ENTITY
    # ─────────────────────────────────────────
    def _build_entity_edges(self, nodes: list[DKGNode]) -> list[DKGEdge]:
        edges: list[DKGEdge] = []
        entity_nodes = [n for n in nodes if n.entities]

        for i, j in itertools.combinations(range(len(entity_nodes)), 2):
            a, b = entity_nodes[i], entity_nodes[j]
            shared = set(e.lower() for e in a.entities) & \
                     set(e.lower() for e in b.entities)
            if shared:
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

        # Build SAME_CATEGORY edges within each cluster
        clusters: dict[int, list[DKGNode]] = {}
        for node in embedded:
            clusters.setdefault(node.cluster_id, []).append(node)

        for cluster_id, members in clusters.items():
            for a, b in itertools.combinations(members, 2):
                edges.append(DKGEdge(
                    source_id  = a.id,
                    target_id  = b.id,
                    rel_type   = RelType.SAME_CATEGORY,
                    axis       = 2,
                    properties = {"cluster_id": cluster_id},
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
