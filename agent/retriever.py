from neo4j import GraphDatabase
from typing import List, Dict
from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ESGComplianceRetriever:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._vector_index_ready = False

    # ─────────────────────────────────────────
    # PUBLIC API  (same signatures as before)
    # ─────────────────────────────────────────

    def semantic_retrieve(self, query: str, limit: int = 5) -> Dict:
        query_embedding = self._get_embedding(query)

        # FIX 1: vector index query — no Python loop, no string parsing
        with self.driver.session() as session:
            top = self._vector_search(session, query_embedding, limit)
            if not top:
                # Fallback: no vector index yet → warn and use legacy
                print("⚠️  Vector index not found — run setup.cypher first for best performance")
                top = self._legacy_similarity(session, query_embedding, limit)

            # FIX 2: single batched context query — no N+1
            top = self._enrich_context_batch(session, top)

        return self._format_response(query, top)

    def get_all_sections(self) -> Dict:
        """Return ALL sections for TOC queries — unchanged."""
        cypher = """
        MATCH (b:Book)-[:CONTAINS*1..3]->(s:Section)
        RETURN s.id AS id, s.title AS title, s.order AS order,
               s.page_start AS page, s.cluster_id AS cluster
        ORDER BY s.order
        """
        with self.driver.session() as session:
            rows = [r.data() for r in session.run(cypher)]

        return {
            "query": "table_of_contents",
            "chunks": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "text": f"{r['title']} (Page {r['page']})",
                    "cluster": r["cluster"],
                    "related": [],
                }
                for r in rows
            ],
            "total_available": len(rows),
        }

    def multi_hop_retrieve(self, query: str, limit: int = 5, hops: int = 2) -> Dict:
        initial    = self.semantic_retrieve(query, limit=limit)
        seed_ids   = [c["id"] for c in initial["chunks"]]

        if not seed_ids:
            return initial

        # Neo4j requires literal path lengths — inline hops value
        cypher = f"""
            MATCH (seed:Section)
            WHERE seed.id IN $seed_ids
            MATCH (seed)-[:SAME_CATEGORY|SHARES_ENTITY*1..{hops}]-(related:Section)
            WHERE NOT related.id IN $seed_ids
            WITH related,
                 count(DISTINCT seed) AS seed_connections
            RETURN DISTINCT
                   related.id         AS id,
                   related.title      AS title,
                   related.text       AS text,
                   related.cluster_id AS cluster,
                   1                  AS hop_distance,
                   seed_connections
            ORDER BY seed_connections DESC
            LIMIT $expand_limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, seed_ids=seed_ids, expand_limit=limit * 2)
            expanded = [r.data() for r in result]

        seen       = set(seed_ids)
        all_chunks = initial["chunks"].copy()

        for r in expanded:
            if r["id"] not in seen:
                seen.add(r["id"])
                all_chunks.append({
                    "id":              r["id"],
                    "title":           r["title"],
                    "text":            r["text"],
                    "cluster":         r["cluster"],
                    "hop_distance":    r["hop_distance"],
                    "seed_connections":r["seed_connections"],
                    "related":         [],
                })

        return {
            "query":           query,
            "chunks":          all_chunks[: limit * 2],
            "total_available": len(all_chunks),
            "seeds":           len(seed_ids),
            "expanded":        len(expanded),
        }

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _get_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # FIX 1a — vector index path (fast)
    def _vector_search(self, session, embedding: np.ndarray, limit: int) -> list:
        try:
            result = session.run("""
                CALL db.index.vector.queryNodes('section_embedding', $limit, $embedding)
                YIELD node AS s, score
                RETURN s.id         AS id,
                       s.title      AS title,
                       s.text       AS text,
                       s.cluster_id AS cluster,
                       s.order      AS doc_order,
                       score
            """, embedding=embedding.tolist(), limit=limit)
            rows = [r.data() for r in result]
            self._vector_index_ready = bool(rows)
            return rows
        except Exception:
            return []

    # FIX 1b — legacy fallback (no vector index)
    def _legacy_similarity(self, session, embedding: np.ndarray, limit: int) -> list:
        import json
        result = session.run("""
            MATCH (s:Section)
            WHERE s.embedding IS NOT NULL
            RETURN s.id         AS id,
                   s.title      AS title,
                   s.text       AS text,
                   s.cluster_id AS cluster,
                   s.order      AS doc_order,
                   s.embedding  AS embedding
        """)
        rows = [r.data() for r in result]

        scored = []
        for row in rows:
            raw = row.pop("embedding")
            # handle string (stored as JSON) or list (stored as array)
            if isinstance(raw, str):
                raw = json.loads(raw)
            emb   = np.array(raw, dtype=np.float32)
            score = self._cosine_similarity(embedding, emb)
            scored.append({**row, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # FIX 2 — single batched context query
    def _enrich_context_batch(self, session, items: list) -> list:
        if not items:
            return items

        ids    = [i["id"] for i in items]
        result = session.run("""
            MATCH (s:Section) WHERE s.id IN $ids
            OPTIONAL MATCH (s)-[:SAME_CATEGORY]-(cm:Section)
            OPTIONAL MATCH (s)-[:SHARES_ENTITY]-(em:Section)
            OPTIONAL MATCH (s)-[:PRECEDES|FOLLOWS]-(nb:Section)
            OPTIONAL MATCH (b:Book)-[:CONTAINS*1..3]->(s)
            RETURN s.id                                    AS id,
                   collect(DISTINCT cm.title)[0..3]        AS cluster_context,
                   collect(DISTINCT em.title)[0..3]        AS entity_context,
                   collect(DISTINCT nb.title)[0..2]        AS sequence_context,
                   b.title                                 AS source_doc
        """, ids=ids)

        ctx_map = {r["id"]: r.data() for r in result}

        for item in items:
            ctx = ctx_map.get(item["id"], {})
            item["related"]     = (ctx.get("cluster_context") or []) + \
                                  (ctx.get("entity_context")  or [])
            item["source_doc"]  = ctx.get("source_doc", "")

        return items

    # ─────────────────────────────────────────
    # SHARED RESPONSE FORMATTER
    # ─────────────────────────────────────────

    def _format_response(self, query: str, items: list) -> Dict:
        return {
            "query": query,
            "chunks": [
                {
                    "id":      r["id"],
                    "title":   r["title"],
                    "text":    r["text"],
                    "cluster": r.get("cluster"),
                    "score":   round(r.get("score", 0.0), 3),
                    "related": r.get("related", []),
                }
                for r in items
            ],
            "total_available": len(items),
        }