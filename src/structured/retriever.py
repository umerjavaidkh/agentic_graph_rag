"""
structured/retriever.py — Universal Structured Graph Retriever
Now with built-in LLM synthesis. Call .ask() for a complete answer.
WITH role-based access control.
"""

import os
import json
import numpy as np
from typing import Optional
from neo4j import GraphDatabase
from ..config.settings import MODEL_PROVIDER, OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL
from ..model_providers.factory import get_model_provider
from ..auth.roles import UserContext, Role, DEFAULT_PUBLIC_CONTEXT
from ..auth.rbac_setup import GraphRBAC

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)
LLM_MODEL = CHAT_MODEL


class StructuredRetriever:
    """
    Universal retriever for any structured Neo4j graph.
    Schema is discovered dynamically — works for Northwind,
    Excel imports, or any future structured dataset.
    
    NEW: Built-in LLM synthesis via .ask(question) → natural language answer.
    """

    def __init__(
        self,
        uri:      str = "bolt://localhost:7687",
        user:     str = "neo4j",
        password: str = "password123",
        user_context: Optional[UserContext] = None,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._schema_cache: Optional[str] = None
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password)

    # ─────────────────────────────────────────
    # PUBLIC API  (end-to-end)
    # ─────────────────────────────────────────

    def ask(self, query: str, limit: int = 5, user_context: Optional[UserContext] = None) -> dict:
        """
        ONE call → natural language answer + sources.
        Usage:
            result = retriever.ask("Which products are most commonly bought together?")
            print(result["answer"])   # human-readable insight
            print(result["sources"])  # raw structured data for citations
        """
        ctx = user_context or self.user_context
        
        # 1. Retrieve
        retrieval = self.retrieve(query, limit, user_context=ctx)
        chunks = retrieval.get("chunks", [])

        # 2. Synthesize (if we got data)
        if chunks and chunks[0].get("id") != "error":
            answer = self._synthesize(query, chunks)
        else:
            answer = "I couldn't find any relevant data to answer that question."

        return {
            "query": query,
            "answer": answer,
            "strategy": retrieval.get("strategy", "unknown"),
            "sources": chunks,
            "total_sources": len(chunks),
            "_access_level": ctx.role.value,
        }

    def retrieve(self, query: str, limit: int = 5, user_context: Optional[UserContext] = None) -> dict:
        """
        Raw retrieval entry point. Returns structured chunks.
        Auto-selects strategy: text2cypher | vector_search | multi_hop
        """
        ctx = user_context or self.user_context
        strategy = self._classify_query(query)

        if strategy == "vector":
            results = self._vector_search(query, limit, user_context=ctx)
            results = self._enrich_graph_context(results)
        elif strategy == "multi_hop":
            results = self._graph_hop_retrieve(query, limit, hops=2, user_context=ctx)
        else:
            results = self._text2cypher(query, limit, user_context=ctx)
            # NEW: enrich text2cypher results too (not just vector)
            results = self._enrich_graph_context(results)

        return self._format_response(query, results, strategy)

    def multi_hop_retrieve(self, query: str, limit: int = 5, hops: int = 2, user_context: Optional[UserContext] = None) -> dict:
        ctx = user_context or self.user_context
        results = self._graph_hop_retrieve(query, limit, hops, user_context=ctx)
        return self._format_response(query, results, "multi_hop")

    def get_schema(self) -> dict:
        schema = self._fetch_schema()
        return {
            "query": "schema",
            "chunks": [
                {"id": "schema", "title": "Graph Schema", "text": schema, "related": []}
            ],
            "total_available": 1,
        }

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────
    # LLM SYNTHESIS  (NEW)
    # ─────────────────────────────────────────

    def _synthesize(self, query: str, chunks: list[dict]) -> str:
        """
        Takes raw structured chunks and produces a natural language answer.
        Handles product IDs gracefully by asking the LLM to interpret contextually.
        """
        context = self._build_context(chunks)

        system_prompt = """You are a senior business analyst answering questions from a Neo4j graph database.
Rules:
- Answer ONLY using the provided data. If data is insufficient, say so.
- When product IDs appear, describe them naturally (e.g., "the most frequently paired product").
- Do NOT invent facts not present in the data.
- Be concise but insightful. Mention specific numbers and rankings when relevant.
- If the data shows duplicate symmetric pairs (e.g., 21+61 and 61+21), treat them as ONE pair."""

        user_prompt = f"""Retrieved Data:
{context}

User Question: {query}

Provide a clear, natural language answer. If products or entities have names available, use them. If only IDs are available, refer to them by their relationship patterns (e.g., "Product 21")."""

        response = provider.chat_completion(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def _build_context(self, chunks: list[dict]) -> str:
        """Formats chunks into a clean text block for the LLM prompt."""
        parts = []
        for i, chunk in enumerate(chunks[:8], 1):  # cap at 8 to stay within context
            title = chunk.get("title", "Unknown")
            text = chunk.get("text", "")
            label = chunk.get("label", "")
            score = chunk.get("score", 0.0)
            
            # If there's a Cypher query, include it for transparency
            cypher = chunk.get("cypher", "")
            
            part = f"[Result {i}]"
            if label:
                part += f" Type: {label}"
            if score:
                part += f" | Relevance: {score}"
            part += f"\nTitle: {title}\nDetails: {text}"
            if cypher:
                part += f"\nQuery: {cypher}"
            parts.append(part)
        
        return "\n\n".join(parts) if parts else "No data retrieved."

    # ─────────────────────────────────────────
    # STRATEGY 1 — TEXT2CYPHER  (improved)
    # ─────────────────────────────────────────

    def _text2cypher(self, query: str, limit: int, user_context: Optional[UserContext] = None) -> list:
        ctx = user_context or self.user_context
        user_id = ctx.user_id
        
        # RBAC: Check if user can query structured knowledge area
        if not self.rbac.can_query_knowledge_area(user_id, 'structured'):
            return [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {user_id} does not have permission to query structured data.",
                "score": 0.0,
                "related": [],
            }]
        
        schema = self._fetch_schema()
        cypher = self._generate_cypher(query, schema, limit)

        if not cypher:
            return []

        # Execute query (no Python-level row filtering needed now)
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                rows = [r.data() for r in result]

            return [
                {
                    "id": f"row_{i}",
                    "title": self._row_title(row),
                    "text": self._row_to_text(row),
                    "raw": row,
                    "score": 1.0,
                    "cypher": cypher,
                    "related": [],
                }
                for i, row in enumerate(rows)
            ]
        except Exception as e:
            return [{
                "id": "error",
                "title": "Query Error",
                "text": f"Generated Cypher failed: {str(e)}\nCypher: {cypher}",
                "score": 0.0,
                "related": [],
            }]

    def _generate_cypher(self, query: str, schema: str, limit: int) -> Optional[str]:
        prompt = f"""You are a Neo4j Cypher expert. Generate a Cypher query for the question below.

GRAPH SCHEMA:
{schema}

RULES:
- Return ONLY the Cypher query, no explanation, no markdown
- Always include LIMIT {limit} unless it's an aggregation
- Use OPTIONAL MATCH for nullable relationships
- For aggregations (count, sum, avg) do NOT add LIMIT
- Property names are case-sensitive — use schema exactly
- For text search use toLower() and CONTAINS
- **CRITICAL**: For "bought together" / "frequently purchased" / market-basket queries:
  Use `WHERE id(p1) < id(p2)` or `WHERE p1.productID < p2.productID` to deduplicate symmetric pairs.
  Return product names (productName, companyName) alongside IDs whenever possible.

QUESTION: {query}

CYPHER:"""

        response = provider.chat_completion(
            model=LLM_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        cypher = response.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher if cypher else None

    # ─────────────────────────────────────────
    # STRATEGY 2 — VECTOR SEARCH
    # ─────────────────────────────────────────

    def _vector_search(self, query: str, limit: int, user_context: Optional[UserContext] = None) -> list:
        ctx = user_context or self.user_context
        user_id = ctx.user_id
        
        # RBAC: Check if user can query structured knowledge area
        if not self.rbac.can_query_knowledge_area(user_id, 'structured'):
            return []
        
        embedding = self._embed(query)
        indexed_labels = self._get_vector_indexed_labels()

        if not indexed_labels:
            return self._text2cypher(query, limit, user_context=ctx)

        results = []
        with self.driver.session() as session:
            for label, index_name, prop in indexed_labels:
                try:
                    rows = session.run(f"""
                        CALL db.index.vector.queryNodes($index, $limit, $embedding)
                        YIELD node AS n, score
                        RETURN n, score, '{label}' AS label
                    """, index=index_name, limit=limit, embedding=embedding.tolist())

                    for r in rows:
                        node = dict(r["n"])
                        results.append({
                            "id": node.get("productID") or node.get("id") or f"{label}_{len(results)}",
                            "title": node.get("productName") or node.get("name") or node.get("title") or label,
                            "text": node.get("text") or self._node_to_text(node),
                            "score": round(r["score"], 4),
                            "label": label,
                            "raw": node,
                            "related": [],
                        })
                except Exception:
                    continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ─────────────────────────────────────────
    # STRATEGY 3 — GRAPH CONTEXT ENRICHMENT  (enhanced)
    # ─────────────────────────────────────────

    def _enrich_graph_context(self, items: list) -> list:
        """
        Now works for ANY item with a productID, not just vector results.
        Also enriches with actual product names so the LLM can speak naturally.
        """
        if not items:
            return items

        with self.driver.session() as session:
            for item in items:
                raw = item.get("raw", {})
                product_id = raw.get("product1") or raw.get("productID") or item.get("id")

                # Case A: It's a Product node (from vector search or direct lookup)
                if item.get("label") == "Product" or raw.get("productID"):
                    result = session.run("""
                        MATCH (p:Product {productID: $id})
                        OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
                        OPTIONAL MATCH (p)<-[:SUPPLIED_BY]-(s:Supplier)
                        OPTIONAL MATCH (o:Order)-[:ORDER_CONTAINS]->(p)
                        RETURN p.productName AS productName,
                               c.categoryName AS category,
                               s.companyName AS supplier,
                               count(DISTINCT o) AS total_orders
                    """, id=str(product_id))
                    ctx = result.single()
                    if ctx:
                        name = ctx["productName"]
                        if name and item.get("title") == item.get("id"):
                            item["title"] = name  # replace ID title with real name
                        item["related"] = [
                            r for r in [ctx["category"], ctx["supplier"]] if r
                        ]
                        item["text"] += (
                            f"\nProduct Name: {ctx['productName']}"
                            f"\nCategory: {ctx['category']}"
                            f"\nSupplier: {ctx['supplier']}"
                            f"\nTotal Orders: {ctx['total_orders']}"
                        )

                # Case B: It's a pair result (market basket / bought together)
                elif "product1" in raw and "product2" in raw:
                    # Look up names for both products
                    names = session.run("""
                        MATCH (p1:Product {productID: $id1})
                        MATCH (p2:Product {productID: $id2})
                        RETURN p1.productName AS name1, p2.productName AS name2
                    """, id1=str(raw["product1"]), id2=str(raw["product2"]))
                    name_row = names.single()
                    if name_row:
                        n1, n2 = name_row["name1"], name_row["name2"]
                        # Inject names into the text so LLM sees them
                        item["text"] += f"\nProduct 1 Name: {n1}\nProduct 2 Name: {n2}"
                        if n1 and n2:
                            item["title"] = f"{n1} + {n2}"
        return items

    # ─────────────────────────────────────────
    # STRATEGY 4 — MULTI-HOP GRAPH TRAVERSAL
    # ─────────────────────────────────────────

    def _graph_hop_retrieve(self, query: str, limit: int, hops: int = 2, user_context: Optional[UserContext] = None) -> list:
        ctx = user_context or self.user_context
        user_id = ctx.user_id
        
        # RBAC: Check if user can query structured knowledge area
        if not self.rbac.can_query_knowledge_area(user_id, 'structured'):
            return []
        
        seed_results = self._vector_search(query, limit, user_context=ctx)
        if not seed_results:
            seed_results = self._text2cypher(query, limit, user_context=ctx)

        seed_ids = [
            r["raw"].get("productID") or r["raw"].get("orderID") or r["id"]
            for r in seed_results
            if r.get("id") != "error"
        ]

        if not seed_ids:
            return seed_results

        cypher = f"""
            MATCH (seed)
            WHERE seed.productID IN $seed_ids
               OR seed.orderID   IN $seed_ids
               OR seed.customerID IN $seed_ids

            MATCH path = (seed)-[*1..{hops}]-(neighbor)
            WHERE neighbor <> seed

            WITH neighbor,
                 labels(neighbor)[0] AS label,
                 count(DISTINCT seed) AS seed_connections,
                 min(length(path)) AS hop_distance
            WHERE label IS NOT NULL

            RETURN DISTINCT
                   neighbor.productID AS productID,
                   neighbor.productName AS productName,
                   neighbor.categoryName AS categoryName,
                   neighbor.companyName AS companyName,
                   neighbor.customerID AS customerID,
                   neighbor.orderID AS orderID,
                   label,
                   hop_distance,
                   seed_connections
            ORDER BY seed_connections DESC, hop_distance ASC
            LIMIT $expand_limit
        """

        expanded = []
        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    seed_ids=seed_ids,
                    expand_limit=limit * 2,
                )
                rows = [r.data() for r in result]

            seen = set(seed_ids)
            all_results = seed_results.copy()

            for row in rows:
                node_id = (
                    row.get("productID") or row.get("orderID") or
                    row.get("customerID") or str(len(all_results))
                )
                if node_id not in seen:
                    seen.add(node_id)
                    all_results.append({
                        "id": node_id,
                        "title": row.get("productName") or row.get("companyName") or
                                 row.get("categoryName") or row.get("customerID") or node_id,
                        "text": self._row_to_text({k: v for k, v in row.items() if v is not None}),
                        "score": 0.0,
                        "label": row.get("label", ""),
                        "hop_distance": row.get("hop_distance", 0),
                        "seed_connections": row.get("seed_connections", 0),
                        "related": [],
                        "raw": row,
                    })

            return all_results[:limit * 2]

        except Exception as e:
            return seed_results

    # ─────────────────────────────────────────
    # SCHEMA DISCOVERY
    # ─────────────────────────────────────────

    def _fetch_schema(self) -> str:
        if self._schema_cache:
            return self._schema_cache

        with self.driver.session() as session:
            labels_result = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
            """)
            nodes = [
                f"{r['nodeType']} {{{', '.join(r['properties'])}}}"
                for r in labels_result
            ]

            patterns_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to
            """)
            patterns = [
                f"(:{r['from']})-[:{r['rel']}]->(:{r['to']})"
                for r in patterns_result
            ]

        schema = (
            "NODE TYPES:\n" + "\n".join(nodes) +
            "\n\nRELATIONSHIPS:\n" + "\n".join(set(patterns))
        )
        self._schema_cache = schema
        return schema

    def _get_vector_indexed_labels(self) -> list[tuple]:
        try:
            with self.driver.session() as session:
                result = session.run("""
                    SHOW VECTOR INDEXES
                    YIELD name, labelsOrTypes, properties
                    RETURN name, labelsOrTypes[0] AS label, properties[0] AS prop
                """)
                return [(r["label"], r["name"], r["prop"]) for r in result]
        except Exception:
            return []

    # ─────────────────────────────────────────
    # QUERY CLASSIFIER
    # ─────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        multi_hop_signals = {
            "and their", "along with", "related to", "connected to",
            "together with", "as well as", "including their",
        }
        semantic_signals = {
            "similar", "like", "recommend", "suggest", "find me",
            "what products", "show me products", "seafood", "describe",
        }
        if any(s in q for s in multi_hop_signals):
            return "multi_hop"
        if any(s in q for s in semantic_signals):
            return "vector"
        return "text2cypher"

    def _apply_role_filter_to_cypher(self, cypher: str, user_context: UserContext) -> str:
        """
        Apply role-based filtering to a Cypher query.
        Adds sensitivity constraints for non-admin users.
        """
        allowed_sensitivity = RoleFilter._get_allowed_sensitivity_levels(user_context)
        sensitivity_list = ", ".join(f"'{s}'" for s in allowed_sensitivity)
        
        # Add a WHERE clause for sensitivity filtering
        # This assumes nodes have an optional 'sensitivity' property
        filter_clause = f"AND (n.sensitivity IS NULL OR n.sensitivity IN [{sensitivity_list}])"
        
        # Simple replacement: add filter to all node matches
        # This is a basic approach; production code might parse Cypher more carefully
        cypher = cypher.replace(
            "MATCH (n:",
            f"MATCH (n: WHERE {filter_clause} WITH n MATCH (n:"
        )
        if " WHERE " not in cypher and "MATCH" in cypher:
            cypher = cypher.replace("RETURN", f"WHERE (NOT exists(n.sensitivity) OR n.sensitivity IN [{sensitivity_list}]) RETURN")
        
        return cypher

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        response = provider.embeddings(model=EMBEDDING_MODEL, input=text[:8000])
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _row_title(self, row: dict) -> str:
        # Prefer names over IDs
        for key in ("productName", "name", "title", "companyName", "categoryName", 
                    "customerID", "orderID", "product1", "product2"):
            if key in row and row[key] is not None:
                return str(row[key])
        return str(list(row.values())[0]) if row else "Result"

    def _row_to_text(self, row: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)

    def _node_to_text(self, node: dict) -> str:
        return "\n".join(
            f"{k}: {v}" for k, v in node.items()
            if v is not None and k != "textEmbedding"
        )

    def _format_response(self, query: str, items: list, strategy: str) -> dict:
        return {
            "query": query,
            "strategy": strategy,
            "chunks": items,
            "total_available": len(items),
        }