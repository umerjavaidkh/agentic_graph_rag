"""
chapter_summary.py — Chapter-level rollup summaries (cheap alternative to
Leiden community detection).

Ingested documents already have real, author-provided structure (Chapter
CONTAINS Section, via Axis 1 edges) — unlike the fully unstructured text
graph community-detection algorithms are built to cluster from scratch.
Reusing that structure by summarizing each Chapter from its own sections'
titles/excerpts gets most of the value of "what does this document/chapter
discuss" retrieval at a fraction of the engineering and LLM cost: no new
graph algorithm dependency, no new node type, one bounded LLM call per
chapter instead of one per discovered community.

Runs in-memory on the node/edge list, mirroring Axis2Builder — no Neo4j
session/driver involved here; the generic exporter write persists the
result via DKGNode.summary.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..config.prompts import load_prompt
from ..config.settings import (
    CHAPTER_SUMMARY_CONCURRENCY,
    CHAPTER_SUMMARY_MAX_CONTEXT_CHARS,
    CHAPTER_SUMMARY_MAX_TOKENS,
    CHAPTER_SUMMARY_MODEL,
    CHAPTER_SUMMARY_SECTION_EXCERPT_CHARS,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
)
from ..model_providers.factory import get_model_provider
from ..models import DKGEdge, DKGNode, NodeType, RelType


def _label(node: DKGNode) -> str:
    return node.type.value if isinstance(node.type, NodeType) else str(node.type)


def _rel(edge: DKGEdge) -> str:
    return edge.rel_type.value if isinstance(edge.rel_type, RelType) else str(edge.rel_type)


class ChapterSummaryBuilder:
    """
    Usage:
        builder = ChapterSummaryBuilder(api_key="sk-...")
        nodes = builder.build(nodes, edges)
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or OPENAI_API_KEY
        self.client = get_model_provider(MODEL_PROVIDER, key)

    def build(self, nodes: list[DKGNode], edges: list[DKGEdge]) -> list[DKGNode]:
        if not self.client:
            return nodes

        chapters = [n for n in nodes if _label(n) == NodeType.CHAPTER.value]
        if not chapters:
            return nodes

        sections_by_chapter = self._sections_by_chapter(nodes, edges)

        def _summarize_one(chapter: DKGNode) -> None:
            sections = sections_by_chapter.get(chapter.id, [])
            context = self._build_context(sections)
            if not context:
                return
            try:
                resp = self.client.chat_completion(
                    model=CHAPTER_SUMMARY_MODEL,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": load_prompt(
                            "chapter_summary",
                            chapter_title=chapter.title or chapter.id,
                            context=context,
                        ),
                    }],
                    max_tokens=CHAPTER_SUMMARY_MAX_TOKENS,
                )
                summary = (resp.choices[0].message.content or "").strip()
                if summary:
                    chapter.summary = summary
            except Exception:
                # Best-effort enrichment — a failed summary call must never
                # fail ingestion itself (mirrors Axis2's per-pair try/except).
                pass

        with ThreadPoolExecutor(
            max_workers=max(1, CHAPTER_SUMMARY_CONCURRENCY), thread_name_prefix="chapter_summary"
        ) as pool:
            list(pool.map(_summarize_one, chapters))

        return nodes

    @staticmethod
    def _sections_by_chapter(
        nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> dict[str, list[DKGNode]]:
        """Map each Chapter id to its descendant Section nodes, reached via
        CONTAINS edges (Chapter -> Section, Section -> numbered subsection).
        Page/Region descendants are skipped — they duplicate Section text or
        hold table/figure content that isn't useful summarization input."""
        nodes_by_id = {n.id: n for n in nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if _rel(e) == RelType.CONTAINS.value:
                children[e.source_id].append(e.target_id)

        chapters = [n for n in nodes if _label(n) == NodeType.CHAPTER.value]
        result: dict[str, list[DKGNode]] = {}
        for chapter in chapters:
            seen: set[str] = set()
            queue = list(children.get(chapter.id, []))
            sections: list[DKGNode] = []
            while queue:
                child_id = queue.pop(0)
                if child_id in seen:
                    continue
                seen.add(child_id)
                child = nodes_by_id.get(child_id)
                if child is None:
                    continue
                if _label(child) == NodeType.SECTION.value:
                    sections.append(child)
                queue.extend(children.get(child_id, []))
            sections.sort(key=lambda n: n.order)
            result[chapter.id] = sections
        return result

    @staticmethod
    def _build_context(sections: list[DKGNode]) -> str:
        parts: list[str] = []
        total = 0
        for sec in sections:
            excerpt = (sec.text or "").strip()[:CHAPTER_SUMMARY_SECTION_EXCERPT_CHARS]
            if not excerpt:
                continue
            block = f"- {sec.title or sec.id}: {excerpt}"
            if total + len(block) > CHAPTER_SUMMARY_MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)
