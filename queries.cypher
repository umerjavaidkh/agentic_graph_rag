// ═══════════════════════════════════════════════════════════════
// DOCUMENT KNOWLEDGE GRAPH — Query Cookbook
// ═══════════════════════════════════════════════════════════════
// Run these in Neo4j Browser after importing your document graph


// ── AXIS 1: STRUCTURAL QUERIES ─────────────────────────────────

// 1. Full Table of Contents
MATCH (b:Book)-[:CONTAINS]->(c:Chapter)
RETURN c.order AS num, c.title AS chapter, c.page_start AS page
ORDER BY c.order;

// 2. All sections inside a chapter
MATCH (c:Chapter {order: 2})-[:CONTAINS]->(s:Section)
RETURN s.order AS num, s.title, s.page_start
ORDER BY s.order;

// 3. Full document tree (3 levels)
MATCH path = (b:Book)-[:CONTAINS*1..3]->(n)
RETURN path;

// 4. What page range does Chapter 3 cover?
MATCH (c:Chapter {order: 3})
RETURN c.title, c.page_start, c.page_end,
       (c.page_end - c.page_start + 1) AS total_pages;

// 5. Navigate sequentially (next/previous chapter)
MATCH (c:Chapter {order: 1})-[:PRECEDES]->(next:Chapter)
RETURN next.title AS next_chapter;

// 6. All pages in a section
MATCH (s:Section)-[:CONTAINS]->(p:Page)
WHERE s.title CONTAINS 'Introduction'
RETURN p.page_start AS page, p.text
ORDER BY p.order;


// ── AXIS 2: SEMANTIC QUERIES ───────────────────────────────────

// 7. Find all sections that discuss similar topics
MATCH (s:Section)-[r:SEMANTICALLY_SIMILAR]->(other:Section)
WHERE r.weight > 0.80
RETURN s.title, other.title, round(r.weight * 100) AS similarity_pct
ORDER BY r.weight DESC;

// 8. Which sections share the same named entities?
MATCH (a:Section)-[r:SHARES_ENTITY]->(b:Section)
RETURN a.title, b.title, r.shared_entities
ORDER BY size(r.shared_entities) DESC
LIMIT 20;

// 9. Thematic clusters — what topics are in cluster 2?
MATCH (n:Chapter) WHERE n.cluster_id = 2
RETURN n.title ORDER BY n.order;

// 10. Cross-chapter references (explicit "see Chapter X" links)
MATCH (a)-[r:REFERENCES]->(b)
RETURN a.title AS from_node, b.title AS to_node, r.matched_text
ORDER BY a.order;

// 11. Find contradictions in the document
MATCH (a)-[:CONTRADICTS]->(b)
RETURN a.title AS claim_a, b.title AS claim_b;

// 12. Prerequisite chain for a concept
MATCH path = (start:Section)-[:PREREQUISITE_OF*1..5]->(end:Section)
WHERE end.title CONTAINS 'Advanced'
RETURN path;


// ── HYBRID QUERIES (both axes) ─────────────────────────────────

// 13. Full context for a query: find entry + neighborhood
// (simulate a question: "tell me about risk management")
MATCH (n:Section)
WHERE n.title CONTAINS 'Risk'
WITH n
MATCH (n)-[:SEMANTICALLY_SIMILAR|SHARES_ENTITY|SAME_CATEGORY]-(related)
RETURN n.title AS topic,
       collect(DISTINCT related.title) AS related_sections
LIMIT 10;

// 14. Answer "what does Chapter 2 cover and what else is related?"
MATCH (c:Chapter {order: 2})-[:CONTAINS]->(s:Section)
WITH c, collect(s.title) AS sections
MATCH (c)-[:SEMANTICALLY_SIMILAR]-(similar:Chapter)
RETURN c.title AS chapter,
       sections,
       collect(similar.title) AS similar_chapters;

// 15. Node connectivity score (most connected nodes = key concepts)
MATCH (n)-[r]-()
WHERE n:Chapter OR n:Section
WITH n, count(r) AS degree
RETURN n.type, n.title, degree
ORDER BY degree DESC
LIMIT 15;
