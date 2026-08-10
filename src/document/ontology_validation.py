"""ontology_validation.py — measurable ontology-accuracy scoring for the
document graph, on both construction axes.

Axis-1 (structural): scored against the PDF's own embedded outline
(PyMuPDF get_toc()) when present — real ground truth written by whatever
tool produced the PDF, not a heuristic self-check (mirrors light/parser.py's
_usable_toc gate exactly, so "no ground truth" here means the same thing
it means during ingestion). Falls back to structural-invariant checks
(page ranges nest inside parents, siblings don't overlap, no orphans) when
no usable outline exists — e.g. HTML-sourced SEC filings carry none.

Axis-2 (idea-linking): no ground truth exists for semantic edges, so this
uses sampled LLM-judge precision — the same class of measurement
RAGAS-style faithfulness/context-precision metrics use elsewhere in the
industry when no labeled set exists. Sampled, not exhaustive, so it's
cheap to run (see feedback_eval_suite_cost — don't over-run this).

Both report a single 0..1 `score` plus the components behind it, so the
design doc's >=90% ontology-accuracy gate is an actual measured number,
not a slogan. This module is pure scoring logic — no Neo4j/blob/LLM calls
live here, so it's testable with plain dicts; scripts/validate_ontology_
accuracy.py wires it up against a live document.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

ONTOLOGY_ACCURACY_TARGET = 0.90

_WS_RE = re.compile(r"\s+")
_NBSP = "\xa0"

# Looser than Axis-2's entity-dedup threshold (0.92 in semantic/axis2.py) —
# outline titles vs. constructed node titles are normally near-verbatim
# (the parser copies the outline title directly onto the node), but this
# still catches parser-side title cleanup (trailing punctuation, a dropped
# running-header prefix) without being so loose it accepts an unrelated
# section as a match.
_TITLE_FUZZY_THRESHOLD = 0.80
_PAGE_TOLERANCE = 1  # off-by-one indexing tolerance


def _normalize_title(title: str) -> str:
    return _WS_RE.sub(" ", (title or "").replace(_NBSP, " ")).strip().lower()


def _titles_match(a: str, b: str) -> bool:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= _TITLE_FUZZY_THRESHOLD


@dataclass
class Axis1Report:
    logical_doc_id: str
    method: str  # "toc_ground_truth" | "structural_invariants"
    score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    matched: int = 0
    total_ground_truth: int = 0
    total_constructed: int = 0
    mismatches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_doc_id": self.logical_doc_id,
            "method": self.method,
            "score": round(self.score, 4),
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "matched": self.matched,
            "total_ground_truth": self.total_ground_truth,
            "total_constructed": self.total_constructed,
            "mismatches": self.mismatches[:10],
        }


@dataclass
class Axis2Report:
    logical_doc_id: str
    score: float
    edge_precision: Optional[float] = None
    entity_grounding_precision: Optional[float] = None
    sampled_edges: int = 0
    sampled_entities: int = 0
    invalid_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_doc_id": self.logical_doc_id,
            "score": round(self.score, 4),
            "edge_precision": round(self.edge_precision, 4) if self.edge_precision is not None else None,
            "entity_grounding_precision": (
                round(self.entity_grounding_precision, 4)
                if self.entity_grounding_precision is not None
                else None
            ),
            "sampled_edges": self.sampled_edges,
            "sampled_entities": self.sampled_entities,
            "invalid_examples": self.invalid_examples[:10],
        }


# ── Axis-1: structural ──────────────────────────────────────────────────────


def extract_toc_ground_truth(pdf_bytes: bytes) -> Optional[list[tuple[int, str, int]]]:
    """The PDF's own embedded outline, if it has a real chapter/section
    structure — mirrors light/parser.py's _usable_toc gate exactly, so a
    None here means the same thing it means during ingestion: no usable
    outline, fall through to structural_invariants (not a failure)."""
    if fitz is None or not pdf_bytes:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        toc = doc.get_toc()
    except Exception:
        return None
    if len(toc) < 5:
        return None
    if not any(level == 1 for level, _title, _page in toc):
        return None
    return toc


def score_axis1_against_toc(
    constructed: list[dict],  # [{id, title, depth, page_start}, ...] Chapter/Section nodes only
    toc: list[tuple[int, str, int]],
    logical_doc_id: str,
) -> Axis1Report:
    """Precision/recall of constructed Chapter/Section nodes against the
    PDF's own embedded outline. Outline level 1 <-> depth 1 (Chapter);
    outline level >=2 <-> depth >=2 (Section-or-deeper) — the graph only
    has two structural tiers below Document, so an outline with 3+ levels
    collapsing into one Section tier is expected, not a mismatch."""
    gt_entries = [(level, title, page) for level, title, page in toc if title.strip()]
    used_constructed: set[int] = set()
    matched = 0
    mismatches: list[str] = []

    for level, title, page in gt_entries:
        want_top = level == 1
        found = False
        for i, node in enumerate(constructed):
            if i in used_constructed:
                continue
            node_is_top = node.get("depth") == 1
            if node_is_top != want_top:
                continue
            if not _titles_match(node.get("title", ""), title):
                continue
            if abs((node.get("page_start") or 0) - page) > _PAGE_TOLERANCE:
                continue
            used_constructed.add(i)
            matched += 1
            found = True
            break
        if not found:
            mismatches.append(f"no match for outline entry: level={level} title={title!r} page={page}")

    recall = matched / len(gt_entries) if gt_entries else 1.0
    precision = matched / len(constructed) if constructed else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return Axis1Report(
        logical_doc_id=logical_doc_id,
        method="toc_ground_truth",
        score=f1,
        precision=precision,
        recall=recall,
        matched=matched,
        total_ground_truth=len(gt_entries),
        total_constructed=len(constructed),
        mismatches=mismatches,
    )


def _bad_title_reason(title: Optional[str]) -> Optional[str]:
    """Why `title` is not a plausible authored heading, or None if it is.

    Shape-based and document-agnostic on purpose -- it asserts only things
    true of headings in any document, never anything about this corpus's
    vocabulary:

    * a heading is ONE line (a multi-line "title" is wrapped body text or a
      wrapped running header that got joined together);
    * a heading is not a table row (markdown pipes, or nothing but numbers
      and punctuation);
    * a heading is not the parser's own "no heading found here" placeholder;
    * a heading is not a whole sentence.
    """
    text = (title or "").strip()
    if not text:
        return "empty title"
    if "\n" in (title or ""):
        return "multi-line title"
    if text.startswith("|") or text.count("|") >= 2:
        return "table row as title"
    if text == "Preamble":
        return "synthetic Preamble catch-all"
    stripped = re.sub(r"[\d\s.,%$()\-–—/]", "", text)
    if not stripped:
        return "numeric/table fragment as title"
    if len(text.split()) > 20:
        return "sentence-length title"
    return None


def score_axis1_structural_invariants(
    constructed: list[dict],  # [{id, parent_id, depth, page_start, page_end}, ...]
    logical_doc_id: str,
) -> Axis1Report:
    """Fallback when no embedded outline exists (e.g. HTML-sourced SEC
    filings carry none) — scores structural self-consistency instead of
    ground truth: does every child's page range fall inside its parent's,
    do siblings avoid overlapping page ranges, is there no orphan. Each is
    a pass/fail check; score = fraction passing.

    Weaker signal than TOC ground truth (it can't catch a whole document
    collapsed into one giant chapter, since that's internally
    "consistent") but still catches real corruption classes seen live
    before checks like these existed — e.g. the Item-7/7A prefix-collision
    bug and the physics-textbook TOC-outline blowup from a broken
    bookmark (both documented in repo memory)."""
    by_id = {n["id"]: n for n in constructed}
    checks = 0
    passed = 0
    mismatches: list[str] = []

    for node in constructed:
        parent = by_id.get(node.get("parent_id"))
        checks += 1
        if parent is None:
            if (node.get("depth") or 0) <= 1:
                passed += 1
            else:
                mismatches.append(f"orphan node (no parent): {node['id']}")
            continue
        ps, pe = node.get("page_start") or 0, node.get("page_end") or 0
        pps, ppe = parent.get("page_start") or 0, parent.get("page_end") or 0
        if pps <= ps and pe <= ppe:
            passed += 1
        else:
            mismatches.append(
                f"page range outside parent: {node['id']} [{ps}-{pe}] not within "
                f"{parent['id']} [{pps}-{ppe}]"
            )

    # Title quality. The page-range invariants above are self-consistency
    # checks: they cannot tell an authored heading from document furniture,
    # so a document whose structure is badly mis-detected still scores ~100%
    # as long as the bogus sections nest tidily. Verified live on a 264-page
    # 10-K that scored 99.76% here while 95 of its 172 section titles were
    # junk -- wrapped running headers ("Management's Discussion and Analysis
    # of / Financial Condition and Results of Operations", used as 25
    # separate section titles), table data rows ("24 % / 14,703 / 33 % ..."),
    # body sentences, and 49 synthetic "Preamble" catch-alls emitted whenever
    # heading detection found nothing. That structure then poisons entities,
    # edges and retrieval downstream, so the gate must be able to see it.
    for node in constructed:
        if (node.get("depth") or 0) <= 0:
            continue  # the document root's title is its filename, not a heading
        if node.get("title") is None:
            # No title supplied at all — not assessable, so not counted either
            # way. Distinct from an EMPTY title, which is a real defect: this
            # is the "caller didn't fetch titles" case, and scoring it as a
            # failure would silently penalise every caller that only cares
            # about the page-range invariants.
            continue
        checks += 1
        bad = _bad_title_reason(node.get("title"))
        if bad is None:
            passed += 1
        else:
            mismatches.append(f"{bad}: {node['id']} title={(node.get('title') or '')[:60]!r}")

    siblings_by_parent: dict[str, list[dict]] = {}
    for node in constructed:
        siblings_by_parent.setdefault(node.get("parent_id") or "", []).append(node)
    for siblings in siblings_by_parent.values():
        ordered = sorted(siblings, key=lambda n: n.get("page_start") or 0)
        for a, b in zip(ordered, ordered[1:]):
            checks += 1
            if (a.get("page_end") or 0) <= (b.get("page_start") or 0):
                passed += 1
            else:
                mismatches.append(f"overlapping siblings: {a['id']} and {b['id']}")

    score = passed / checks if checks else 1.0
    return Axis1Report(
        logical_doc_id=logical_doc_id,
        method="structural_invariants",
        score=score,
        matched=passed,
        total_ground_truth=checks,
        total_constructed=len(constructed),
        mismatches=mismatches,
    )


def score_axis1(
    constructed: list[dict],
    logical_doc_id: str,
    *,
    pdf_bytes: Optional[bytes] = None,
) -> Axis1Report:
    """Single entry point: uses TOC ground truth when available, otherwise
    falls back to structural invariants. Callers that already resolved
    the ground-truth TOC (or lack of it) can call the two scorers above
    directly instead."""
    toc = extract_toc_ground_truth(pdf_bytes) if pdf_bytes else None
    if toc:
        return score_axis1_against_toc(constructed, toc, logical_doc_id)
    return score_axis1_structural_invariants(constructed, logical_doc_id)


# ── Axis-2: idea-linking (sampled LLM-judge) ────────────────────────────────

_EDGE_JUDGE_PROMPT = """You are auditing a knowledge graph edge for correctness.

Source passage:
{source_text}

Target passage:
{target_text}

Claimed relationship: {rel_type} (shared concept/entity: {shared})

Does the target passage genuinely share this concept/entity with the source
passage, in a way that would help a reader understand one by way of the
other? Answer strictly as JSON: {{"valid": true/false, "reason": "..."}}
"""

_ENTITY_JUDGE_PROMPT = """You are auditing whether an extracted entity is actually
grounded in its source text (not hallucinated).

Source passage:
{source_text}

Extracted entity: {entity}

Is this entity genuinely referenced in the passage above (allowing for
paraphrase/synonym, not requiring an exact substring)? Answer strictly as
JSON: {{"valid": true/false}}
"""


_ENTITY_TYPE_SUFFIX_RE = re.compile(r"\s*\([A-Z]+\)\s*$")


def _entity_centered_window(text: str, needles: list[str], *, window: int = 800) -> str:
    """`window` chars of `text` centered on the earliest occurrence of any
    of `needles`, not just the first `window` chars -- a naive prefix
    truncation shows the judge a passage that may not even contain the
    entity/relationship it's being asked to verify at all, if the real
    mention falls later in a long section. Verified live: a section with a
    genuine, grounded "TCO" (Tengizchevroil) mention at character offset
    1789 was judged "not meaningfully connected" purely because the judge's
    800-char prefix window ended at 800 -- the entity was never shown to
    it. Falls back to the plain prefix when none of the needles are found
    (a paraphrase-only match a substring search can't locate), same
    behavior as before this fix for that harder case.
    """
    if not text:
        return text
    for needle in needles:
        base = _ENTITY_TYPE_SUFFIX_RE.sub("", needle).strip()
        if not base:
            continue
        idx = text.lower().find(base.lower())
        if idx == -1:
            continue
        half = window // 2
        start = max(0, idx - half)
        end = min(len(text), start + window)
        start = max(0, end - window)  # slide back if we hit the end early
        return text[start:end]
    return text[:window]


def _shared_entity_texts(shared: Any) -> list[str]:
    """`shared` is the raw r.properties JSON string/dict off a SHARES_ENTITY
    edge (or another Axis-2 rel's own properties shape) -- extracts the
    entity text list to search for, tolerating any other rel type's
    differently-shaped properties (e.g. SEMANTICALLY_SIMILAR's bare
    {"score": ...} has no entities to center on) by returning []."""
    if isinstance(shared, str):
        try:
            shared = json.loads(shared)
        except Exception:
            return []
    if isinstance(shared, dict):
        entities = shared.get("shared_entities")
        if isinstance(entities, list):
            return [e for e in entities if isinstance(e, str)]
    return []


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return {}
    return {}


def score_axis2_idea_linking(
    edges_sample: list[dict],  # [{source_text, target_text, rel_type, shared}, ...]
    entities_sample: list[dict],  # [{source_text, entity}, ...]
    *,
    provider,
    model: str,
    logical_doc_id: str,
) -> Axis2Report:
    """Sampled LLM-judge precision over Axis-2 edges and extracted
    entities. Fails an item closed (counts as invalid) on a malformed
    judge response or provider error — unlike verification.py's
    deliberate fail-OPEN pattern for a live user-facing answer, an
    ontology-accuracy AUDIT must undercount rather than overcount
    correctness, or the >=90% gate would be meaningless."""
    invalid_examples: list[str] = []

    def _judge(prompt: str) -> bool:
        try:
            resp = provider.chat_completion(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            data = _extract_json(resp.choices[0].message.content or "")
            return bool(data.get("valid", False))
        except Exception:
            return False

    edge_valid = 0
    for e in edges_sample:
        # A shared entity (SHARES_ENTITY) is the actual connecting evidence,
        # so both windows center on it. Edge types with no single shared
        # entity to point at (SAME_CATEGORY's properties are just
        # {cluster_id, signal}; likewise CONTRADICTS/ELABORATES/
        # PREREQUISITE_OF) fall back to each side's OWN entity list --
        # still far better than an arbitrary first-800-char prefix that may
        # land on boilerplate rather than either side's substantive content.
        shared_needles = _shared_entity_texts(e.get("shared"))
        source_needles = shared_needles or (e.get("source_entities") or [])
        target_needles = shared_needles or (e.get("target_entities") or [])
        ok = _judge(
            _EDGE_JUDGE_PROMPT.format(
                source_text=_entity_centered_window(e.get("source_text") or "", source_needles),
                target_text=_entity_centered_window(e.get("target_text") or "", target_needles),
                rel_type=e.get("rel_type", ""),
                shared=e.get("shared", ""),
            )
        )
        if ok:
            edge_valid += 1
        else:
            invalid_examples.append(f"edge {e.get('rel_type')} shared={e.get('shared')!r}")

    entity_valid = 0
    for ent in entities_sample:
        ok = _judge(
            _ENTITY_JUDGE_PROMPT.format(
                source_text=_entity_centered_window(ent.get("source_text") or "", [ent.get("entity", "")]),
                entity=ent.get("entity", ""),
            )
        )
        if ok:
            entity_valid += 1
        else:
            invalid_examples.append(f"entity {ent.get('entity')!r} not grounded")

    edge_precision = edge_valid / len(edges_sample) if edges_sample else None
    entity_precision = entity_valid / len(entities_sample) if entities_sample else None
    parts = [p for p in (edge_precision, entity_precision) if p is not None]
    score = sum(parts) / len(parts) if parts else 1.0

    return Axis2Report(
        logical_doc_id=logical_doc_id,
        score=score,
        edge_precision=edge_precision,
        entity_grounding_precision=entity_precision,
        sampled_edges=len(edges_sample),
        sampled_entities=len(entities_sample),
        invalid_examples=invalid_examples,
    )
