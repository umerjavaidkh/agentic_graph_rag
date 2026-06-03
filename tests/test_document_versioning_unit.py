"""Offline checks for document versioning (no Neo4j, no PDF parser).

Run: python3 tests/test_document_versioning_unit.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_versioning_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib

    importlib.import_module("src.models")
    # Stub parent package so versioning loads without pulling in parser (fitz).
    doc_pkg = types.ModuleType("src.document")
    doc_pkg.__path__ = [str(ROOT / "src" / "document")]
    doc_pkg.__package__ = "src.document"
    sys.modules["src.document"] = doc_pkg

    spec = importlib.util.spec_from_file_location(
        "src.document.versioning",
        ROOT / "src/document/versioning.py",
        submodule_search_locations=[str(ROOT / "src/document")],
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    mod.__package__ = "src.document"
    sys.modules["src.document.versioning"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_logical_id_from_doc_key(v):
    assert v.resolve_logical_id(Path("Any Name.pdf"), doc_key="Go.Data Manual") == "go.data_manual"


def test_build_revision_plan_ids(v, tmp_path: Path):
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 minimal")
    plan = v.build_revision_plan(
        f, doc_key="corp-policy", version_number=3, content_root_id="doc_x"
    )
    assert plan.logical_id == "corp-policy"
    assert plan.revision_id == "corp-policy:r3"
    assert plan.content_root_id == "corp-policy:r3::doc_x"
    assert len(plan.content_hash) == 64


def test_apply_revision_to_graph_remaps_ids(v, tmp_path: Path):
    from src.models import DKGNode, NodeType

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"same bytes")
    plan = v.build_revision_plan(f, doc_key="k", version_number=1, content_root_id="root")
    root = DKGNode(id="root", type=NodeType.DOCUMENT, title="Doc", text="", order=0)
    sec = DKGNode(id="s1", type=NodeType.SECTION, title="Intro", text="hello", order=1)
    nodes, _ = v.apply_revision_to_graph([root, sec], [], plan)
    assert nodes[0].id == plan.content_root_id
    assert nodes[0].logical_doc_id == "k"
    assert nodes[1].id == f"{plan.revision_id}::s1"
    assert nodes[1].lifecycle_status == "ACTIVE"


def main() -> None:
    v = _load_versioning_module()
    test_resolve_logical_id_from_doc_key(v)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_build_revision_plan_ids(v, tmp)
        test_apply_revision_to_graph_remaps_ids(v, tmp)
    print("document versioning unit checks: OK")


if __name__ == "__main__":
    main()
