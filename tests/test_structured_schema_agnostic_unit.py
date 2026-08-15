"""
Guard: the structured side must not name any particular dataset's fields.

This repo shipped with Northwind demo data, and its field names leaked into
prompts and helper lists that outlived it. That is not a cosmetic problem:
the text-to-cypher prompt instructed the model to read line-item values off
the relationship (`li.unitPrice`), which against a different schema produced
`sum(li.freight)` over a relationship with no properties and reported a
total of "zero" for a true 2,251,910. A clarification menu offered three
metrics defined in terms of `unitPrice x quantity x (1 - discount)`, none of
which existed, so an easy question could not be answered at all.

So this asserts absence rather than trusting review. Comment lines are
exempt: explaining why a fix exists is the opposite of the problem.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Field and relationship names specific to one dataset. Generic words a real
# schema might also use ("name", "price", "total") are deliberately absent --
# this bans dataset vocabulary, not English.
BANNED = (
    "unitPrice", "companyName", "contactName", "productName", "categoryName",
    "ORDER_CONTAINS", "SUPPLIED_BY", "SHIPPED_TO", "orderDate", "shipCountry",
    "employeeID", "customerID", "supplierID",
)

STRUCTURED_SOURCES = [
    "src/retrieval/structured",
    "src/presentation/structured_planner.py",
    "src/conversation/thread_memory.py",
    "src/prompts/structured_text2cypher.txt",
    "src/prompts/structured_multistep_plan.txt",
]


def _files() -> list[Path]:
    out: list[Path] = []
    for entry in STRUCTURED_SOURCES:
        path = ROOT / entry
        if path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
        elif path.exists():
            out.append(path)
    return out


def _docstring_lines(path: Path) -> set[int]:
    """Line numbers occupied by docstrings.

    Exempt for the same reason comments are: naming the old field in prose
    that explains why a fix exists is the opposite of depending on it. Only
    docstrings are exempt, not every string literal -- a banned name inside
    an ordinary literal (a tuple of column names, say) is exactly the kind of
    hidden coupling this is looking for.
    """
    import ast

    if path.suffix != ".py":
        return set()
    covered: set[int] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            covered.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return covered


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_dataset_specific_field_names(path: Path):
    offenders: list[str] = []
    exempt = _docstring_lines(path)
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if _is_comment(line) or lineno in exempt:
            continue
        for token in BANNED:
            if token in line:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} {token}")
    assert not offenders, (
        "dataset-specific field names in structured code/prompts:\n  "
        + "\n  ".join(offenders)
    )
