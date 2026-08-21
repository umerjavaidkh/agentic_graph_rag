"""Every relative import in src/ must name a module that exists.

Two did not, and nothing caught them. Splitting api.py into routers rewrote
top-level imports for their new depth, but function-local ones are indented
and were missed, so `from ..unstructured.document.page_report import ...`
inside a route resolved to src.interface.unstructured -- a package that does
not exist.

Neither the test suite nor an import-time check could see it: a function-local
import only executes when that function is called, so the module imports
cleanly and the failure appears as a 500 at request time. The graph inspector
returned "500 Internal Server Error" for page validation and ontology scoring
while every other endpoint looked fine.

Resolving the imports statically catches the whole class in one pass.
"""
import ast
import subprocess
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())


def _tracked_py() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "src/"],
                         capture_output=True, text=True).stdout.split()
    return [f for f in out if f.endswith(".py")]


def _known_modules(files: list[str]) -> set[str]:
    known: set[str] = set()
    for rel in files:
        parts = list(Path(rel).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        known.add(".".join(parts))
        for i in range(1, len(parts)):          # parent packages
            known.add(".".join(parts[:i]))
    return known


def test_every_relative_import_resolves():
    files = _tracked_py()
    assert files, "no source files found"
    known = _known_modules(files)

    broken = []
    for rel in files:
        pkg = list(Path(rel).parts[:-1])
        for node in ast.walk(ast.parse((REPO / rel).read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            # `from ..x import y` in a/b/c.py resolves against a/
            up = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
            target = ".".join(up + ([node.module] if node.module else []))
            if target.startswith("src") and target not in known:
                spec = "." * node.level + (node.module or "")
                broken.append(f"{rel}:{node.lineno} {spec} -> {target}")

    assert not broken, "relative imports naming modules that do not exist:\n  " + "\n  ".join(broken)
