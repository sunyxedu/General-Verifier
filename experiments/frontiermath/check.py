"""Mechanical answer checking for the FrontierMath public problems.

Integers: exact match after int parsing. Symbolic answers: sympy
equivalence (simplify(candidate - official) == 0), run in an isolated
`uv run --with sympy` subprocess so the project venv stays lean.
Fail-closed: anything unparseable is wrong, never a pass.
"""
import re
import subprocess


def parse_final(text: str):
    """Extract the last FINAL ANSWER: <expr> line (or None)."""
    if not text:
        return None
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text)
    return m[-1].strip().rstrip(".") if m else None


def matches(candidate: str | None, problem: dict) -> bool:
    if candidate is None:
        return False
    official = problem["answer"]
    if problem["answer_type"] == "integer":
        try:
            return int(candidate.replace(",", "").replace("_", "")) \
                == int(official)
        except ValueError:
            return False
    # symbolic: sympy equivalence, fail-closed on any error
    code = (
        "import sympy\n"
        "from sympy import *\n"
        f"a = sympy.sympify({candidate!r})\n"
        f"b = sympy.sympify({official!r})\n"
        "print('EQUAL' if sympy.simplify(a - b) == 0 else 'DIFFERENT')\n"
    )
    try:
        proc = subprocess.run(
            ["uv", "run", "--no-project", "--with", "sympy", "python", "-c",
             code], capture_output=True, text=True, timeout=600)
        return proc.stdout.strip().endswith("EQUAL")
    except Exception:
        return False
