"""The prompts are domain-free by construction, and this test keeps them
so: no seat prompt or axiom may name a tool, format, language, file
layout, benchmark, or the concrete subject of one observed failure. A
rule that needs one of these words is a patch, not a principle — it
belongs in docs/axioms.md as a worked example, never in a prompt."""
from __future__ import annotations

import re

from verifier.llm import prompts

# concrete nouns that mark a domain-bound or case-bound rule
BANNED = (
    # tools, languages, formats
    "git", "grep", "find/glob", "pytest", "numpy", "torch", "pandas",
    "dataframe", "sqlite", "docker", "bash", "shell", "python",
    "sorry", "mathlib", "coq", "json schema", "sha256", "sha-256", "yaml",
    "csv", "html", "javascript", "node_modules", "pip ",
    # paths and layouts
    "/app", "task.md", "baseline/", "manifest.json", "git.json",
    "events.jsonl", "hook_feed", "readme", "dataset card",
    # benchmarks and harnesses
    "terminal-bench", "terminal bench", "harbor", "swe-bench",
    "frontiermath", "codex", "claude",
    # the concrete subjects of once-observed failures
    "write-ahead", "checkpoint", "tokenizer", "token", "chess", "board",
    "forced-outcome", "pid", "socket", "ocr", "contact sheet",
    "bounding box", "crop", "overlay", "eigen", "embedding", "segmentation",
    "database", "row-level", "sum these fields", "merely named",
    "not positively designated", "family_role",
    "version string", "trailing newline", "unit test", "sanitizer",
    "call path", "hugging",
    # paraphrases of 2026-09-23's triggering cases: a rule that only reads
    # right with one of these is a worked example (docs/axioms.md), and a
    # reworded example still is one
    "line terminator", "closing line", "invites work", "disassembl",
    "emulat", "namespace", "container", "printer",
    "proof framework", "analyzer",
)

SURFACES = {
    "AXIOMS": prompts.AXIOMS,
    "OBLIGATIONS_SYSTEM": prompts.OBLIGATIONS_SYSTEM,
    "OBLIGATIONS_HARNESS_ADDENDUM": prompts.OBLIGATIONS_HARNESS_ADDENDUM,
    "OBLIGATIONS_TEMPLATE": prompts.OBLIGATIONS_TEMPLATE,
    "CRITIC_SYSTEM": prompts.CRITIC_SYSTEM,
    "CRITIC_TEMPLATE": prompts.CRITIC_TEMPLATE,
    "CRITIC_FEEDBACK_TEMPLATE": prompts.CRITIC_FEEDBACK_TEMPLATE,
    "ORACLES_SYSTEM": prompts.ORACLES_SYSTEM,
    "ORACLES_TEMPLATE": prompts.ORACLES_TEMPLATE,
    "DISCHARGE_SYSTEM": prompts.DISCHARGE_SYSTEM,
    "DISCHARGE_TEMPLATE": prompts.DISCHARGE_TEMPLATE,
    "JUDGER_SYSTEM": prompts.JUDGER_SYSTEM,
    "JUDGER_TEMPLATE": prompts.JUDGER_TEMPLATE,
    "SUFFICIENCY_SYSTEM": prompts.SUFFICIENCY_SYSTEM,
    "SUFFICIENCY_TEMPLATE": prompts.SUFFICIENCY_TEMPLATE,
    "EXTRACT_SYSTEM": prompts.EXTRACT_SYSTEM,
    "EXTRACT_TEMPLATE": prompts.EXTRACT_TEMPLATE,
    "SCREENER_SYSTEM": prompts.SCREENER_SYSTEM,
    "SCREENER_TEMPLATE": prompts.SCREENER_TEMPLATE,
    "SCRIPT_JUDGER_SYSTEM": prompts.SCRIPT_JUDGER_SYSTEM,
    "SCRIPT_JUDGER_TEMPLATE": prompts.SCRIPT_JUDGER_TEMPLATE,
}

COMPACT_SURFACES = {
    "COMPACT_COMPLETENESS_COMPILER_ADDENDUM":
        prompts.COMPACT_COMPLETENESS_COMPILER_ADDENDUM,
    "COMPACT_COMPLETENESS_CRITIC_ADDENDUM":
        prompts.COMPACT_COMPLETENESS_CRITIC_ADDENDUM,
}


def _hits(text: str):
    low = text.lower()
    out = []
    for word in BANNED:
        # whole-word for short tokens, substring for phrases/paths
        if len(word) <= 5 and word.isalpha():
            if re.search(rf"\b{re.escape(word)}\b", low):
                out.append(word)
        elif word in low:
            out.append(word)
    return out


def test_prompts_name_no_tool_format_benchmark_or_case():
    offenders = {
        name: _hits(text)
        for name, text in {**SURFACES, **COMPACT_SURFACES}.items()
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, offenders


def test_axioms_are_short_and_eight():
    axioms = [prompts.AXIOM_AUTHORITY, prompts.AXIOM_INDEPENDENCE,
              prompts.AXIOM_NECESSITY, prompts.AXIOM_SUFFICIENCY,
              prompts.AXIOM_WITNESS, prompts.AXIOM_TIME,
              prompts.AXIOM_REFERENT, prompts.AXIOM_STRENGTH]
    assert len(axioms) == 8
    for text in axioms:
        assert len(text.split()) <= 240, text[:60]
    assert prompts.AXIOMS == "".join(axioms)


def test_every_seat_composes_only_axioms():
    """A seat may add its own protocol text, but the only cross-seat rule
    blocks are the axioms: no legacy criterion constant survives."""
    import verifier.llm.prompts.criteria as criteria
    exported = {n for n in dir(criteria) if n.isupper()}
    assert exported == {"AXIOM_AUTHORITY", "AXIOM_INDEPENDENCE",
                        "AXIOM_NECESSITY", "AXIOM_SUFFICIENCY",
                        "AXIOM_WITNESS", "AXIOM_TIME", "AXIOM_REFERENT",
                        "AXIOM_STRENGTH", "AXIOMS"}


def test_prompt_corpus_stays_bounded():
    """Prose rules have diminishing returns; growth is the smell of case
    patches. Raise this bound deliberately, never incidentally.
    (Raised from 17,000 when two seats were added — the admission screener
    and the script reader — not when an existing seat grew.)"""
    total = sum(len(t.split()) for t in SURFACES.values())
    assert total <= 24_000, total
