# FrontierMath (official public sample) — 3-arm comparison

FrontierMath proper is private: Epoch AI evaluates models by request
(math_evals@epoch.ai); there is no public dataset release. What IS official
and public are 12 sample problems (statements, answers, and solutions)
published at <https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems> —
10 from Tiers 1-3 plus 2 from Tier 4. `problems.json` transcribes them
verbatim (raw page HTML kept in `_raw/` for provenance).

Answer checking mirrors Epoch's scheme: 11 answers are exact integers,
1 (the Tier-4 BMO problem) is a closed form checked by sympy equivalence
(`check.py`, fail-closed).

## Run

```bash
uv run python experiments/frontiermath/run.py --arm gv           # GV, one prompt
uv run python experiments/frontiermath/run.py --arm plain --n 3  # stock Claude Code
uv run python experiments/frontiermath/run.py --arm codex        # stock Codex
uv run python experiments/frontiermath/select_best.py                 # LLM-as-a-Verifier best-of-3
uv run python experiments/frontiermath/summarize.py
```

Solvers work in scratch dirs under `/tmp/gv-frontiermath/` and may run
code. All runners resume. Expect very low absolute solve rates — that is
the point of the benchmark; the comparison of interest is judge-selection
accuracy vs certificate honesty (verified&WRONG count) on whatever the
solvers produce.
