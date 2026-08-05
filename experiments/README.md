# Benchmark experiments — General Verifier vs LLM-as-a-Verifier vs stock agents

Three public benchmarks, the same three-way comparison on each:

| arm | what runs |
|---|---|
| **gv** | headless Claude Code with the full General Verifier mounted (MCP sidecar + hooks + solver protocol). The solver gets **one prompt** — the system is used exactly as a normal user would use it, no components isolated. |
| **plain / codex** | stock headless Claude Code / stock Codex CLI, same prompt minus the protocol paragraph. `--n >1` plain samples feed the judge arm. |
| **judge** | LLM-as-a-Verifier-style baseline (arXiv 2607.05391, ported): a text-only judge (K repeated 0-10 scorings) picks the best of N plain candidates; the pick is scored mechanically. |

Every benchmark scores **mechanically** (official harness or official answers);
the GV arm additionally reports certificate verdicts, so the headline safety
number everywhere is **verified & wrong = false certificates**.

All runners are resumable (recorded units are skipped; Harbor jobs resume by
name). Results live under `experiments/<bench>/results/` (gitignored).

Prerequisites: `uv sync` done, `claude` + `codex` CLIs authenticated, Docker
running (SWE-bench Pro + Terminal-Bench), `uv tool install harbor`
(Terminal-Bench only). On Apple Silicon both docker benchmarks run amd64
images under emulation — slow but supported.

## 1. FrontierMath (official public sample — 12 problems)

The benchmark proper is private (evaluation-by-request at Epoch); the 12
officially published sample problems (10× Tier 1-3, 2× Tier 4, statements +
answers from epoch.ai) are in `frontiermath/problems.json`. Answers check
mechanically (exact integer / sympy equivalence).

```bash
uv run python experiments/frontiermath/run.py --arm gv
uv run python experiments/frontiermath/run.py --arm plain --n 3
uv run python experiments/frontiermath/run.py --arm codex
uv run python experiments/frontiermath/select_best.py          # judge best-of-3
uv run python experiments/frontiermath/summarize.py
```

## 2. SWE-bench Pro (public set, ScaleAI/SWE-bench_Pro)

Agents patch a host checkout of the repo extracted from the **official
per-instance Docker image**; patches are evaluated by the **official
harness** (`scaleapi/SWE-bench_Pro-os`) in those same images.

```bash
uv run experiments/swebench_pro/prep.py --pull   # subset + harness + images
uv run python experiments/swebench_pro/run.py --arm gv
uv run python experiments/swebench_pro/run.py --arm plain --n 3
uv run python experiments/swebench_pro/run.py --arm codex
uv run python experiments/swebench_pro/select_best.py          # judge best-of-3
uv run python experiments/swebench_pro/evaluate.py --arm gv
uv run python experiments/swebench_pro/evaluate.py --arm plain
uv run python experiments/swebench_pro/evaluate.py --arm codex
uv run python experiments/swebench_pro/evaluate.py --arm judge
uv run python experiments/swebench_pro/summarize.py
```

## 3. Terminal-Bench 2.1 (via the official Harbor harness)

Agents run **inside** the task containers; Harbor's own verifiers score.
The gv arm is a custom Harbor agent (`gv_agent.py`) that is exactly the
stock claude-code agent plus the General Verifier shipped into the
container.

```bash
uv run python experiments/terminal_bench/run.py --arm plain --k 3
uv run python experiments/terminal_bench/run.py --arm gv
uv run python experiments/terminal_bench/run.py --arm codex
uv run python experiments/terminal_bench/select_best.py        # judge best-of-3
uv run python experiments/terminal_bench/summarize.py
```
