# Benchmark experiments — General Verifier vs LLM-as-a-Verifier vs stock agents

Three public benchmarks, the same four-way comparison on each. **Every arm
is one prompt to a solver** — the arms differ only in which verification
harness is attached to the same stock Claude Code:

| arm | what runs |
|---|---|
| **gv** | Claude Code + the full **General Verifier** mounted (MCP server + hooks + solver protocol) — verification in the loop, used exactly as a normal user would. |
| **turbo** | Claude Code + **TurboAgent**, the official LLM-as-a-Verifier Claude Code plugin ([llm-as-a-verifier/TurboAgent](https://github.com/llm-as-a-verifier/TurboAgent), arXiv 2607.05391), run as shipped: an API proxy that fans out N parallel candidates per model call and picks the best via the fine-grained logprob verifier + Probabilistic Pivot Tournament — their verification in the loop. |
| **plain** | stock headless Claude Code, no harness. |
| **codex** | stock headless Codex CLI, no harness. |

Every benchmark scores **mechanically** (official harness or official
answers); the GV arm additionally reports certificate verdicts, so the
headline safety number everywhere is **verified & wrong = false
certificates**.

All runners are resumable (recorded units are skipped; Harbor jobs resume
by name). Results live under `experiments/<bench>/results/` (gitignored).

Prerequisites: `uv sync` done, `claude` + `codex` CLIs authenticated,
Docker running (SWE-bench Pro + Terminal-Bench), `uv tool install harbor`
(Terminal-Bench only). On Apple Silicon both docker benchmarks run amd64
images under emulation — slow but supported.

Turbo arm: TurboAgent is auto-cloned to `experiments/vendor/TurboAgent/`
(gitignored) and started per run. It needs `ANTHROPIC_API_KEY` (candidate
generations) **and** `VERTEX_API_KEY` (its verifier scores with Gemini
**logprobs** — Anthropic models cannot serve the verifier side; the
paper's own App. A).

## 1. FrontierMath (official public sample — 12 problems)

The benchmark proper is private (evaluation-by-request at Epoch); the 12
officially published sample problems (10× Tier 1-3, 2× Tier 4, statements +
answers from epoch.ai) are in `frontiermath/problems.json`. Answers check
mechanically (exact integer / sympy equivalence).

```bash
uv run python experiments/frontiermath/run.py --arm gv
uv run python experiments/frontiermath/run.py --arm turbo
uv run python experiments/frontiermath/run.py --arm plain
uv run python experiments/frontiermath/run.py --arm codex
uv run python experiments/frontiermath/summarize.py
```

## 2. SWE-bench Pro (public set, ScaleAI/SWE-bench_Pro)

Agents patch a host checkout of the repo extracted from the **official
per-instance Docker image**; patches are evaluated by the **official
harness** (`scaleapi/SWE-bench_Pro-os`) in those same images.

```bash
uv run experiments/swebench_pro/prep.py --pull   # subset + harness + images
uv run python experiments/swebench_pro/run.py --arm gv
uv run python experiments/swebench_pro/run.py --arm turbo
uv run python experiments/swebench_pro/run.py --arm plain
uv run python experiments/swebench_pro/run.py --arm codex
uv run python experiments/swebench_pro/evaluate.py --arm gv     # official eval
uv run python experiments/swebench_pro/evaluate.py --arm turbo
uv run python experiments/swebench_pro/evaluate.py --arm plain
uv run python experiments/swebench_pro/evaluate.py --arm codex
uv run python experiments/swebench_pro/summarize.py
```

## 3. Terminal-Bench 2.1 (via the official Harbor harness)

Agents run **inside** the task containers; Harbor's own verifiers score.
The gv arm is a custom Harbor agent (`gv_agent.py`) that is exactly the
stock claude-code agent plus the General Verifier shipped into the
container; the turbo arm is the stock claude-code agent talking to a
host-side TurboAgent proxy (`ANTHROPIC_BASE_URL=host.docker.internal`).

```bash
uv run python experiments/terminal_bench/run.py --arm plain
uv run python experiments/terminal_bench/run.py --arm turbo
uv run python experiments/terminal_bench/run.py --arm gv
uv run python experiments/terminal_bench/run.py --arm codex
uv run python experiments/terminal_bench/summarize.py
```
