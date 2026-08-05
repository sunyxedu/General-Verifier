# SWE-bench Pro (public set) — 3-arm comparison

Dataset: `ScaleAI/SWE-bench_Pro` (HuggingFace, split=test, 731 public
instances). Evaluation: the official `scaleapi/SWE-bench_Pro-os` harness
against the official prebuilt per-instance images
(`jefzda/sweap-images:<tag>`, linux/amd64 — emulated on Apple Silicon).

Design: agents produce patches on a **host checkout** extracted from the
official image (`docker cp /app`), so all arms share one native, fast
working setup; the patches are then scored inside the official containers.
GV-arm scaffolding files (`.mcp.json`, `.claude/`, `CLAUDE.md`, `.gv/`) are
kept out of patches via `.git/info/exclude`.

## Run

```bash
uv run experiments/swebench_pro/prep.py --pull     # pick subset (default 12,
                                                   # seed 0, fast-repo pool),
                                                   # clone harness, pull images
uv run python experiments/swebench_pro/run.py --arm gv     # + General Verifier
uv run python experiments/swebench_pro/run.py --arm turbo  # + TurboAgent (official LLM-as-a-Verifier plugin)
uv run python experiments/swebench_pro/run.py --arm plain
uv run python experiments/swebench_pro/run.py --arm codex
uv run python experiments/swebench_pro/evaluate.py --arm gv      # official eval
uv run python experiments/swebench_pro/evaluate.py --arm turbo
uv run python experiments/swebench_pro/evaluate.py --arm plain
uv run python experiments/swebench_pro/evaluate.py --arm codex
uv run python experiments/swebench_pro/summarize.py
```

The turbo arm needs `ANTHROPIC_API_KEY` + `VERTEX_API_KEY` (TurboAgent's
logprob verifier).

Subset selection is deterministic (`--seed`, round-robin over the repo
pool); delete `instances.json` to reselect. Image pull + extraction caches
under `~/.cache/gv-experiments/swebench_pro/`. Evaluation is the slow part
on Apple Silicon (amd64 emulation) — keep `--num_workers` low.
