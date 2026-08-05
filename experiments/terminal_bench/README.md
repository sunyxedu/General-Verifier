# Terminal-Bench 2.1 — 3-arm comparison via Harbor

Dataset: `terminal-bench/terminal-bench-2-1` (89 public tasks; subset in
`tasks.txt`). Harness: the official Harbor CLI (`uv tool install harbor`);
agents run inside the task containers and Harbor's task verifiers assign
reward, so scoring is entirely the benchmark's own.

The gv arm is `gv_agent.py:GVClaudeCode` — Harbor's stock claude-code agent
plus, at install time: the General Verifier codebase shipped to
`/opt/general-verifier` (uv-synced with a managed Python), the MCP sidecar
registered user-scope via Harbor's own mechanism, Pre/PostToolUse hooks in
the user-level `settings.json`, and the solver protocol as user-level
`CLAUDE.md`. The task instruction gets one appended paragraph pointing at
the protocol; `/gv-state` is downloaded as a trial artifact for
verdict-vs-reward analysis.

## Run

```bash
uv run python experiments/terminal_bench/run.py --arm plain --k 3
uv run python experiments/terminal_bench/run.py --arm gv
uv run python experiments/terminal_bench/run.py --arm codex
uv run experiments/terminal_bench/select_best.py    # judge best-of-3 trials
uv run python experiments/terminal_bench/summarize.py
```

Notes:
- jobs land in `results/jobs/<arm>/` and resume by job name; add
  `--dry-run` to print the harbor command without running.
- `ANTHROPIC_API_KEY` must be exported (Harbor forwards it into the
  container); the codex arm needs codex auth (`CODEX_API_KEY`/OpenAI key
  per Harbor's codex agent).
- first GV trial per task pays a one-time in-container `uv sync` (slow
  under amd64 emulation) — the agent-setup timeout is already multiplied.
- registry task names are org-prefixed (`terminal-bench/fix-git`); run.py
  adds the prefix automatically. Smoke-test the subset cheaply first with
  the oracle agent:
  `harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -o /tmp/tb-oracle $(grep -v '^#' experiments/terminal_bench/tasks.txt | sed 's|^|-i terminal-bench/|')`
