# Terminal-Bench 2.1 — 3-arm comparison via Harbor

Dataset: `terminal-bench/terminal-bench-2-1` (89 public tasks; subset in
`tasks.txt`, the complete pinned task-name set in `all89_tasks.txt`). Harness:
the official Harbor CLI (`uv tool install harbor`);
agents run inside the task containers and Harbor's task verifiers assign
reward, so scoring is entirely the benchmark's own.

The gv arm is `gv_agent.py:GVClaudeCode` — Harbor's stock claude-code agent
plus, at install time: the General Verifier codebase shipped to
`/opt/general-verifier` (uv-synced with a managed Python), the gv MCP server
registered user-scope via Harbor's own mechanism, Pre/PostToolUse hooks in
the user-level `settings.json`, and the solver protocol as user-level
`CLAUDE.md`. The task instruction gets one appended paragraph pointing at
the protocol; `/gv-state` is downloaded as a trial artifact for
verdict-vs-reward analysis.

## Run

```bash
uv run python experiments/terminal_bench/run.py --arm plain
uv run python experiments/terminal_bench/run.py --arm turbo  # + TurboAgent (official LLM-as-a-Verifier plugin)
uv run python experiments/terminal_bench/run.py --arm gv     # + General Verifier
uv run python experiments/terminal_bench/run.py --arm codex
uv run python experiments/terminal_bench/summarize.py

# Full 89-task native Terra recording run (five independent traces per task).
uv run python experiments/terminal_bench/run.py \
  --arm rec-codex --tasks-file experiments/terminal_bench/all89_tasks.txt \
  --dataset terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a \
  --model gpt-5.6-terra --effort medium --k 5

# A fresh Fast-mode run (use a new job name; do not mix it into the standard job).
uv run python experiments/terminal_bench/run.py \
  --arm rec-codex --tasks-file experiments/terminal_bench/all89_tasks.txt \
  --model gpt-5.6-terra --effort medium --service-tier fast --k 5 \
  --job-name rec-codex-terra-medium-fast-tb21-all89-k5-r1

# Score Codex GV seats in Fast mode; tier is part of the config fingerprint.
uv run python experiments/terminal_bench/score.py \
  --job rec-codex-terra-medium-fast-tb21-all89-k5-r1 \
  --seats codex --model gpt-5.6-terra --effort medium --service-tier fast
```

The turbo arm starts the TurboAgent proxy on the host; the in-container
Claude Code reaches it via `ANTHROPIC_BASE_URL=http://host.docker.internal:<port>`
(Docker Desktop resolves this natively). Needs `VERTEX_API_KEY`
(TurboAgent's logprob verifier); generations use the built-in DeepSeek key.

Notes:
- jobs land in `results/jobs/<arm>/` and resume by job name; add
  `--dry-run` to print the harbor command without running.
- Codex experiments default to an explicitly forced `standard` tier. Passing
  `--service-tier fast` adds both `-c service_tier="fast"` and
  `--enable fast_mode`; the standard branch similarly forces
  `service_tier="default"` and disables `fast_mode`, so user-level Codex config
  cannot silently change an experiment. Fast mode is expected to be about 1.5x
  faster for GPT-5.6 and consumes 2.5x ChatGPT credits. Keep it under a distinct
  job/config identity.
- `rec` and `rec-codex` resolve the selected dataset packages before launch
  and automatically add safe absolute paths named by public `instruction.md`
  and declared `task.toml` artifacts.  Discovery never reads `tests/` or
  `solution/`; `/app` remains the explicit default, while broad roots such as
  `/`, `/etc`, and `/var/log` are rejected.  Harbor 0.20 only supports
  job-level artifact arguments, so the launcher passes the de-duplicated union
  of the selected tasks' paths.  Every rejected candidate is logged with its
  source and a general reason (for example `url_path`, `broad_root`, or
  `runtime_socket_unsupported`).  `--dry-run` resolves dataset metadata but
  does not download missing task packages; it prints the pinned dataset digest,
  path audit, and final artifact union.
- claude arms run on DeepSeek-V4-Flash: run.py exports the DeepSeek env
  (`verifier/llm/provider.py`) and Harbor forwards `ANTHROPIC_BASE_URL` +
  the key into the container; the codex arm needs codex auth
  (`CODEX_API_KEY`/OpenAI key per Harbor's codex agent).
- first GV trial per task pays a one-time in-container `uv sync` (slow
  under amd64 emulation) — the agent-setup timeout is already multiplied.
- registry task names are org-prefixed (`terminal-bench/fix-git`); run.py
  adds the prefix automatically. Smoke-test the subset cheaply first with
  the oracle agent:
  `harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -o /tmp/tb-oracle $(grep -v '^#' experiments/terminal_bench/tasks.txt | sed 's|^|-i terminal-bench/|')`
