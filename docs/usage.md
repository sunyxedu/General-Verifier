# Usage guide

Start with the [README](../README.md) for installation and a first verification run. This guide covers backend settings, collecting your own runs, inspecting results, and interactive use. Commands assume a POSIX shell and the General-Verifier repository as the current directory unless stated otherwise.

For the reasoning behind the checks and verdicts, read the [documentation](https://hemlix.me/General-Verifier/).

## Backend configuration

The default backend uses the Claude Agent SDK with the provider configured in `.env`:

```bash
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` to use the default provider. For OpenRouter or a custom Anthropic-compatible endpoint, use the corresponding settings in [.env.example](../.env.example). `GV_MODEL` selects the model; environment variables override `.env`.

Alternatively, with an authenticated `codex` CLI on your `PATH`, use `--seats codex` when scoring. This backend uses its own authentication and does not require the SDK provider's API key or `.env` setup.

## Capture and score your own runs

Create one directory per finished attempt:

```text
run/
├── task.md                 # original task and acceptance requirements
├── final.txt               # solver's final response
├── hook_feed.jsonl         # recorded tool calls and results
├── baseline/               # pre-run inputs, when checks need them
├── world_manifest.json     # captured addresses and their scope
└── world/                  # captured deliverables
    └── app/answer.txt       # e.g. the run's /app/answer.txt
```

The [example manifest](../examples/sum/world_manifest.json) maps `/app` to `world/app` and declares that `/app` was captured. An uncaptured path's absence proves nothing. Without `world/`, offline scoring treats the delivered world as unavailable; it does not substitute the scoring machine's current files. Preserve original task inputs in `baseline/` when an independent check needs them.

**Claude Code:** the hooks in [hooks/](../hooks/) record calls and results. The experiment recorder arms also save the task and relevant snapshots; see [experiments/README.md](../experiments/README.md) for the complete collection workflow. Recording hooks alone do not capture the delivered filesystem.

**Codex:** convert a native rollout into the same feed format:

```bash
uv run python -m verifier.stream.codex /path/to/rollout.jsonl /path/to/run
```

The converter writes only `hook_feed.jsonl` and `final.txt`; it cannot recover a filesystem that was never saved. Add the original `task.md` and any required baseline and deliverable capture, then score the completed input directory:

```bash
uv run python -m verifier.cli score --state-dir /path/to/run --seats codex
```

## Read the result

| Verdict | Meaning |
|---|---|
| `verified` | All compiled final obligations held at decisive strength, with at least one world-based check. |
| `supported` | All held, but some evidence was bounded or analytic; inspect its bridging assumptions. |
| `partial` | Some checks were unavailable within the retained evidence boundary, or an advisory (`should`) condition was violated. No mandatory condition was decisively refuted. |
| `unverified` | Unresolved or failed checks prevent a conclusion. A final supported only by records also receives this verdict. |
| `refuted` | An admissible decisive counterexample refutes a mandatory (`must`) condition of the final contract. |

`unscoreable: true` is a separate scoring failure, such as unsuccessful extraction or an invalid input seal. It comes with an `error`, and `verdict` is `null`.

The `score` and `certify` commands exit **0 only for `verified`**. Other verdicts, including `supported`, exit 1; invalid command-line arguments exit 2.

A scored run contains:

| Output | Purpose |
|---|---|
| `score.json` | Verdict, configuration identity, certificate reference, and diagnostic counts. |
| `artifacts/<certificate_id>.json` | Full certificate; other JSON files in `artifacts/` retain the referenced evidence. |
| `events.jsonl` | Source events, extracted commits, and verification history. |
| `extract.json` | Extraction metadata and the input/configuration stamp. |
| `claims.sqlite` | Materialized claim graph and judgments used for resuming work. |
| `verification.json` | Identifies the inputs, source events, pipeline, and configuration behind those cached judgments. |

Run `score` again with the same directory to resume. Changed source inputs, pipeline code, or model configuration invalidate incompatible derived judgments. Adapter-sealed runs whose captured inputs have changed are rejected. Keep captured inputs intact while verification is running.

### Open the certificate

With `RUN_DIR` set to the scored run directory:

```bash
uv run python - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

state = pathlib.Path(sys.argv[1])
score = json.loads((state / "score.json").read_text())
if score.get("certificate_id"):
    certificate = state / "artifacts" / (score["certificate_id"] + ".json")
    print(certificate.read_text())
else:
    print(score.get("error", "No certificate was produced."))
PY
```

## Compare attempts

Score each attempt in its own directory. `verifier.offline.rank_key(score)` orders candidates by:

1. Verdict: `verified` > `supported` > `partial` > `unverified` > `refuted` > unscoreable.
2. Within the same verdict, fewer violated advisory conditions on the final contract.

Counts of passed checks and the length of the supporting argument do **not** break ties. `verifier.selection.argmax_set(rows)` returns every tied winner; `argmax(rows)` returns the first one for callers that need a single selection. These ranks are an ordering, not calibrated probabilities of correctness.

## Inspect and configure a run

```bash
# Inspect scoring options; --model overrides the selected backend's default.
uv run python -m verifier.cli score --help

# Rebuild claim structure from events without calling a model.
uv run python -m verifier.cli replay --events /path/to/run/events.jsonl

# Render the saved graph, checks, and witnesses.
uv run python script/claim_graph.py --state-dir /path/to/run -o graph.svg
```

`replay` reconstructs source claims and dependencies in a fresh store. It does not rerun checks or restore previously verified statuses; use the certificate or graph renderer to inspect saved judgments.

`--max-rounds` controls extraction retries; `--critic-rounds` controls decomposition review. Codex scoring also accepts `--effort` and `--service-tier standard|fast`. These settings are recorded in the configuration identity. Offline concurrency defaults to four claims and eight obligations at a time, bounded by a process-wide limit of 12 model seats; tune `GV_PARALLEL_CLAIMS`, `GV_PARALLEL_OBLIGATIONS`, and `GV_MAX_CONCURRENT_SEATS` to fit your usage budget.

## Interactive mode

Interactive mode exposes `commit`, `verify`, `status`, and `certificate` through MCP. The live solver receives verification feedback as it works. It needs `claude` on your `PATH` and the SDK provider configuration described in [Backend configuration](#backend-configuration).

After repairing a result, submit the claim with `revises` pointing to its old label, even if the wording stays the same. GV creates a new version and invalidates the old checks and any conclusions that reused their evidence. Repeating a commit without `revises` keeps the existing version.

Launch it **from a task project outside this repository**:

```bash
GV_REPO="$(pwd)"                  # run this line from General-Verifier
cd /path/to/your/task-project
uv run --project "$GV_REPO" python "$GV_REPO/gv_tui.py"
```

The launcher keeps its state and configuration under the task project's `.gv/`. It does not capture a pre-task baseline or write `task.md`; without a supplied task statement, completion checks use the committed final claim's wording. See the [solver protocol](../protocol/solver-protocol.md) and [MCP configuration example](../protocol/mcp.example.json) for manual integration.

For a trace that already contains committed claims, `certify` checks its final claim and prints a certificate summary:

```bash
uv run python -m verifier.cli certify \
  --events /path/to/events.jsonl --state-dir /path/to/certificate-state --final F
```
