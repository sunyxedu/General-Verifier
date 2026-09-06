<div align="center">

# General Verifier

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-8B5CF6)](https://github.com/sunyxedu/General-Verifier)
[![Status: experimental](https://img.shields.io/badge/status-experimental-F59E0B)](#project-status-and-limitations)

[Quick start](#quick-start) · [Architecture](#architecture) · [How it works](#how-it-works) · [Verdicts](#reading-the-result) · [Your own traces](#score-your-own-traces) · [Development](#development)

</div>

General Verifier (GV) takes a **finished agent run**: the original task, recorded tool calls and results, the agent's final answer, and any captured deliverables. It turns the task into explicit checks, uses model agents to carry them out and audit the evidence, and saves a certificate explaining what passed, what failed, and what remains unknown. You can inspect one run or use the verdicts to select among several attempts at the same task.

For example, suppose the task is “compute 1 + 2 + 3 + 4 + 5 and save the result in `/app/answer.txt`.” A transcript saying “done” tells you what the agent reported. Checking success requires computing the expected sum from the task and comparing it with the **captured answer file**. GV is built to make that distinction explicit.

The [usage guide](docs/usage.md) covers capture, configuration, and integration. The [technical report](https://hemlix.me/General-Verifier/) introduces the verification workflow and links to focused explanations of the design, implementation, results, limitations, and evaluation. A runnable example is included in Getting started. [Markdown source](docs/report/report.md) and [publishing instructions](docs/report/README.md) are also available; report changes on `main` publish automatically.

> [!IMPORTANT]
> A `verified` certificate means GV accepted the compiled task checks and their evidence. Models still interpret the task, propose checks, and judge whether those checks are sufficient. A mistaken interpretation or audit can produce a false verdict. The certificate makes the judgment inspectable; it is not a guarantee of correctness.

## Architecture

![GV architecture: captured task, trace and deliverables flow through extraction, task checks, execution and audit, then a certificate and optional selection](docs/architecture.svg)

<div align="center"><a href="docs/architecture.svg">Open the diagram at full size</a></div>

The main workflow is **offline**: the solver finishes before verification starts, and receives no verifier feedback. An optional [interactive mode](#interactive-mode) uses the same verification engine while the solver is working.

## Quick start

### Install

You need Python **3.12+**, [uv](https://docs.astral.sh/uv/), and access to one of the model backends below. These commands use a POSIX shell.

```bash
git clone https://github.com/sunyxedu/General-Verifier.git
cd General-Verifier
uv sync
uv run python -m verifier.cli --help
```

Help and graph replay require no model credentials. Scoring runs model agents, can execute commands and install verification tools, and consumes model usage.

### Configure a backend

The default backend uses the Claude Agent SDK with the provider configured in `.env`:

```bash
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` to use the default provider. For OpenRouter or a custom Anthropic-compatible endpoint, use the corresponding settings in [.env.example](.env.example). `GV_MODEL` selects the model; environment variables override `.env`.

Alternatively, with an authenticated `codex` CLI on your `PATH`, use `--seats codex` when scoring. This backend uses its own authentication and does not require the SDK provider's API key or `.env` setup.

### Verify a small captured run

[examples/sum/](examples/sum/) is a hand-written example of a finished run. Its task asks for the sum of the integers from 1 through 5; its delivered file contains `15`. The example includes the tool record, final message, and capture manifest. It needs no pre-run files because all mathematical inputs are in the task.

Copy it to a separate run directory, then score it:

```bash
RUN_DIR="$(mktemp -d)"
cp -R examples/sum/. "$RUN_DIR/"

uv run python -m verifier.cli score --state-dir "$RUN_DIR"
```

For the Codex backend, replace the last command with:

```bash
uv run python -m verifier.cli score --state-dir "$RUN_DIR" --seats codex
```

Scoring prints a JSON summary and saves it as `score.json`. A successful verification includes these fields; the precise checks and verdict depend on the model run:

```json
{
  "unscoreable": false,
  "verdict": "verified",
  "verdict_strength": "decisive",
  "certificate_id": "<content hash>"
}
```

Open the full certificate, including each check and its evidence references:

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

To try a wrong answer, make a **fresh copy** of the example, replace `world/app/answer.txt` with `14`, and score that directory. A check that recomputes the sum should refute the delivered answer even though the recorded success message is unchanged.

## How it works

GV represents an agent's statements as **claims**. Each claim is decomposed into **obligations**: concrete conditions that must be checked to establish it.

1. **Read the run.** Recorded tool events become an append-only ledger. An extractor reconstructs the claims and their dependencies. Invalid references or a missing final claim trigger a bounded structural retry; verification verdicts are never fed back to the extractor.
2. **Define what success requires.** The compiler derives the final claim's obligations from `task.md`, independently of the solver's argument. A critic challenges whether the proposed checks are both required by the task and sufficient to establish success. If no task is supplied, GV can only check the final claim's own wording.
3. **Choose and run checks.** For each obligation, an oracle proposes methods and declares their evidence basis and strength. An executor carries out the method and saves artifacts. A separate judger audits whether the method was sufficient, whether it was followed, and whether any weaker evidence was justified.
4. **Apply the evidence rules.** The kernel validates structured outputs, enforces declared evidence requirements, and records failures and counterexamples. Missing evidence or a failed check cannot count as a pass. An absence witness can refute only a condition declared to require presence.
5. **Freeze the result.** The certificate records the final verdict, individual obligations, evidence, assumptions, and unresolved work. The supporting argument is retained for inspection; the final verdict comes from the independently checked final contract.

Two distinctions matter when inspecting the evidence:

| Question | What GV records |
|---|---|
| **What was checked?** | `record`: what a retained tool result reports. `world`: the captured deliverable or an independent computation. A log saying a file was written does not establish its final contents. |
| **How much does the check establish?** | `decisive`: declared to settle the obligation. `bounded` or `analytic`: weaker support with explicit bridging assumptions. These declarations are audited by models. |

Intermediate claims may be checked **assuming their stated premises**. Their effective status also accounts for those premises, so an open premise remains visible. Reuse is scoped to the obligation's evidence requirement, absence rule, and permitted premises. The final claim admits no assumed premises: earlier conclusions must not become unearned proof of task completion.

For the detailed contracts, see the [evidence axioms](docs/axioms.md) and [strength ladder](docs/evidence-strength-ladder.md).

## Reading the result

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

### Select among attempts

Score each attempt in its own directory. `verifier.offline.rank_key(score)` orders candidates by:

1. Verdict: `verified` > `supported` > `partial` > `unverified` > `refuted` > unscoreable.
2. Within the same verdict, fewer violated advisory conditions on the final contract.

Counts of passed checks and the length of the supporting argument do **not** break ties. `verifier.selection.argmax_set(rows)` returns every tied winner; `argmax(rows)` returns the first one for callers that need a single selection. These ranks are an ordering, not calibrated probabilities of correctness.

## Score your own traces

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

The [example manifest](examples/sum/world_manifest.json) maps `/app` to `world/app` and declares that `/app` was captured. An uncaptured path's absence proves nothing. Without `world/`, offline scoring treats the delivered world as unavailable; it does not substitute the scoring machine's current files. Preserve original task inputs in `baseline/` when an independent check needs them.

**Claude Code:** the hooks in [hooks/](hooks/) record calls and results. The experiment recorder arms also save the task and relevant snapshots; see [experiments/README.md](experiments/README.md) for the complete collection workflow. Recording hooks alone do not capture the delivered filesystem.

**Codex:** convert a native rollout into the same feed format:

```bash
uv run python -m verifier.stream.codex /path/to/rollout.jsonl /path/to/run
uv run python -m verifier.cli score --state-dir /path/to/run --seats codex
```

Between conversion and scoring, supply the original `task.md` and any required baseline and deliverable capture. The converter writes only `hook_feed.jsonl` and `final.txt`; it cannot recover a filesystem that was never saved.

### Inspect or configure a run

```bash
# Inspect scoring options; --model overrides the selected backend's default.
uv run python -m verifier.cli score --help

# Rebuild claim structure from events without calling a model.
uv run python -m verifier.cli replay --events /path/to/run/events.jsonl

# Render the saved graph, checks, and witnesses.
uv run python script/claim_graph.py --state-dir /path/to/run -o graph.svg
```

`replay` reconstructs source claims and dependencies in a fresh store. It does not rerun checks or restore previously verified statuses; use the certificate or graph renderer to inspect saved judgments.

`--max-rounds` controls extraction retries; `--critic-rounds` controls decomposition review. Codex scoring also accepts `--effort` and `--service-tier standard|fast`. These settings are recorded in the configuration identity. Offline concurrency defaults to four claims and four obligations at a time, bounded by a process-wide limit of 12 model seats; tune `GV_PARALLEL_CLAIMS`, `GV_PARALLEL_OBLIGATIONS`, and `GV_MAX_CONCURRENT_SEATS` to fit your usage budget.

## Interactive mode

Interactive mode exposes `commit`, `verify`, `status`, and `certificate` through MCP. The live solver receives verification feedback as it works. It needs `claude` on your `PATH` and the SDK provider configuration described above.

After repairing a result, submit the claim with `revises` pointing to its old label, even if the wording stays the same. GV creates a new version and invalidates the old checks and any conclusions that reused their evidence. Repeating a commit without `revises` keeps the existing version.

Launch it **from a task project outside this repository**:

```bash
GV_REPO="$(pwd)"                  # run this line from General-Verifier
cd /path/to/your/task-project
uv run --project "$GV_REPO" python "$GV_REPO/gv_tui.py"
```

The launcher keeps its state and configuration under the task project's `.gv/`. It does not capture a pre-task baseline or write `task.md`; without a supplied task statement, completion checks use the committed final claim's wording. See the [solver protocol](protocol/solver-protocol.md) and [MCP configuration example](protocol/mcp.example.json) for manual integration.

For a trace that already contains committed claims, `certify` checks its final claim and prints a certificate summary:

```bash
uv run python -m verifier.cli certify \
  --events /path/to/events.jsonl --state-dir /path/to/certificate-state --final F
```

## Project status and limitations

GV is experimental. The deterministic tests exercise evidence rules, provenance, cache invalidation, and verdict folding; they do not measure model accuracy or establish a zero false-positive rate. Interpretation, decomposition coverage, method sufficiency, and audit quality remain model-dependent.

A capture only supports checks within the evidence it retained. Truncated tool output, missing pre-run inputs, uncaptured services, or unavailable instruments can leave a task unresolved. Content hashes identify retained bytes; they do not establish that the original capture was truthful or complete. Verification agents execute commands and may access the network; use an appropriate execution environment for the data and tools in your task.

Benchmark collection, baseline comparisons, and selection reports live under [experiments/](experiments/README.md). Evaluate false `verified` verdicts against independent ground truth alongside selection accuracy and verification cost.

## Development

```bash
uv sync
uv run pytest -q
```

The default test command collects only `tests/`, excluding downloaded benchmark runs and their artifacts.

| Location | Responsibility |
|---|---|
| `verifier/offline.py`, `verifier/selection.py` | Finished-run scoring, cache identity, and candidate selection. |
| `verifier/core/`, `verifier/stream/` | Data contracts, evidence scope, event ingestion, and claim graph. |
| `verifier/engine/` | Compilation, critique, verification, and certificates. |
| `verifier/execution/`, `verifier/llm/` | Check execution, retained artifacts, model roles, prompts, and backends. |
| `verifier/cli.py`, `verifier/mcp_server.py` | CLI and interactive entry points. |
| `hooks/`, `protocol/`, `experiments/` | Trace recording, solver integration, and evaluation harnesses. |
