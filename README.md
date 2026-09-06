<h1 align="center">General Verifier</h1>

<p align="center"><strong>Test-time verification for AI agents</strong></p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/usage.md">Usage guide</a> ·
  <a href="https://hemlix.me/General-Verifier/">Documentation</a>
</p>

General Verifier (GV) checks whether an **LLM agent's result satisfies its task**. It reads the original instructions, recorded tool calls and results, final response, and captured files, then uses verification agents to investigate whether the task was completed. The output is a verdict linked to the checks and evidence behind it.

## How it works

Suppose an agent is asked to compute a sum and save it to a file. A successful write does not establish that the answer is correct. Verification needs to derive the expected value from the task and compare it with the file the agent actually delivered.

[![General Verifier architecture: the task and captured agent run lead to reviewed requirements, tool-based verification and evidence audit, then a certificate. Review feedback can revise checks or verification methods.](docs/architecture.svg)](docs/architecture.svg)

GV first reconstructs the agent's claims and turns the original task into explicit completion checks. A reviewer challenges whether those checks cover the requirements without adding new ones. For each check, verification agents choose a method, execute it with tools, and audit whether the resulting evidence supports the conclusion.

**Task requirements define success.** The final checks must stand without assuming the agent's intermediate claims are true. A certificate preserves the checks, their evidence, and unresolved issues so you can inspect how the verdict was reached. The [documentation](https://hemlix.me/General-Verifier/) explains the design and implementation, starting with the [architecture](https://hemlix.me/General-Verifier/architecture.html).

## Installation

You need Python **3.12+**, [uv](https://docs.astral.sh/uv/), and access to a model backend. From a POSIX shell:

```bash
git clone https://github.com/sunyxedu/General-Verifier.git
cd General-Verifier
uv sync
```

The default backend uses the Claude Agent SDK. Copy the provider configuration, then set `DEEPSEEK_API_KEY` in `.env`:

```bash
cp .env.example .env
```

If you already have an authenticated Codex CLI on your `PATH`, you can skip the `.env` setup and add `--seats codex` to the scoring command below. Other providers and model settings are covered in [backend configuration](docs/usage.md#backend-configuration).

## Quickstart

The [sum example](examples/sum/) is a small, hand-written capture of a finished agent run. Its task asks for the sum of the integers from 1 through 5, saved as an integer followed by a newline in `/app/answer.txt`. The captured file, `world/app/answer.txt`, contains `15` followed by a newline.

Copy the example to a fresh directory and run verification. This calls the selected model backend and executes verification commands.

```bash
RUN_DIR="$(mktemp -d)"
cp -R examples/sum/. "$RUN_DIR/"
uv run python -m verifier.cli score --state-dir "$RUN_DIR"
```

Scoring prints a JSON summary and saves it in `$RUN_DIR/score.json`. For a successful verification, the relevant fields look like this:

```json
{
  "verdict": "verified",
  "verdict_strength": "decisive",
  "certificate_id": "<content hash>"
}
```

Use the `certificate_id` to open `$RUN_DIR/artifacts/<certificate_id>.json` and inspect the checks and evidence. If verification cannot produce a certificate, the summary records an error instead. Checks and verdicts depend on the model run; see [reading the result](docs/usage.md#read-the-result) for the possible outcomes.

To explore what is being checked, make another fresh copy and replace the captured answer with `14`. Recomputing the sum should expose the mismatch, even though the recorded success message is unchanged.

## Verify your own runs

Save the original task, tool trace, final response, and delivered files in one directory per attempt. Preserve pre-run inputs when they are needed to establish the expected result. The [capture guide](docs/usage.md#capture-and-score-your-own-runs) explains the directory layout and how to collect Claude Code or Codex runs.

Once the capture is ready, use the same command:

```bash
uv run python -m verifier.cli score --state-dir /path/to/run
```

For agent evaluation or test-time selection, score each attempt separately and [compare their verdicts](docs/usage.md#compare-attempts). The evidence remains available when you need to understand why one attempt ranked above another.

## Project status

GV is experimental. Models interpret requirements, propose checks, and audit evidence, so a certificate is an inspectable judgment, not a guarantee of correctness. Missing inputs or incomplete captures can leave a task unresolved. See [scope and limitations](https://hemlix.me/General-Verifier/limitations.html) for the trust assumptions and how to measure reliability and cost.

## Development

After installation, run the tests with `uv run pytest -q`. The [code map](https://hemlix.me/General-Verifier/reference.html#code-map) connects the design to its implementation; [experiments/](experiments/README.md) contains the benchmark collection and comparison workflows.
