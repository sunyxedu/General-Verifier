# General Verifier

A test-time verification harness for LLM solvers (e.g. Claude Code). It verifies the
claims a solver produces **during** reasoning — not by asking a model for a second
opinion, but by forcing every claim through an evidence chain that ends in mechanical
gates, and issuing a certificate a consumer (you, or CI) can refuse to merge without.

## Why

Two observations drive the design:

1. **Opinions are not evidence.** A verifier that only reasons cannot check claims that
   require touching the environment — running code, probing what is installed. Roles
   that must act are agents with tools, not text completions.
2. **"Passed" must mean what you think it means.** If a claim is checked by a method
   that cannot tell right from wrong, a pass is worthless. So every method must first
   pass an exam: it has to reject known-wrong variants of the claim before its verdict
   on the real claim counts.

Master principle: **LLM generation affects efficiency only, never correctness.** Every
prompt template under `verifier/` carries zero domain knowledge — domain content emerges
at runtime (environment probing, the model's knowledge, memory retrieval) — and
correctness is guarded entirely by mechanical gates. A crash, timeout, or unparseable
output is always a FAILED check, never a pass (fail-closed).

## How it works

![General Verifier architecture](docs/architecture.svg)

1. **Commit.** The solver commits claims (`assertion` / `final`) over MCP; hooks turn
   its tool calls into `effect` claims. Everything is one append-only JSONL stream —
   replay of that file is the whole offline mode.
2. **Trigger.** One scheduling rule: an unverified claim is verified the moment another
   claim first depends on it.
3. **Obligations.** An agent compiles the claim into falsifiable obligations (each must
   state what failure would look like). A coverage audit attacks the obligation *set*:
   claim-level wrong variants that no obligation rejects reveal holes, and a patch role
   fills them in.
4. **Binding game** (per obligation): an oracle agent probes the environment and
   proposes candidate check methods; each candidate must survive
   - **gate 1 — discrimination exam**: it must reject every known-wrong variant of the
     claim (variants are self-confirmed false, with disproofs on record);
   - **discharge**: the method actually runs; artifacts are snapshotted to a
     content-addressed store; a fabrication detector independently re-runs any
     extractable check script in a sandbox and rejects the method on disagreement;
   - **gate 2 — entailment audit**: an independent agent opens the artifacts and judges
     whether the evidence really discharges the obligation.
   A rejected/unreachable method demotes to the next candidate; exhaustion = failed,
   with every attempt on record.
5. **Settlement** is honest per claim: any counterexample → `refuted` (and the
   downstream cone of a refuted claim is hard-killed); all obligations hold →
   `verified`; a mix → `partial`. The certificate collapses its slice the same way:
   refuted > unverified > partial > verified.

Design notes:

- **Dual-track SDK**: roles that act (probe / execute / audit) are Claude Agent SDK
  agents with structured output and a hard tool panel (`verifier/llm/agents.py`); pure
  text-in/JSON-out roles are one-shot OpenAI SDK calls, provider-agnostic
  (`verifier/llm/roles.py`).
- **Search support**: the branch is part of a claim's identity; solvers can explore
  hypotheses in parallel, `refuted` kills a branch's downstream cone, and identical
  conclusions reached on different branches deduplicate and reuse evidence.
- **Memory is advisory only**: three stores (obligation patterns / attack patterns /
  instrument registry) feed prompt slots via deterministic retrieval. Hits make the
  system faster; they can never make a wrong claim verify.

## Usage

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12+). Three third-party
dependencies — `mcp` (sidecar), `claude-agent-sdk` (agent seats), `openai`
(one-shot roles):

```bash
uv sync
```

Credentials: agent seats go through the Claude Agent SDK (`ANTHROPIC_API_KEY` or an
authenticated Claude Code environment); one-shot roles use the OpenAI SDK and honor
`OPENAI_API_KEY` / `OPENAI_BASE_URL` (OpenAI, Anthropic's compat endpoint, local vLLM, …).

### Online — sidecar to a stock Claude Code session

1. **Mount the MCP server** in the solver's project — copy `protocol/mcp.example.json`
   into `.mcp.json` (if the solver project is not this repo, add
   `"--project", "/path/to/General-Verifier"` after `"run"`). The solver gets five
   verbs: `gv.commit`, `gv.verify`, `gv.certificate`, `gv.status`, `gv.branches`.
   Sidecar flags: `--state-dir` (required), `--seats agent|sandbox`, `--model`,
   `--no-game`, `--no-coverage`, `--memory-dir`, `--toolchain-dir`.
2. **Wire the hooks** so tool calls become referenceable effect claims — in the
   solver's `.claude/settings.json`:

   ```json
   {"hooks": {
     "PreToolUse":  [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 hooks/pre_tool_use.py"}]}],
     "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 hooks/post_tool_use.py"}]}]}}
   ```

   Hooks append to a raw feed file (`GV_HOOK_FEED`, default `.gv/hook_feed.jsonl`);
   the sidecar drains it before every MCP verb.
3. **Give the solver the protocol** — put `protocol/solver-protocol.md` into the
   solver's CLAUDE.md. It is advisory: a solver that ignores it simply gets no
   certificate, never a wrong one.

### Offline — CI certificate gate

```bash
# exit 0 iff the verdict is `verified` (partial/unverified/refuted → 1, no final claim → 2)
uv run python -m verifier.cli certify --events trace.jsonl --state-dir .gv-ci \
    [--final F] [--seats agent|sandbox] [--model m] [--no-game]

# LLM-free: rebuild the claim graph from a trace, print label -> status
uv run python -m verifier.cli replay --events trace.jsonl
```

Enforcement is inverted: the solver is never forced to verify — the consumer refuses to
merge without a certificate.

## Module map

| Module | Contents |
|---|---|
| `verifier/core/` | data contracts: content addressing + status machine (`model.py`), append-only event stream (`events.py`), claim store + single trigger rule + slices/branches (`store.py`) |
| `verifier/stream/` | ingestion: incremental graph building (`builder.py`), torn-line-tolerant online tail (`tail.py`) |
| `verifier/llm/` | LLM seats: one-shot roles with schema gate (`roles.py`, OpenAI SDK), acting agent roles (`agents.py`, Claude Agent SDK) |
| `verifier/execution/` | the hands: python sandbox + artifact store, fail-closed (`instruments.py`), two discharge seats (`discharge.py`), fabrication detector (`strength.py`) |
| `verifier/engine/` | verification core: trigger entry + settlement + dedup + pruning (`kernel.py`), binding game with two gates (`game.py`), coverage leak check (`coverage.py`), certificate + honest verdict (`certificate.py`) |
| `verifier/memory.py` | three stores (advisory only) |
| `verifier/sidecar.py` + `cli.py` | entry points: MCP service + hook ingestion; CI gate |
| `hooks/` + `protocol/` | solver-side PreToolUse/PostToolUse hooks; MCP mount example + solver protocol |
