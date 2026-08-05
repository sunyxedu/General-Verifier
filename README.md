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
2. **Every "yes" must have an owner.** A pass nobody is accountable for is worthless.
   The responsibilities are named: the oracle signs the guarantee that a method
   suffices (and grades its assurance level on a scale of its own devising), the
   executor owns doing it faithfully, and a judger examines the delivered result
   against that promise — feeding its reasons back for redos, at most 3 rounds. Every
   verdict traces back to a named guarantee.

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
3. **Obligations.** A graph-aware agent compiles the claim into concretely checkable
   obligations. It sees the whole claim graph: premises already verified may be relied
   on (obligations can be *residual*), and a condition already proven anywhere is
   reused verbatim instead of re-demanded. Obligations materialize as first-class,
   content-addressed graph nodes: the same check under two claims is **one node**, so
   a settled verdict transfers across claims for free. An obligation whose text
   matches a claim the graph already settled inherits that verdict the same way —
   verified passes, refuted refutes, evidence carried over. There is no count cap:
   however many obligations compile, all are verified.
4. **Binding game** (per obligation node): the oracle lists only *sufficient* methods —
   sufficiency is its own signed guarantee, researched first when the domain is
   unfamiliar — each graded with an assurance level **of the oracle's own devising**
   and listed ascending; execution follows the oracle's order exactly (simplest
   sufficient first, most assured last). The discharge executor carries the method out
   faithfully (a missing instrument is installed on the spot; a truly unobtainable one
   settles the claim at `partial`). A **judger** then examines the delivered result:
   was the bound method faithfully executed and the promised artifact delivered?
   Unfaithful → the judger's reason goes back to the executor for a redo and a blind
   re-judge — a loop of at most **3 rounds**. Exhaustion = failed, with every attempt
   on record.
5. **Settlement** is honest per claim: a claim's status is the fold of its obligation
   nodes — any refuted node → `refuted` (and the downstream cone is hard-killed); all
   verified → `verified`; some verified and some not → `partial`; blocked only by an
   unobtainable instrument → `partial` as well. The
   certificate freezes the slice with each claim's obligation nodes and their evidence
   nested inside, and collapses the same way: refuted > unverified > partial >
   verified.

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
dependencies — `mcp` (MCP server), `claude-agent-sdk` (agent seats), `openai`
(one-shot roles):

```bash
uv sync
```

Credentials: agent seats go through the Claude Agent SDK (`ANTHROPIC_API_KEY` or an
authenticated Claude Code environment); one-shot roles use the OpenAI SDK and honor
`OPENAI_API_KEY` / `OPENAI_BASE_URL` (OpenAI, Anthropic's compat endpoint, local vLLM, …).

### Online — MCP server for a stock Claude Code session

1. **Mount the MCP server** in the solver's project — copy `protocol/mcp.example.json`
   into `.mcp.json` (if the solver project is not this repo, add
   `"--project", "/path/to/General-Verifier"` after `"run"`). The solver gets five
   verbs: `gv.commit`, `gv.verify`, `gv.certificate`, `gv.status`, `gv.branches`.
   Server flags: `--state-dir` (required), `--seats agent|sandbox`, `--model`,
   `--no-game`, `--memory-dir`, `--toolchain-dir`.
2. **Wire the hooks** so tool calls become referenceable effect claims — in the
   solver's `.claude/settings.json`:

   ```json
   {"hooks": {
     "PreToolUse":  [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 hooks/pre_tool_use.py"}]}],
     "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 hooks/post_tool_use.py"}]}]}}
   ```

   Hooks append to a raw feed file (`GV_HOOK_FEED`, default `.gv/hook_feed.jsonl`);
   the server drains it before every MCP verb.
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

## Experiments

Three-way benchmark comparisons (General Verifier vs an LLM-as-a-Verifier
judge vs stock Claude Code / Codex) on FrontierMath's official public
sample, SWE-bench Pro's public set, and Terminal-Bench 2.1 — see
[experiments/README.md](experiments/README.md) for one-command runners.

## Module map

| Module | Contents |
|---|---|
| `verifier/core/` | data contracts: content addressing + status machine (`model.py`), append-only event stream (`events.py`), claim store + obligation nodes + single trigger rule + slices/branches (`store.py`) |
| `verifier/stream/` | ingestion: incremental graph building (`builder.py`), torn-line-tolerant online tail (`tail.py`) |
| `verifier/llm/` | LLM seats: one-shot roles with schema gate (`roles.py`, OpenAI SDK), acting agent roles (`agents.py`, Claude Agent SDK) |
| `verifier/execution/` | the hands: python sandbox + artifact store, fail-closed (`instruments.py`), two discharge seats (`discharge.py`); `strength.py` retained but unused |
| `verifier/engine/` | verification core: trigger entry + settlement + obligation-node reuse + pruning (`kernel.py`), binding game: oracle-ordered sufficient methods + fidelity judger with a bounded redo loop (`game.py`), certificate with nested obligations (`certificate.py`) |
| `verifier/memory.py` | three stores (advisory only) |
| `verifier/mcp_server.py` + `cli.py` | entry points: MCP server + hook ingestion; CI gate |
| `hooks/` + `protocol/` | solver-side PreToolUse/PostToolUse hooks; MCP mount example + solver protocol |
