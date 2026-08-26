# General Verifier

An **outcome verifier for agent traces**. A stock agent (Claude Code, Codex) solves a
task N times with nothing attached but a pair of recording hooks; afterwards, each
finished trace is scored in isolation — the verifier reconstructs the **claim graph**
the run implicitly relied on, checks every claim down to mechanical gates, freezes a
**certificate**, and folds it into a rank. Best-of-N selection then happens over the
N certificates. The verifier never touches the generator: no feedback, no protocol in
the solver's context, no change to its distribution — the same contract as an
LLM-as-a-Verifier reward model, with one difference: **the score is backed by
executed checks, not by a model's opinion of the transcript.**

## Why not just ask a model?

An LLM judge reads a trajectory and rates it. That fails in a specific, systematic
way: a run that *confidently reports* wrong actions reads exactly like a run that did
the right thing — the judge sees reporting quality, not task success. This system is
built around three counters to that failure:

1. **Claims, not vibes.** A trace is decomposed into atomic, falsifiable claims with
   explicit dependencies. Every claim is either checked against evidence or its
   status says it wasn't — nothing is silently absorbed into an overall impression.
2. **Two evidence indexes, never mixed.** A claim about a *past action* ("the solver
   ran X and it printed Y") is checked against the **ledger** — the hooks' trusted
   record of every tool call and result. A claim about the *delivered state* ("the
   answer file now contains Z") is checked against the **delivered world** — the
   working tree captured when the run ended. A trace whose claims are all accurate
   past reports proves only that the solver narrates honestly; the delivered-state
   claims are what carry the verdict.
3. **Opinions propose, mechanics decide.** LLM seats compile obligations, pick
   methods, run checks — but every seat's output passes a schema gate, every check
   leaves an auditable artifact, effect claims are re-hashed against the ledger, and
   every fold takes the worst case: a claim's verdict is the weakest of its
   obligations, and what a consumer may *rely on* is the weakest link of the whole
   premise chain — in status, in evidence basis (world over record), and in
   strength (decisive over bounded over analytic). Crashes, timeouts, malformed
   output: FAILED, never a pass. A wrong extraction or a lazy decomposition can
   cost recall; it cannot mint a false `verified`.

## Architecture

![General Verifier architecture](docs/architecture.svg)

Solid lines are the **offline ORM path** (the primary object). The dashed teal
overlay is the **online, interactive mode**: the same graph, kernel and certificate,
but the live solver commits claims itself over MCP (no extractor) and the verdict
flows back to it — feedback that the ORM path deliberately does not have.

Per trace, left to right:

1. **Record.** The solver runs untouched; `PreToolUse`/`PostToolUse` hooks append
   every tool call and result to a raw feed. The run leaves behind the feed, the
   authoritative task statement, a pre-run baseline snapshot, the solver's final
   output, and the delivered working tree.
2. **Ingest.** Feed lines become a seq-validated, append-only event stream —
   deterministic, no LLM.
3. **Extract.** An LLM seat reconstructs the commit events the solver never made:
   atomic claims in topological order, past observations anchored to ledger events
   (`effect:<seq>`), delivered-state claims in the present tense, exactly one
   `final`. The extractor is **outcome-blind**: structural errors (an unresolvable
   reference, no final) bounce back for a bounded retry, but verification verdicts
   never reach it — extraction stays independent of scoring.
4. **Gate.** The extracted graph is replayed LLM-free through the same builder that
   serves every other mode; only a structurally valid graph proceeds.
5. **Verify.** The kernel settles the final claim's dependency closure, one claim at
   a time — in any order, because every local verdict is **conditional**: it answers
   "does this claim hold *given its premises*", never blocking on the premises'
   own verification. The **obligation compiler** (answer-blind, graph-aware)
   assumes every premise true — verified or not — and compiles the **residual**:
   what must hold beyond the premises, the inference gap included ("decoded 11
   records" plus "wrote the file" does not yet give "the file content is
   correct"). An obligation that merely restates a premise verbatim is dropped
   mechanically — the premise edge already carries it. The **decomposition
   critic** attacks the residual by constructing a world where all premises and
   all obligations hold and the claim is still false, sending the compiler back
   to repair (bounded rounds, game record binding both seats). Each obligation
   node then plays the **binding game**: an oracle lists only *sufficient*
   methods, each with a signed evidence **basis** (world | record) and
   **strength** (decisive | bounded | analytic); a discharge executor carries
   the bound method out faithfully (installing missing instruments on the spot);
   a judger audits sufficiency, fidelity, and — for any non-decisive binding —
   whether the degradation from stronger tiers was genuinely forced. Only
   decisive evidence may refute. Executors share a per-trace **evidence
   workspace** (durable building material — a growing Lean project, decoded
   inputs): borrowed work must be re-established by the instrument or declared
   as an **assumed premise**, mechanically gated to the claim's own dependency
   closure (a final admits none). Obligation nodes are content-addressed: the
   same check under two claims is one node, and settled verdicts transfer free —
   but only *effectively* settled ones (see below); a conditional verdict never
   transfers its open assumptions away.
6. **Certify.** Every node carries **two verdicts**: the *local* one (holds,
   given its premises — the fold of its own obligation nodes: one refuted ⇒
   refuted; all hold ⇒ verified, basis and strength folded weakest-link;
   instrument-blocked ⇒ partial) and the *effective* one, computed at read time
   by folding the whole premise chain — weakest link in status, basis and
   strength, with a refuted premise poisoning everything downstream. Nothing is
   rewritten: a premise settled later changes the read-out automatically. The
   **final is exempt** from the premise fold — its obligations are compiled from
   `task.md` alone, independent of the solver's argument, so its verdict is its
   own adjudication (a refuted narration step neither kills nor rescues the
   deliverable). The certificate freezes the final's slice with both verdicts,
   open premises, assumed premises, bases, strengths and evidence disclosed per
   entry; a verified final resting anywhere on non-decisive evidence reads out
   as `supported`, with its bridging assumptions attached.
7. **Rank.** A certificate folds into a lexicographic key — `(verdict band,
   verified fraction of the final claim's task-contract obligations)`. Verdict
   dominates absolutely; only the final contract breaks ties, so
   narration/decomposition padding cannot change the order. Conditional and
   whole-slice counts remain diagnostics only — they never rank. No tuned
   weights anywhere — selection only needs an order.
8. **Select.** Across the N traces: argmax by rank, or verdict-weighted voting when
   answers are comparable. A trace whose extraction fails outright ranks below
   everything — fail-closed, like the rest of the system.

The evidence-strength ladder (decisive / bounded / analytic, strictness-descending
execution, audited degradation, `supported` read-out) is specified in
[docs/evidence-strength-ladder.md](docs/evidence-strength-ladder.md).

## Usage

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12+):

```bash
uv sync
```

### Score a recorded trace

A state dir is what a recorded run leaves behind: `hook_feed.jsonl`, `task.md`,
`baseline/`, `final.txt`, optionally `world/` (the delivered tree).

```bash
# extract + certify + rank; prints the feature dict; exit 0 iff verdict == verified
uv run python -m verifier.cli score --state-dir path/to/.gv [--model m]

# LLM-free: rebuild the claim graph from a trace, print label -> status
uv run python -m verifier.cli replay --events path/to/.gv/events.jsonl

# render the extracted graph for inspection (full claim text, obligations, witnesses)
uv run python script/claim_graph.py --state-dir path/to/.gv -o graph.svg --png
```

Codex traces score through the same pipeline: `verifier/stream/codex.py` rewrites a
Codex rollout log into the hook-feed format — the ledger format is the interface, not
the agent.

### Verifier seat backends

Every LLM seat (extractor, compiler, critic, oracle, discharge, judger) runs through
one of two interchangeable backends:

- **Claude Agent SDK** (default): seats are Claude Code agents with hard tool
  palettes, on the provider configured in `.env` (copy `.env.example`; default
  DeepSeek-V4-Flash, `openrouter`/`custom` Anthropic-compatible endpoints supported).
- **Codex CLI** (`--seats codex` on the experiment scorers): seats run as
  `codex exec` calls on a ChatGPT-subscription model (default `gpt-5.6-luna`), which
  also lets the verifier and an LLM-as-a-Verifier baseline share the same verifier
  model. Same schema gate, same retries; see `verifier/llm/codex_backend.py` for the
  fidelity notes. `--service-tier fast` opts GV scorers into Codex Fast mode;
  standard is the default and is explicitly forced/recorded, so standard and fast
  runs have different config identities and never share derived state. For the
  Codex-backed LLM-as-a-Verifier path, use `--codex-service-tier fast`.

### Interactive mode (online)

The same kernel also serves a live Claude Code session over MCP — the solver commits
claims as it works (`gv.commit` verifies before returning, handing back the failing
obligation and its counterexample), and requests `gv.certificate` at the end. This
mode *does* feed verdicts back into generation, so it is a different object from the
offline verifier above — useful interactively, kept out of the ORM experiments.

```bash
uv run python gv_tui.py     # one-command session: mounts server + hooks + protocol
```

Manual mounting (`.mcp.json`, hooks, solver protocol) is documented in
[protocol/](protocol/); the CI gate is `uv run python -m verifier.cli certify
--events trace.jsonl --state-dir .gv-ci`.

## Experiments

Benchmark harnesses under [experiments/](experiments/) (FrontierMath public sample,
SWE-bench Pro, Terminal-Bench 2.1). The ORM comparison is built on **recorder arms**
(`rec`: stock Claude Code + recording hooks; `rec-codex`: stock Codex, its own
rollout log as the ledger): generate k trials per task, score them offline
(`score.py`, resumable, `--task` to spend verification where selection differs), and
compare selection strategies in `summarize.py` — pass@1, verifier argmax, pass@k
oracle, plus verified-precision (a `verified` certificate on a wrong trace is the
one number that must stay at zero). `lav.py` runs the official
[LLM-as-a-Verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier) selector
over the *same* candidate traces for a head-to-head baseline; TurboAgent (its
Claude Code plugin) is wired as an online arm. See
[experiments/README.md](experiments/README.md).

## Module map

| Module | Contents |
|---|---|
| `verifier/core/` | data contracts: content addressing + status machine + basis/strength lattice (`model.py`), append-only event stream (`events.py`), claim store + obligation nodes + trigger rule + slices + the read-time effective fold (`store.py`) |
| `verifier/stream/` | ingestion: incremental graph building (`builder.py`), online tail (`tail.py`), Codex rollout → hook feed (`codex.py`) |
| `verifier/offline.py` | the ORM path: feed ingestion, outcome-blind extraction with structural retry, certify, feature fold, lexicographic `rank_key` |
| `verifier/selection.py` | best-of-N strategies: argmax, majority vote, verdict-weighted vote, pass@N oracle |
| `verifier/llm/` | LLM seats: agent roles with hard tool palettes (`agents.py`), schema gate + fixture backend (`schema.py`), Codex CLI backend (`codex_backend.py`), provider wiring (`provider.py`), prompt text as a package (`prompts/` — one file per seat, zero domain knowledge by design; `prompts/criteria/` holds the 16 shared rule blocks in 5 topic groups: referent, scope, provenance, task-contract, execution) |
| `verifier/execution/` | the hands: artifact store + ledger instrument (`instruments.py`), discharge seat with immutable workdir snapshots, the durable per-trace evidence workspace, and assumed-premise declarations (`discharge.py`) |
| `verifier/engine/` | verification core: residual compilation, conditional settlement, effective-verdict inheritance gates (`kernel.py`), binding game with the strength ladder (`game.py`), certificate with the two-verdict disclosure (`certificate.py`) |
| `verifier/mcp_server.py` + `cli.py` | entry points: interactive MCP server; `score` / `certify` / `replay` CLI |
| `hooks/` + `protocol/` | recording hooks (record, never intervene); MCP mount example + solver protocol (interactive mode) |
| `script/claim_graph.py` | graph inspector: any state dir → laid-out SVG/PNG with full claim text, obligations, witnesses |
| `experiments/` | benchmark harnesses, recorder arms, offline scorers, LLM-as-a-Verifier baseline |
