# General Verifier — Test-time Verification

A general-verifier meta-harness: it verifies the claims a solver (e.g. Claude Code)
produces during reasoning — not by "asking the model again", but through an evidence
chain of **obligation compilation → method-binding game → mechanical execution → dual audits**.
Two founding questions: (1) a verifier that only reasons cannot verify claims that require
interacting with the environment; (2) if the verifier uses a verification method you never
intended it to use, "passed" does not mean "correct".

Design document: `Test-time Verification 技术报告.pdf` (v0.3.1, HTML source `report/report.html`).
**Hard constraint**: every prompt template under `verifier/` carries zero domain knowledge —
domain content only emerges at runtime (environment probing, the model's latent knowledge,
memory retrieval); correctness is guarded entirely by mechanical gates
(overarching principle: generation affects efficiency only, never correctness).

## Architecture (fully implemented)

```
solver (stock Claude Code) ──MCP stdio──> sidecar (verifier/sidecar.py)
     └─ hooks ──> hook_feed.jsonl ──tail──> unified event stream events.jsonl (append-only)
                                                  │ builder (incremental graph building)
                                                  v
                claim store (SQLite, content-addressed, state machine, branches/revisions/taint guard)
                                                  │ single trigger rule:
                                                  │ "verify an unverified claim the moment it
                                                  │  gains its first depender"
                                                  v
        kernel ── obligation compilation (agent) ─ coverage audit (leak check) ─ binding game ─ settlement
                     binding game (verifier/game.py, strength-first):
                     oracle enumeration (agent probes the environment) → mechanical ranking
                     by evidence strength
                     → gate 1: discrimination exam (every wrong variant must be rejected)
                     → discharge (agent truly executes, artifacts snapshotted to disk)
                     → gate 2: entailment audit (independent agent re-reviews the artifact)
                     → unreachable/rejected → demote to next-strongest method → failed if exhausted
                                                  v
                certificate: dependency-closure slice + evidence chain + freeze
                verdict ∈ {verified (with weakest level), unverified (with reason), refuted (with counterexample)}
```

- **Dual-track SDK**: roles that act (probe/execute/audit) = Claude Agent SDK agents
  (`verifier/agents.py`, official `output_format` structured output + hard `tools` panel);
  plain one-shot LLM calls = OpenAI SDK (`verifier/roles.py`, provider-agnostic).
- **Level honesty**: mechanical sandbox execution earns `demonstrated`; agent self-reports
  are capped at `corroborated`; the entailment audit assigns the level and the evidence
  type sets the ceiling (formal→proven / execute→demonstrated / text→corroborated) —
  take the weaker of the two; overall = the weakest link of the slice.
- **Search support (M3)**: the branch is part of a claim's identity; `refuted` hard-kills
  the downstream cone (event `branch_pruned`, advisory — soundness rests on the taint
  guard); identical conclusions reached across branches are deduplicated and reuse evidence.
- **Memory (M4)**: three stores — obligation patterns / attack patterns / instrument
  registry — with mechanical retrieval (bigram Jaccard), injected only into prompt
  slots: hits make the system faster, and can never let a wrong claim verify as right.

## Validity experiment (experiments/results/README.md)

12 true/false math claims, verifier arm vs same-model direct-judge arm: the verifier
scored **0 false passes / 0 false alarms / 0 abstentions, 12/12** — all 6 false claims
refuted with counterexamples, all 6 true claims verified (4 of them proven: Lean
formalization / sympy recomputation that passed the entailment audit). The three
mechanism fixes — obligation-pattern transport in the discrimination exam, exam-question
audit with reject-and-regenerate, and the world-checkable obligation contract — were all
driven out by this experiment.

## Usage

```bash
# Tests (120 offline; live acceptance with GV_LIVE=1 spends real tokens)
.venv/bin/python -m pytest
GV_LIVE=1 .venv/bin/python -m pytest tests/test_agents_live.py tests/test_sidecar_live.py

# sidecar: mount from the solver side via .mcp.json
# {"mcpServers": {"gv": {"type": "stdio", "command": ".venv/bin/python",
#   "args": ["-m", "verifier.sidecar", "--state-dir", ".gv", "--memory-dir", ".gv-mem"]}}}
# solver protocol text (put into CLAUDE.md, advisory): protocol/solver-protocol.md
# hooks that capture tool events: hooks/pre_tool_use.py, hooks/post_tool_use.py

# CI certification gate (consumer-side enforcement: exit code 1 unless the verdict is verified)
.venv/bin/python -m verifier.cli certify --events trace.jsonl --state-dir .gv-ci
.venv/bin/python -m verifier.cli replay --events trace.jsonl   # LLM-free state replay

# Validity experiment (false-pass rate vs LLM direct judge)
.venv/bin/python experiments/false_pass.py --out experiments/results/run1
```

## Module map

| Module | Report section | Contents |
|---|---|---|
| `verifier/model.py` | §14 | content addressing, Claim/Obligation/Evidence, levels |
| `verifier/events.py` | 5 | unified event stream (append-only JSONL, seq validation) |
| `verifier/store.py` | 10/12/13 | claim store, single trigger rule, slices, branches/dead branches |
| `verifier/builder.py` + `tail.py` | 4/6/3 | incremental graph building; torn-line-tolerant online tail |
| `verifier/roles.py` | 15 | one-shot roles (OpenAI SDK), schema gate |
| `verifier/agents.py` | 15 | agent roles (Claude Agent SDK), five seats |
| `verifier/instruments.py` | 25/26 | python sandbox (fail-closed), audit instruments, artifact store |
| `verifier/discharge.py` | 20 | two discharge seats: sandbox / agent |
| `verifier/game.py` | 17-19/21 | strength-first binding game, two gates, demotion |
| `verifier/coverage.py` | 22 | leak check: claim-level variants test the obligation set, fill-in obligations |
| `verifier/kernel.py` | 14/20/21 | trigger entry, taint guard, settlement, dedup, pruning |
| `verifier/certificate.py` | 13/23/33 | settlement residue, freeze, honest verdict |
| `verifier/sidecar.py` + `cli.py` | §11 | MCP service, hook ingestion, CI gate |
| `verifier/memory.py` | §13 | three stores (advisory only) |
