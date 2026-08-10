# Verification-driven search protocol (solver side, advisory)

Put this text into the solver's CLAUDE.md (or an appendix to the system prompt). It
specifies **mechanism** only and contains no domain knowledge; soundness does not depend
on it — a solver that ignores the protocol simply gets no certificate, never a wrong one.

## Submission discipline

- Whenever you reach a conclusion that later reasoning will **depend on**, immediately
  `gv.commit` it:
  - `kind`: `assertion` (intermediate conclusion) | `final` (final answer)
  - `label`: short and stable (`A1`, `lemma-even`, …); later `deps` reference claims by label
  - `deps`: every already-committed claim this conclusion directly depends on; to reference
    a tool-execution result use `effect:<seq>` (the event sequence number captured by the hooks)
  - To correct a conclusion, resubmit with `revises=<old label>`: the old conclusion and
    everything downstream of it are automatically invalidated
- A claim's statement must be **self-contained and checkable**: write "for all 1≤n≤100,
  6 divides n³−n", not "the property above holds".
- **Give every claim a temporal index.** A claim about the current/final state is checked
  against the live world at verification time — keep it true through the end, or revise it
  when your own later actions change the state it describes. A claim about a **past action
  or past state** ("the test run succeeded", "at discovery, X was unreachable") must say so
  explicitly ("as of effect:<seq>", "before <event>") and cite the tool execution that
  witnessed it in `deps` (`effect:<seq>`): the hooks record every tool call and its output
  in the verifier's ledger, so such claims verify against the record even after the world
  has legitimately moved on. Present-tense claims about past observations are the single
  most common cause of refuted-but-honest claims.
- You do **not** need to prove it yourself — verification is the verifier's job, and when
  it happens is decided by the trigger rule (the first time a claim is depended on).

## Parallel hypotheses (AND-OR search)

- Explore mutually exclusive hypotheses in parallel with subagents; each subagent commits
  its claims with `branch=<hypothesis name or agent id>`.
- Call `gv.branches` at any time to inspect: once a branch yields a `refuted`, its entire
  downstream cone is a dead branch (**hard kill**: stop spending budget on it immediately);
  `verified` serves only as a soft ordering hint.
- When a claim is refuted, treat it as a **learned exclusion**: read the counterexample
  (witness) out, so later hypotheses don't fall into the same pit.
- Identical conclusions re-derived on different branches are deduplicated by content
  addressing: the second derivation costs no verification.

## Wrap-up

- After committing `final`, call `gv.certificate(final_label=...)`.
- The verdict has exactly four honest values: `verified` (every obligation on the slice
  holds), `partial` (some hold, some could not be discharged), `unverified` (unresolved
  residue, with reasons), `refuted` (a counterexample exists on the slice). Relay the
  verdict and certificate_id verbatim to the user/CI; do not paraphrase them into
  anything beyond plain pass/fail wording.
- `unverified` is not a failure — it means "couldn't be verified": you can add supporting
  grounds and resubmit with a revise.
- **Never leave a `refuted` or `unverified` final standing.** Read the obligation reasons
  and the counterexample witness out of the certificate, fix the world (or the wording, if
  the statement no longer matches reality — e.g. it describes a state that your own later
  actions changed), resubmit with `revises=`, and request the certificate again.
