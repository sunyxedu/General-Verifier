# Verification-driven search protocol

The `gv` tools mounted in this session (`commit`, `verify`, `certificate`, `status`)
connect you to an independent verifier: it checks every claim you commit, and certifies
your final answer. This protocol specifies **mechanism** only — how to submit claims and
read verdicts; it contains no knowledge about your task.

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
- You do **not** need to prove it yourself. Verification is the verifier's job, and it
  happens **at commit**: `gv.commit` returns only once the claim has been checked. Expect
  the call to take a while; that wait is the check running. Any status other than
  `verified` comes back with the obligation that failed and its counterexample (or, if
  the claim rests on something already refuted, the fail-closed reason) — read that
  residue and revise **before** building anything on top of the claim.

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
