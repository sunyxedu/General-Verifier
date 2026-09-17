# Verification-driven search protocol

The `gv` tools mounted in this session (`commit`, `verify`, `certificate`, `status`)
connect you to an independent verifier: it checks the claims you commit and certifies
your final answer. This protocol specifies **mechanism** only — how to submit claims and
read verdicts; it contains no knowledge about your task.

## What deserves to enter the claim graph

Every `commit` blocks until the verifier has dealt with the claim, so the graph should
hold **load-bearing conclusions, not a diary**. Decide per claim, from the task itself:

- **Commit** a conclusion when later work depends on it AND its being wrong would be
  expensive to discover late — a diagnosis of the root cause, a reading of what the task
  or an authority requires, a design assumption about the environment, an intermediate
  result the deliverable is built from. These are worth an independent check *before*
  you build on them.
- **Do not commit** what the record already shows. The hooks retain every tool call and
  its output; a command's result needs no claim of its own — cite it as `effect:<seq>`
  from the claim that relies on it. Do not commit narration ("I will now …"), restated
  task text, or trivially re-observable facts.
- Most tasks need **zero to four** intermediate claims plus one `final`. Fewer, sharper
  claims mean less waiting and a clearer certificate; the final's verdict does not depend
  on how many intermediate claims you made.

## Submission discipline

- `gv.commit` fields:
  - `kind`: `assertion` (intermediate conclusion) | `final` (final answer)
  - `label`: short and stable (`A1`, `lemma-even`, …); later `deps` reference claims by label
  - `deps`: every already-committed claim this conclusion directly depends on; to reference
    a tool-execution result use `effect:<seq>` (the event sequence number captured by the hooks)
  - To correct a conclusion, resubmit with `revises=<old label>`: the old conclusion and
    its checks are invalidated, along with consumers of shared or inherited evidence.
    The statement may stay the same after you repair the underlying result: an explicit
    revision creates a new claim version and runs a fresh check. Repeating a commit
    without `revises` reuses the existing version.
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
  happens **at commit**: `gv.commit` returns only once the claim has been dealt with.
  Expect the call to take a while; that wait is the check running.

## Reading a commit result

- `verified`: independently established. Build on it.
- `confident: true` (status stays `unverified`): the verifier's screener found the claim
  settled by the retained record — a plain restatement of something your tool outputs
  already show — and **admitted it without spending a check**. Build on it as you would
  on a verified claim; the certificate lists it as `confident`, not `verified`. If you
  want the full independent check anyway, call `gv.verify(label)`.
- Any other status comes back with the obligation that failed and its counterexample (or,
  if the claim rests on something already refuted, the fail-closed reason). Read that
  residue and revise **before** building anything on top of the claim.

## Wrap-up

- After committing `final`, call `gv.certificate(final_label=...)`.
- The verdict comes from the final claim's independently checked task contract; the final
  is never screened and never admitted on confidence. Earlier claims remain in the
  certificate for audit; their status does not decide the final verdict. The final admits
  no assumed premises. If no `task.md` is supplied, its own wording is the available
  contract.
- There are five verdicts: `verified` (all final obligations hold at decisive strength,
  including at least one world-based check), `supported` (all hold, but some evidence
  is bounded or analytic), `partial` (checks are unavailable or advisory conditions
  were violated), `unverified` (unresolved checks or insufficient evidence), and
  `refuted` (an admissible decisive counterexample to a mandatory final obligation).
  Relay the verdict and certificate_id verbatim to the user/CI. Include unresolved
  conditions and, for `supported`, the disclosed bridging assumptions.
- `unverified` is not a failure — it means "couldn't be verified": you can add supporting
  grounds and resubmit with a revise.
- **Never leave a `refuted` or `unverified` final standing.** Read the obligation reasons
  and the counterexample witness out of the certificate, fix the world (or the wording, if
  the statement no longer matches reality — e.g. it describes a state that your own later
  actions changed), resubmit with `revises=`, and request the certificate again.
