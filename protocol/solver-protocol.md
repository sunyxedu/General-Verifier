# Verification-driven search protocol

The `gv` tools mounted in this session (`commit`, `verify`, `certificate`, `status`)
connect you to an independent verifier: it checks the claims you commit and certifies
your final answer. This protocol specifies **mechanism** only — how to submit claims and
read verdicts; it contains no knowledge about your task.

## What deserves to enter the claim graph

Every `commit` starts an independent check, so the graph should hold **load-bearing
conclusions, not a diary**. Decide per claim, from the task itself:

- **Commit** a conclusion when later work depends on it AND its being wrong would be
  expensive to discover late — a diagnosis of the root cause, a reading of what the task
  or an authority requires, a design assumption about the environment, an intermediate
  result the deliverable is built from. These are worth an independent check *before*
  you build on them.
- **Do not commit** what the record already shows. The hooks retain every tool call and
  its output; a command's result needs no claim of its own — cite it as `effect:<seq>`
  from the claim that relies on it. Do not commit narration ("I will now …"), restated
  task text, or trivially re-observable facts.
- **One conclusion per claim, without the evidence.** The counts, listings, and outputs
  that support a conclusion are already on the record; cite the tool executions that
  produced them as `effect:<seq>` deps instead of copying them into the statement. A
  bundled claim ("A, because B (12345 lines), and therefore C") compiles into many
  checks and waits for the slowest of them; three short claims each verify quickly, and a
  refutation then points at the one that is wrong.
- Most tasks need **zero to four** intermediate claims plus one `final`. Fewer, sharper
  claims mean less waiting and a clearer certificate; the final's verdict does not depend
  on how many intermediate claims you made.

## Submission discipline

- `gv.commit` fields:
  - `kind`: `assertion` (intermediate conclusion) | `final` (final answer)
  - `label`: short and stable (`A1`, `lemma-even`, …); later `deps` reference claims by label
  - `deps`: every already-committed claim this conclusion directly depends on; to reference
    a tool-execution result use `effect:<seq>` (the event sequence number captured by the hooks).
    Every `gv` response lists the tool results recorded since your previous `gv` call under
    `recent_effects`, each with its `effect:<seq>` id and a glimpse of its input and output:
    cite ids from there. The ledger's numbering is not the line numbering of any file you
    can read.
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
  has legitimately moved on. Index by the record, never by the calendar: a date or clock
  time in a claim is one more term to establish, and the record rarely settles it. Present-tense claims about past observations are the single
  most common cause of refuted-but-honest claims.
- You do **not** need to prove it yourself. Verification is the verifier's job. An
  `assertion`'s `gv.commit` returns as soon as the claim is recorded (`checking: true`);
  its check runs in the background while you keep working. `gv.status(label)` shows the
  verdict, and when a background check refutes a claim, the refutation — the failing
  obligation and its counterexample — arrives in the `news` field of your next `gv` call.
  Read `news` whenever it appears, and rely on a claim still being checked only as far as
  you would on any unchecked conclusion. A `final`'s commit waits for its check, and
  `gv.certificate` waits for every check in its slice. A final's check can take tens of
  minutes; while it runs, `gv.status` marks it `checking`. Do not commit another version
  of the final meanwhile unless you have changed the deliverable: every version starts its
  own full check, and the one already running is not sped up by it.

## Reading a commit result

- `verified`: independently established. Build on it.
- `confident: true` (status stays `unverified`): the verifier's screener found the claim
  settled by the retained record — a plain restatement of something your tool outputs
  already show — and **admitted it without spending a check**. Build on it as you would
  on a verified claim; the certificate lists it as `confident`, not `verified`. If you
  want the full independent check anyway, call `gv.verify(label)`.
- Any other status — on a final's commit, in `gv.status`, or in `news` — comes with the
  obligation that failed and its counterexample (or,
  if the claim rests on something already refuted, the fail-closed reason), plus
  `still_holding`: the conditions of the same contract that were established. A fix for
  the failing condition must keep every one of those true — trading one requirement for
  another is the most common way a repair makes the delivery worse. `gv.verify` on an
  already-checked claim repeats the same check on the same statement; it is not a second
  roll of the dice — change the world or the wording and `revises=` instead.
- A refuted **intermediate** claim means the statement as written is false. Each failing
  obligation comes back with `required_by_task`, the verifier's reading of the task
  statement: `required` — the deliverable must satisfy it, change the world;
  `beyond_task` — your statement promised more than the task demands, narrow or drop the
  claim and leave the deliverable alone; `unclear` — read the task yourself. The task
  statement is the only contract the deliverable must satisfy; your own claims are not a
  second one.
- The verifier checks world conditions against the world **as it is now**. After you
  change files or state, earlier world-based verdicts are stale and are re-checked when
  next relied on; commit the `final` only after your last change to the deliverable.

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
  grounds and resubmit with a revise. A certificate does not rerun a check on the world it
  already ran on (the claim is marked `not_rechecked`): change what the residue says is
  missing, or, after a change outside the files (a service started, a package installed),
  call `gv.verify`.
- **Never leave a `refuted` or `unverified` final standing.** Read the obligation reasons
  and the counterexample witness out of the certificate, fix the world (or the wording, if
  the statement no longer matches reality — e.g. it describes a state that your own later
  actions changed), resubmit with `revises=`, and request the certificate again.
