# The eight axioms, with worked examples

The verifier's seats are told exactly eight things about evidence
(`verifier/llm/prompts/criteria/axioms.py`). Each is epistemic and
domain-free. Everything task-specific enters at run time through the task
statement, the authorities it names, and the standard public semantics of
the task's domain. Three of the axioms have mechanical counterparts in the
kernel; the prompt states the principle, the kernel enforces the type.

This document is for people: it records the concrete failures each axiom
was distilled from, so that the prompts never have to.

| # | Axiom | Mechanism in the kernel |
|---|-------|-------------------------|
| 1 | AUTHORITY — meaning comes from the controlling statement, the authorities it names, and public domain semantics; never from the solver | — (prompt) |
| 2 | INDEPENDENCE — a delivered artifact may be the subject or an input of a check, never its authority | `model.circular_inputs`; kernel and binding game refuse a `holds` whose declared authority is `delivered` |
| 3 | NECESSITY — keep only obligations the task entails; clauses carry a modality (must / should) judged by wording and consequence, never topic | `Obligation.severity`; `_settle` folds a should-refutation to `partial`, never `refuted` |
| 4 | SUFFICIENCY — obligations at their floors must establish the claim; consistency-only decompositions are insufficient | — (prompt; the all-record abstention in `_settle` is its coarse counterpart) |
| 5 | WITNESS — a refutation is a positive, retained, in-domain observation; absence refutes only a presence requirement | `model.witness_admissible`; `DischargeResult.witness_kind`; kernel converts an inadmissible refutation to `failed` |
| 6 | TIME — decide each proposition at its own time index and basis; record never upgrades to world | `required_basis` floor gates (pre-existing) |
| 7 | REFERENT — an obligation identifies its subject when read alone | — (prompt) |
| 8 | STRENGTH — decisive / bounded / analytic, honestly declared; only decisive refutes; unobtainable is a boundary state | strength ladder (pre-existing) |

Selection follows the same spirit: inside a verdict band only the count
of contrary signals (violated should-clauses) ranks candidates; equal keys
are an explicit tie (`selection.argmax_set`), never a coin flip on how many
incidental checks happened to be runnable.

## Worked examples (the cases the axioms were distilled from)

**Independence.** A database-recovery task: the candidate repaired a
corrupt store and wrote a summary of what it recovered. The check compared
the summary against the repaired store — the solver's own output as ground
truth — and passed a wrong candidate. A retrieval task: the candidate
ranked documents and wrote the fifth; the check re-read the candidate's own
ranking. Both are the same shape: `delivered` in the `authority` role. The
kernel now refuses that `holds` regardless of which model produced it.

**Necessity / modality.** A media-processing task said "you can only use
these libraries" and the only passing candidate imported one extra
standard module. A distributed-training task's ledger showed the final
command as part of a compound shell line rather than alone. Both were
refuted on clauses the task voiced as guidance or incidental form, not as
acceptance conditions. Under the modality rule both compile as `should`:
the violation is recorded as a contrary signal and lowers the verdict to
`partial`; it cannot refute.

**Witness.** A converted trace had a corrupted paired result, so no tool
output printed the pinned library version. One verifier refuted "the
pinned version was used" on that absence; another's audit seat objected
that a successful recorded import of the installed package establishes the
proposition without a version-printing output, and that the absence "does
not entail refutation". The kernel rejects a declared `absence` witness
unless the obligation permits refutation by absence. That gate depends on
the model correctly identifying the witness and the presence requirement;
it does not establish that the underlying interpretation is correct.

**Time.** An imperative such as "repair the journal" is satisfied by the
repair occurring and its effect holding at that time; the journal may
legitimately be absorbed by a later checkpoint. Demanding the intermediate
object's presence in the final world is a necessity violation, and its
absence there refutes nothing about the repair.

**Strength / unobtainable.** A segmentation task needed a model the
verification sandbox could not run; every candidate fell to `unverified`
and the old tie-breaker (fraction of runnable checks) picked a failing
candidate by a few hundredths. Under the new selection rule these
candidates tie explicitly, which is the honest read-out.

**Authority / referent.** The rules of a public game, a published format,
and standard mathematics are legitimate semantic authority for a task set
in that domain; demanding that the task ship its own specification of
them manufactures a phantom evidence gap. Conversely, "the selected rows"
is not a referent until the selection event and its qualifiers are
inlined, and no later method may pick which selection was meant.

**Strength / proportionality.** A secret-extraction task: the candidate
recovered the key by running the supplied binary and wrote it to the named
file with a shell `echo`, so the file ended in a newline. The oracle chose
a "decisive" method that rebuilt the binary's semantics from scratch (a
proof-producing parser, an instruction emulator, a coverage certificate)
and added a ban on "newline removal" the obligation never stated; the
executor spent 23 minutes building that toolchain, the judger 13 more
auditing it, and the check refuted a correct delivery on the trailing
byte. The task's own reference solution wrote the file the same way and
its grader strips whitespace. Two rules came out of it, both in the
prompts as principles: a method inherits the obligation's comparison level
and adds no strictness of its own, and what a supplied program computes is
decided by running it — reconstruction is the fallback when execution
cannot be made faithful. The compiler reads a deliverable under the
standard conventions of the kind the task names (a text file's line
terminator, a value's canonical spelling are renderings) unless the task
fixes the byte-level form.

## How to keep the prompts general

* `tests/test_prompt_generality.py` bans concrete nouns (tools, formats,
  benchmarks, the subjects of once-observed failures) from every prompt
  surface and bounds the corpus size. A rule that needs a banned word is a
  worked example for this file, not a prompt.
* New behaviour goes into a mechanism (a schema field the kernel gates on)
  before it goes into prose. Prose that already existed for the three
  failures above was present in the old criteria and was violated anyway;
  the type-level gates are what changed the outcome.
* Generality is measured, not asserted: freeze the prompts, run a domain
  the rules were never tuned on, and treat any prompt edit that becomes
  necessary as a domain rule to be removed.
