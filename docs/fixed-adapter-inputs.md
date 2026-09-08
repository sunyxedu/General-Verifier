# Verifier-owned fixed subject inputs

Experimental per-MethodIR opt-in, not an automatic rewrite of existing methods.
No paid speedup or no-loss result is established for this change.

The previous semantic matrix repeatedly fell back because Oracle wrote a
physical ledger filename where an instrument required the logical subject
`ledger:events`. Requiring a model to repeat a known instrument constant adds
a failure opportunity without adding semantic judgment. The implementation
follows the narrow known-argument guidance in
[OpenAI function-calling documentation](https://developers.openai.com/api/docs/guides/function-calling#best-practices-for-defining-functions).
It does not adopt the separate suggestion to combine sequential functions:
the project's Compiler, Critic, Oracle, executor and Judger remain separate.

## Contract

An adapter may register immutable `FixedAdapterInput` descriptors. A descriptor
defines only a **subject** slot: either a literal reference or a reference
prefix followed by an exact positive-integer argument. There are no task-name
rules, path aliases, command parsing, source reads, or inferred authority.
The registry captures descriptors so its advertised identity and binding use
the same values, including after an extension replaces a class attribute.

| Instrument | Fixed subject | Still chosen by Oracle |
| --- | --- | --- |
| `ledger.tool-call-string` | `ledger:events` / ledger | pointer, predicate, value, scope and watermarks |
| `ledger.event` | `event:` + exact integer `event_seq` / ledger | event number and expected type/payload |

Oracle opts in with `adapter_input_binding: "fixed-v1"` and leaves these slots
out of `adapter_inputs`, using `[]` if there are no other inputs. All original
four adapter fields must still be present. Without the new field, the original
strict explicit-input behavior is unchanged. Availability in the catalog is
not a guarantee that the model will choose this mode or stop making errors.

The existing Oracle-return boundary materializes a full MethodIR before an
attempt is constructed. Execution, ordinary fallback/redo, independent Judger,
evidence and final binding all use that same expanded method. Direct execution
of an unmaterialized fixed-mode method fails closed. The evidence and execution
events retain the binding mode. New fixed-mode methods cannot use accepted-method
reuse; fresh Oracle and Judger are still required.

Conflicting refs/provenances/roles, additional subjects, duplicate slots,
unsupported versions, malformed/partial JSON, and excessive input counts are
not corrected. The original executor fallback remains. Semantic arguments
(including invalid predicates) are never stripped or changed. An unavailable
event stays unavailable, and absence authorization is unchanged. An expanded
input list is a declaration, not proof that a predicate entails an obligation.

## Tests and remaining gate

Tests cover constant and integer-derived subjects, unrelated extension adapter
names, immutable descriptor capture, concurrent non-aliasing, legacy errors,
direct-execution guards, input/evidence/method identity, online/offline positive
and contradictory records, absence authorization, Oracle repair, Judger rejection
and execution redo. They use local stubs, not paid model judgments.

The earlier paid `any_identity/opaque` failure also involved a separate Judger
coverage interpretation. Fixed inputs do not fix that semantic issue or establish
10× end-to-end performance. Subject snapshots and online source-epoch invalidation
are not implemented here; no new time boundary is silently imposed.
