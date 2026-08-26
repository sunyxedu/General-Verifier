"""Referent identity and binder closure: what object a proposition is about must be fixed before anything may decide it — enumerated antecedents, event/object binders, and the ban on methods choosing referents retroactively."""

ENUMERATED_MULTI_REFERENT_RULE = (
    "ENUMERATED MULTI-REFERENT RULE: more than one antecedent chain may form "
    "one referent only when the controlling phrase itself explicitly enumerates "
    "or cardinally quantifies the antecedent objects (for example `both A and "
    "B`, `these two named files`, or `all three runs`) and the dependency "
    "topology uniquely and exhaustively identifies exactly those members. "
    "Ordinary plural or collection nouns such as `the selected rows`, `the "
    "files`, `the tokens`, `the results`, or `the dataset` still require one "
    "uniquely identified collection-producing antecedent; grammatical plurality "
    "and the mere existence of several compatible parents never authorize "
    "merging their collections, components, events, or chains. Resolving an "
    "enumerated set fixes member identity only; it does not establish union, "
    "overlap handling, deduplication, task-scope coverage, or additivity. Those "
    "remain semantic conditions controlled by the claim and authority.\n"
)

TRANSFORMATION_EVENT_BINDING = (
    "TRANSFORMATION EVENT BINDING: for a task whose program or action mutates "
    "an input, preserve the task's own quantifier and preconditions; closing a "
    "binder must not turn one invocation or a conditional requirement into an "
    "universal claim. Every behavioral obligation must be self-contained around "
    "one event and object: name an invocation E, the input object or path P "
    "supplied to E, P immediately before E, and the same P immediately after E. Bind "
    "each output, removal, preservation, formatting, safety, or equivalence "
    "predicate to P after that same E, and each comparison to P before that "
    "same E. Repeat the complete binder in every independently checked sibling "
    "obligation; a sibling may not supply a missing producer, invocation, "
    "input, output, time, or same-object relation.\n"
)

REFERENT_METHOD_BOUNDARY = (
    "REFERENT IDENTITY MUST PRECEDE METHOD SELECTION: the obligation itself "
    "must identify its subject when read in isolation. Task context may "
    "interpret an explicit task term or named-authority reference, but a "
    "candidate method, execution artifact, or retained effect may not "
    "retroactively choose an omitted source, configuration, split, filter, "
    "population, time, or event among multiple compatible objects. A generic "
    "relational phrase such as `rows selected by the solver` is not a unique "
    "binder merely because one convenient selection appears in the method. "
    + ENUMERATED_MULTI_REFERENT_RULE
    + "If the proposition's target remains ambiguous, enumerate "
    "no method or "
    "return sufficient=no; evidence may decide a predicate of an identified "
    "target, never supply the target identity that the obligation omitted.\n"
)
