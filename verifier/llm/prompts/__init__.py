"""
Prompt text for the verifier's agent roles.

Every role has two parts: SYSTEM (the role's identity and rules, appended
to the SDK's claude_code preset) and TEMPLATE (the per-call prompt,
rendered with str.format over the role's variables).

Templates stay generic. The only cross-seat rules are the eight axioms in
criteria/axioms.py: epistemic, domain-free, and free of case patches.
Domain content arrives at runtime — from the task statement, the
authorities it names, the standard public semantics of its domain, and
the environment the seat can probe — never from text in the prompts.
"""

from .criteria import (
    AXIOM_AUTHORITY,
    AXIOM_INDEPENDENCE,
    AXIOM_NECESSITY,
    AXIOM_SUFFICIENCY,
    AXIOM_WITNESS,
    AXIOM_TIME,
    AXIOM_REFERENT,
    AXIOM_STRENGTH,
    AXIOMS,
)
from .obligations import (
    OBLIGATIONS_SYSTEM,
    COMPACT_COMPLETENESS_COMPILER_ADDENDUM,
    OBLIGATIONS_HARNESS_ADDENDUM,
    CONTRACT_LOCUS_NONFINAL,
    CONTRACT_LOCUS_FINAL,
    OBLIGATIONS_TEMPLATE,
)
from .critic import (
    CRITIC_SYSTEM,
    COMPACT_COMPLETENESS_CRITIC_ADDENDUM,
    CRITIC_TEMPLATE,
    CRITIC_FEEDBACK_TEMPLATE,
    APPEND_DELTA_FEEDBACK_TEMPLATE,
    APPEND_DELTA_OUTPUT_CONTRACT,
    APPEND_DELTA_GUARD_FEEDBACK,
    APPEND_REPAIR_GUARD_FEEDBACK,
)
from .oracles import (
    ORACLES_SYSTEM,
    ORACLES_TEMPLATE,
)
from .discharge import (
    DISCHARGE_SYSTEM,
    DISCHARGE_TEMPLATE,
)
from .judger import (
    JUDGER_SYSTEM,
    JUDGER_TEMPLATE,
    COUNTERSIGN_REFUTATION,
)
from .sufficiency import (
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_TEMPLATE,
)
from .extract import (
    EXTRACT_SYSTEM,
    EXTRACT_TEMPLATE,
)
from .screener import (
    SCREENER_SYSTEM,
    SCREENER_TEMPLATE,
)
from .script_judger import (
    SCRIPT_JUDGER_SYSTEM,
    SCRIPT_JUDGER_TEMPLATE,
)
