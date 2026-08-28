"""
Prompt text for the verifier's agent roles.

Every role has two parts: SYSTEM (the role's identity and rules, appended to the SDK's claude_code preset) 
and TEMPLATE (the per-call prompt, rendered with str.format over the role's variables).

Temapltes stay generic. 
Domain content arrives at runtime: from the environment the agent can probe, from the model's latent knowledge, 
and from retrieved memory, but never from text written in the prompts.
"""

from .criteria import (
    PROOF_CARRYING_EFFECTS,
    SIGNED_WORLD_EVIDENCE_REUSE,
    OFFLINE_FROZEN_EXECUTION,
    TASK_COMPLETE_OCCURRENCE_COMPOSITION,
    SIBLING_PARTITION_COMPLETENESS,
    ENUMERATED_MULTI_REFERENT_RULE,
    REFERENT_METHOD_BOUNDARY,
    SUPPORTING_PROVENANCE_BOUNDARY,
    SOURCE_TERM_ROLE_BINDING,
    TASK_INSTANCE_CONTRACT,
    EDGE_NECESSITY_GATE,
    NODE_NEGATION_GATE,
    QUANTIFIER_ROLE_BINDING,
    DIRECTIONAL_EXTREMAL_SUFFICIENCY,
    PARTIAL_OBSERVATION_COMPLETIONS,
    DERIVED_STATE_PROVENANCE,
    PERCEPTUAL_SEMANTIC_ANCHOR,
    TRANSFORMATION_EVENT_BINDING,
)
from .obligations import (
    OBLIGATIONS_SYSTEM,
    OBLIGATIONS_HARNESS_ADDENDUM,
    CONTRACT_LOCUS_NONFINAL,
    CONTRACT_LOCUS_FINAL,
    OBLIGATIONS_TEMPLATE,
)
from .critic import (
    CRITIC_SYSTEM,
    CRITIC_TEMPLATE,
    CRITIC_FEEDBACK_TEMPLATE,
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
)
from .extract import (
    EXTRACT_SYSTEM,
    EXTRACT_TEMPLATE,
)
