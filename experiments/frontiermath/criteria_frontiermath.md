# FrontierMath — Verifier Criteria

## Ground Truth Note

Do NOT trust the agent's self-assessment, confidence, or narration. Judge
only what the mathematics and the observed computation output support. Each
task has exactly one correct final answer (an integer or an exact closed
form); plausible-looking but wrong answers are common.

## Criteria

### Final Answer Correctness {#correctness}

Work the mathematics of the task yourself as far as you can and judge
whether the FINAL ANSWER stated at the end of the trajectory is exactly
correct. Score HIGH only when the stated answer equals the true answer of
the problem; score LOW when it is wrong or missing, no matter how plausible
the derivation looks. Ignore style, length, and confidence — only exact
correctness of the final answer matters.

### Derivation Soundness {#derivation}

Check the key steps of the trajectory's mathematical argument: are the
lemmas actually true, are case analyses exhaustive, do the computations the
agent ran (and their printed outputs) support the claims made from them?
Score HIGH when every load-bearing step is correct or mechanically
confirmed by executed code output; score LOW when a load-bearing step is
wrong, unjustified, or contradicted by the agent's own program output.
Ignore harmless detours and dead ends that the agent abandoned.

### Computational Verification {#verification}

Look at the code the agent actually ran and what it printed, not what the
agent claimed about it. Reward trajectories that verified the final answer
mechanically (e.g. brute force on small cases consistent with the general
argument, an independent second computation, or a direct check of the
defining property). Penalize trajectories whose final answer was never
checked by any executed computation, or whose printed outputs disagree with
the stated answer.
