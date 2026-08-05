"""Shared LLM-as-a-Verifier-style judge (arXiv 2607.05391, ported).

Deviations from the paper, honestly stated: Anthropic models expose no
scoring-token logits, so the paper's
expectation over the score distribution is estimated by its other scaling
axis — K repeated single-shot judgments, mean score, granularity G=11
(0..10). The judge reads text only; no tools, no execution — judging text
is the method under test.

Used by each benchmark's select.py for paper-style best-of-N selection over
the plain arm's candidates.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from verifier.llm.agents import AgentBackend, AgentRole   # noqa: E402
from verifier.llm.roles import RoleError                  # noqa: E402


def make_judge(domain_system: str) -> AgentRole:
    """A 0-10 scoring judge; `domain_system` states what it is judging."""
    return AgentRole(
        name="llm_judge",
        system=domain_system + (
            " Reason carefully, then give a calibrated score."),
        template=(
            "Task:\n{task}\n\n"
            "Candidate solution to judge:\n{candidate}\n\n"
            "Criterion: {criterion}\n"
            "Score the candidate on an integer scale from 0 (certainly "
            "fails the criterion) to 10 (certainly satisfies it). "
            "Output JSON: {{\"score\": s}}"
        ),
        schema={"score": int},
        tools=[],           # the method under test reads; it does not execute
        max_turns=1,
    )


def make_backend(judge: AgentRole, cwd: str, model: str) -> AgentBackend:
    os.makedirs(cwd, exist_ok=True)
    return AgentBackend(roles={"llm_judge": judge}, cwd=cwd, model=model)


def score(judge: AgentRole, backend: AgentBackend, task: str,
          candidate: str, criterion: str, k: int = 4):
    """Mean of K independent judgments. Returns (mean, scores, failures)."""
    scores, failures = [], 0
    for _ in range(k):
        try:
            payload = judge.call(backend, {
                "task": task, "candidate": candidate, "criterion": criterion})
            scores.append(max(0, min(10, int(payload["score"]))))
        except RoleError:
            failures += 1
    mean = sum(scores) / len(scores) if scores else None
    return mean, scores, failures
