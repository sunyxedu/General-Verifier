# AIME 2024 — General Verifier vs LLM-as-a-Verifier

First head-to-head: does an evidence-backed verifier beat a
continuous-score LLM judge ([arXiv 2607.05391](https://arxiv.org/abs/2607.05391))
at telling right answers from wrong ones, on the verifier's home turf
(mechanically checkable math)?

## Data

7 problems from AIME 2024 I ([problems.py](problems.py)), each contributing
two claims — the official answer and a **trap**: the value a plausible
mistake produces. The nastiest is p13's 134, which genuinely satisfies
17² | m⁴+1 (one of the four roots mod 289) but is not the *least* such m —
a verifier that checks divisibility and skips minimality passes a wrong
answer. Every label is re-derived by brute force in
[labels.py](labels.py) before any run; a mismatch aborts the experiment.

## Arms

- **A — General Verifier** ([run_gv.py](run_gv.py)): each claim is a fresh
  state dir, committed as a single `final`; the certificate request
  triggers the full pipeline (obligation compiler → oracle → discharge →
  judger, agent seats, binding game on). Verdict ∈ {verified, partial,
  unverified, refuted}.
- **B — LLM judge, paper-style** ([run_judge.py](run_judge.py)): the same
  claim text, no tools; continuous score via K=8 repeated single-shot
  judgments (0–10), mean. Repeats stand in for the paper's scoring-token
  logit expectation, which Anthropic models do not expose — the paper's
  own limitation section notes this exclusion, and repeated evaluation is
  its second scaling axis for the same quantity.

Both arms run the same model (default `sonnet`): the comparison is
evidence-vs-opinion at matched capability, not model shopping.

## Scoring

| | label=correct | label=wrong |
|---|---|---|
| GV verified | right | **FALSE PASS** (one-vote veto) |
| GV refuted | false alarm | right |
| GV partial/unverified | abstention | abstention |
| judge mean ≥ 5.5 | right | wrong |
| judge mean < 5.5 | wrong | right |

Judge accuracy is also reported across thresholds so the 5.5 cut is not
doing hidden work. Cost (wall time per claim) is reported for both arms.

## Run

```bash
uv run python experiments/aime2024/labels.py     # must print ALL LABELS CONFIRMED
uv run python experiments/aime2024/run_judge.py  # fast arm (~minutes)
uv run python experiments/aime2024/run_gv.py     # slow arm (real instruments)
uv run python experiments/aime2024/summarize.py
```

Both runners are resumable (recorded claims are skipped). Results land in
`results/` (untracked); the shared instrument toolchain persists at
`~/.cache/gv-toolchain`.
