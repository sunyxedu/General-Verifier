"""Bootstrap for the OFFICIAL LLM-as-a-Verifier implementation.

The judge arms run the authors' own package (arXiv 2607.05391) exactly as
shipped — fine-grained logprob reward + Probabilistic Pivot Tournament via
`llm_verifier.select(...)` — from a vendored clone (gitignored). Nothing of
the pipeline is reimplemented here.

Backend requirement (theirs): a logprob-capable endpoint —
  VERTEX_API_KEY=...                        -> gemini-2.5-flash (default)
  OPENAI_BASE_URL=... OPENAI_API_KEY=...    -> any OpenAI-compatible server
                                               that returns logprobs (vLLM…)
Anthropic models cannot serve this arm (no logprobs; the paper's App. A).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "vendor", "llm-as-a-verifier")
GIT_URL = "https://github.com/llm-as-a-verifier/llm-as-a-verifier.git"


def load():
    """Clone the official repo if missing, import and return `llm_verifier`."""
    if not os.path.isdir(os.path.join(VENDOR, "llm_verifier")):
        os.makedirs(os.path.dirname(VENDOR), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", GIT_URL, VENDOR],
                       check=True)
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import llm_verifier
    return llm_verifier
