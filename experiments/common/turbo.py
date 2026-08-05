"""TurboAgent — the official LLM-as-a-Verifier Claude Code plugin.

Runs the authors' proxy (github.com/llm-as-a-verifier/TurboAgent) as
shipped, from a vendored clone: Claude Code talks to it via
ANTHROPIC_BASE_URL, and on every model call the proxy fans out
`num_candidates` parallel generations and picks the best with the
fine-grained logprob verifier + Probabilistic Pivot Tournament.

The turbo arm is the plugin used exactly as documented:
    turbo-agent            # with a turbo-agent.yaml in cwd
    ANTHROPIC_BASE_URL=http://localhost:8888 claude

Keys: ANTHROPIC_API_KEY for the backend generations, plus VERTEX_API_KEY —
the verifier scores with Gemini logprobs (Anthropic models expose no
logprobs, the paper's own App. A).
"""
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "vendor", "TurboAgent")
GIT_URL = "https://github.com/llm-as-a-verifier/TurboAgent.git"

# claude CLI aliases -> concrete API model ids for the litellm backend
MODEL_IDS = {
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "haiku": "claude-haiku-4-5-20251001",
}

CONFIG_TEMPLATE = """\
log_dir: default

backend:
  models:
    - name: anthropic/{backend_model}
      api_key: $ANTHROPIC_API_KEY
      num_candidates: {num_candidates}
      temperature: 1

verifier:
  model:
    name: gemini/{verifier_model}
    api_key: $VERTEX_API_KEY
    provider: vertex_ai

  majority_voting: true

  method:
    name: pivot_tournament
    pivots: 2
    n_verifications: {n_verifications}
    seed: 0
    note: >-
      There is no reference solution available. Judge each trajectory purely on
      how plausibly it solved the task correctly.
    criteria:
      - name: Task Success
        description: >-
          How likely the agent correctly and completely solved the task. The
          strongest signal is the agent verifying its solution against the
          task's specific requirements. Trajectory length, number of steps, and
          apparent confidence do not predict correctness.
"""


def ensure_vendor():
    if not os.path.isdir(os.path.join(VENDOR, "turbo_agent")):
        os.makedirs(os.path.dirname(VENDOR), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", GIT_URL, VENDOR],
                       check=True)


def write_config(config_dir, solver_model="sonnet", num_candidates=3,
                 verifier_model="gemini-2.5-flash", n_verifications=1):
    os.makedirs(config_dir, exist_ok=True)
    backend = MODEL_IDS.get(solver_model, solver_model)
    with open(os.path.join(config_dir, "turbo-agent.yaml"), "w") as f:
        f.write(CONFIG_TEMPLATE.format(
            backend_model=backend, num_candidates=num_candidates,
            verifier_model=verifier_model, n_verifications=n_verifications))


class Proxy:
    """`with Proxy(config_dir, port) as base_url:` — start/stop turbo-agent."""

    def __init__(self, config_dir, port=8899):
        self.config_dir = config_dir
        self.port = port
        self.proc = None

    def __enter__(self):
        ensure_vendor()
        log = open(os.path.join(self.config_dir, "turbo-agent.log"), "a")
        self.proc = subprocess.Popen(
            ["uv", "tool", "run", "--from", VENDOR, "turbo-agent",
             "-p", str(self.port)],
            cwd=self.config_dir, stdout=log, stderr=log)
        deadline = time.time() + 180
        url = f"http://localhost:{self.port}/v1/models"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"turbo-agent exited (see {self.config_dir}/"
                    "turbo-agent.log)")
            try:
                urllib.request.urlopen(url, timeout=2)
                print(f"[turbo] proxy up on :{self.port}", flush=True)
                return f"http://localhost:{self.port}"
            except OSError:
                time.sleep(1)
        raise RuntimeError("turbo-agent did not come up in 180s")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False


def check_keys():
    missing = [k for k in ("ANTHROPIC_API_KEY", "VERTEX_API_KEY")
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"turbo arm needs {' and '.join(missing)} exported "
                 "(generations + logprob verifier)")
