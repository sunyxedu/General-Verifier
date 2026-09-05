"""TurboAgent — the official LLM-as-a-Verifier Claude Code plugin.

Runs the authors' proxy (github.com/llm-as-a-verifier/TurboAgent) as
shipped, from a vendored clone: Claude Code talks to it via
ANTHROPIC_BASE_URL, and on every model call the proxy fans out
`num_candidates` parallel generations and picks the best with the
fine-grained logprob verifier + Probabilistic Pivot Tournament.

The turbo arm is the plugin used exactly as documented:
    turbo-agent            # with a turbo-agent.yaml in cwd
    ANTHROPIC_BASE_URL=http://localhost:8888 claude

Keys: DEEPSEEK_API_KEY for the backend generations (defaulted from
verifier.llm.provider), plus VERTEX_API_KEY — the verifier scores with
Gemini logprobs (the backend models expose no logprobs, the paper's own
App. A).
"""
import os
import subprocess
import sys
import time
import urllib.request

from verifier.llm import provider   # DeepSeek wiring (single source of truth)

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "vendor", "TurboAgent")
GIT_URL = "https://github.com/llm-as-a-verifier/TurboAgent.git"

# claude CLI aliases -> the configured provider's litellm model id
MODEL_IDS = {
    "sonnet": provider.LITELLM_MODEL,
    "opus": provider.LITELLM_MODEL,
    "haiku": provider.LITELLM_MODEL,
}

CONFIG_TEMPLATE = """\
log_dir: default

backend:
  models:
    - name: {backend_model}
      api_key: ${key_env}
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
            backend_model=backend, key_env=provider.KEY_ENV,
            num_candidates=num_candidates,
            verifier_model=verifier_model, n_verifications=n_verifications))


class Proxy:
    """`with Proxy(config_dir, port) as base_url:` — start/stop turbo-agent."""

    def __init__(self, config_dir, port=8899):
        self.config_dir = config_dir
        self.port = port
        self.proc = None

    def __enter__(self):
        provider.validate_configuration()
        ensure_vendor()
        # the proxy's litellm backend reads the provider key from its env
        os.environ.setdefault(provider.KEY_ENV, provider.API_KEY)
        log = open(os.path.join(self.config_dir, "turbo-agent.log"), "a")
        self.proc = subprocess.Popen(
            ["uv", "tool", "run", "--from", VENDOR, "turbo-agent",
             "-p", str(self.port)],
            cwd=self.config_dir, stdout=log, stderr=log)
        # no startup deadline: first launch may build the vendored package,
        # which can take arbitrarily long. We only fail if the process dies.
        url = f"http://localhost:{self.port}/v1/models"
        while True:
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

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False


def check_keys():
    provider.validate_configuration()
    os.environ.setdefault(provider.KEY_ENV, provider.API_KEY)
    vertex = provider.cfg("VERTEX_API_KEY")
    if vertex:
        os.environ.setdefault("VERTEX_API_KEY", vertex)
    if not os.environ.get("VERTEX_API_KEY"):
        sys.exit("turbo arm needs VERTEX_API_KEY exported or in .env (the "
                 "logprob verifier scores with Gemini; generations use the "
                 f"configured provider key ${provider.KEY_ENV})")
