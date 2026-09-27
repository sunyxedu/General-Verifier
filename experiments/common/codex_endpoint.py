"""OpenAI-compatible endpoint backed by the local Codex CLI.

Exists to run third-party baselines (LLM-as-a-Verifier) on the user's
ChatGPT subscription models — the codex CLI authenticates with the
subscription, so there is no API key and no direct HTTP endpoint to point
a baseline at. This server is that endpoint: it speaks the slice of
/v1/chat/completions those baselines use and answers each request by
running one `codex exec` in a scratch directory.

KNOWN LIMITATION — no logprobs. The subscription path returns text only,
so `logprobs`/`top_logprobs` in a request are accepted and ignored, and
the response carries no logprobs field. LLM-as-a-Verifier detects this
and falls back to parsing the score letter out of the text, which is its
documented degraded mode (coarse 20-level scores instead of the
fine-grained expectation over the token distribution). Any comparison
run through here must say so.

Usage:
  python experiments/common/codex_endpoint.py --port 8900 --model gpt-5.6-luna
  python experiments/common/codex_endpoint.py --port 8900 --model gpt-5.6-terra \
      --service-tier fast
  OPENAI_BASE_URL=http://127.0.0.1:8900/v1 OPENAI_API_KEY=unused <baseline>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from verifier.llm.codex.service_tier import (                    # noqa: E402
    DEFAULT_SERVICE_TIER,
    SERVICE_TIERS,
    normalize_service_tier,
    service_tier_cli_args,
)
from verifier.llm.endpoints.presets import default_model  # noqa: E402

DEFAULT_MODEL = default_model("codex")
DEFAULT_EFFORT = "low"          # judging is short-form; keep latency sane


def parse_exec_events(output):
    """Return ``(assistant_text, usage, thread_id)`` from ``codex --json``.

    Usage is accepted only when Codex reports every counter needed by the
    verifier ledger.  Callers can then fail closed instead of silently
    recording estimated or partial token counts.
    """
    assistant_text = ""
    usage = None
    thread_id = None
    required_usage = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    # Native Codex JSONL uses LF records. NEL/LS/PS may occur literally in a
    # JSON string and are not record separators.
    for line in (output or "").split("\n"):
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id") or thread_id
        elif event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    assistant_text = text.strip()
        elif event.get("type") == "turn.completed":
            candidate = event.get("usage")
            if not isinstance(candidate, dict):
                continue
            if all(
                isinstance(candidate.get(name), int)
                and not isinstance(candidate.get(name), bool)
                and candidate[name] >= 0
                for name in required_usage
            ):
                usage = {name: candidate[name] for name in required_usage}
    return assistant_text, usage, thread_id


class CodexRunner:
    def __init__(self, model, effort, timeout, concurrency,
                 service_tier=DEFAULT_SERVICE_TIER):
        self.model = model
        self.effort = effort
        self.service_tier = normalize_service_tier(service_tier)
        self.timeout = timeout
        self.slots = threading.Semaphore(concurrency)
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    def complete_with_usage(self, prompt: str):
        """One Codex exec returning text and its provider-reported usage."""
        with self.slots:
            work = tempfile.mkdtemp(prefix="codex-endpoint-")
            out = os.path.join(work, "last.txt")
            cmd = ["codex", "exec", "--model", self.model,
                   "--skip-git-repo-check", "--sandbox", "read-only",
                   "-c", f"model_reasoning_effort={self.effort}"]
            cmd.extend(service_tier_cli_args(self.service_tier))
            cmd.extend(["--json", "-o", out, "--", prompt])
            try:
                with open(os.devnull) as devnull:
                    # stdin MUST be closed: codex otherwise waits on it and
                    # the request hangs forever
                    proc = subprocess.run(cmd, cwd=work, stdin=devnull,
                                          capture_output=True, text=True,
                                          timeout=self.timeout)
                event_text, usage, thread_id = parse_exec_events(proc.stdout)
                text = ""
                if os.path.exists(out):
                    with open(out, encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                if not text:
                    text = event_text
                with self._lock:
                    self.calls += 1
                    if not text:
                        self.failures += 1
                return text, usage, thread_id
            except subprocess.TimeoutExpired:
                with self._lock:
                    self.calls += 1
                    self.failures += 1
                return "", None, None
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def complete(self, prompt: str) -> str:
        """One Codex exec; returns the final assistant message."""
        text, _usage, _thread_id = self.complete_with_usage(prompt)
        return text


def flatten(messages) -> str:
    """Chat messages -> one prompt. Image parts are dropped (the codex CLI
    takes text only here); text parts are concatenated in order."""
    parts = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content
                                if isinstance(c, dict) and c.get("type") in
                                ("text", "input_text"))
        if not content:
            continue
        role = m.get("role", "user")
        parts.append(content if role == "user" else f"[{role}]\n{content}")
    return "\n\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    runner: CodexRunner = None       # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # quiet: one line per call below
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            # first entry wins: baselines that auto-resolve the model name
            # off /v1/models must land on the model we actually serve
            self._send(200, {"object": "list", "data": [
                {"id": self.runner.model, "object": "model",
                 "owned_by": "codex-cli"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as err:
            self._send(400, {"error": {"message": f"bad json: {err}"}})
            return
        prompt = flatten(body.get("messages"))
        started = time.time()
        text, usage, _thread_id = self.runner.complete_with_usage(prompt)
        seconds = time.time() - started
        print(f"[codex-endpoint] {len(prompt)}B prompt -> {len(text)}B "
              f"in {seconds:.0f}s"
              + ("  (EMPTY)" if not text else ""), flush=True)
        if not text:
            self._send(502, {"error": {
                "message": "codex exec produced no output"}})
            return
        if usage is None:
            self._send(502, {"error": {
                "message": "codex exec did not report token usage"}})
            return
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(started),
            "model": self.runner.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                # no logprobs on the subscription path (see module docstring)
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "prompt_tokens_details": {
                    "cached_tokens": usage["cached_input_tokens"],
                },
                "completion_tokens": usage["output_tokens"],
                "completion_tokens_details": {
                    "reasoning_tokens": usage["reasoning_output_tokens"],
                },
                "total_tokens": (
                    usage["input_tokens"] + usage["output_tokens"]
                ),
            },
        })


def serve(port, model, effort, timeout, concurrency,
          service_tier=DEFAULT_SERVICE_TIER):
    Handler.runner = CodexRunner(
        model, effort, timeout, concurrency, service_tier=service_tier)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"[codex-endpoint] http://127.0.0.1:{port}/v1 model={model} "
          f"effort={effort} service_tier={Handler.runner.service_tier} "
          f"concurrency={concurrency} (no logprobs)",
          flush=True)
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument(
        "--service-tier", choices=SERVICE_TIERS,
        default=DEFAULT_SERVICE_TIER,
        help="Codex service tier (default: standard)",
    )
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-call seconds")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="max concurrent codex processes")
    args = ap.parse_args()
    server = serve(args.port, args.model, args.effort, args.timeout,
                   args.concurrency, args.service_tier)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[codex-endpoint] {Handler.runner.calls} calls, "
              f"{Handler.runner.failures} failures", flush=True)


if __name__ == "__main__":
    main()
