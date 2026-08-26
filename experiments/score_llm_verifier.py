"""Post-hoc LLM-as-a-Verifier scoring for saved Terminal-Bench traces.

The scorer reads either a Harbor job (saved ``result.json`` plus
``agent/trajectory.json`` files) or the official Terminal-Bench trajectory
release (``*_trajectory.json`` files under ``--trajectory-root``).  It never
invokes Harbor and never regenerates a solver trajectory.

Two deliberately explicit verifier paths are supported:

* ``codex-text`` asks a ChatGPT-subscription Codex model for one literal A-T
  rating per candidate and criterion. This is a discrete 20-bin baseline;
  token logprobs are not available through the Codex CLI.
* ``deepseek-logprob`` / ``gemini-logprob`` require real token logprobs and
  compute the probability-weighted A-T expectation from them.
* ``vllm-logprob`` / ``openrouter-logprob`` connect to an OpenAI-compatible
  endpoint and deliberately use DeepSeek's 32K high-thinking path, not the
  package's generic 4096-token OpenAI-compatible path.  The OpenRouter route
  additionally pins one upstream provider with no fallback.

TSV results go to stdout.  One durable JSONL record is appended after each
completed task so interrupted runs resume without rescoring completed tasks.
Diagnostics go to stderr, so stdout can be piped through ``tee`` without
contaminating the table.

This script is a full-audit protocol: it scores every task (including
unanimous all-pass/all-fail pools) and invokes ``llm_verifier.select`` once per
task, resetting the tournament seed for each task.  That is intentionally not
the paper's exact aggregate ``scripts/run_bo5.py`` protocol, which skips
unanimous pools and advances one global RNG across swing tasks.  The protocol
label is included in every run identity so the two cannot be conflated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VENDORED_LLM_VERIFIER = HERE / "vendor" / "llm-as-a-verifier"
for _path in (REPO_ROOT, VENDORED_LLM_VERIFIER):
    rendered = str(_path)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)

import llm_verifier
import llm_verifier.fine_grained_reward as fine_reward

from experiments.common import candidate_seal
from verifier.llm.codex_service import (
    DEFAULT_SERVICE_TIER,
    SERVICE_TIERS,
    normalize_service_tier,
)

DEFAULT_CACHE_DIR = (
    HERE / "terminal_bench" / "results" / "llm_verifier_cache"
)
OFFICIAL_CRITERIA_PATH = (
    HERE / "vendor" / "llm-as-a-verifier" / "criteria" / "terminal_bench.md"
)
OFFICIAL_CRITERIA_COUNT = 3
DEFAULT_TRACE_CHARS = 0
# Total tries for a stochastic logprob sample whose answer cannot be scored
# strictly (for example, reasoning exhausted the output budget or the final
# score block has no usable token boundary). No probability mass is fabricated
# between tries.
LOGPROB_SAMPLE_ATTEMPTS = 3
VLLM_DEEPSEEK_MAX_TOKENS = 32_768
VLLM_DEEPSEEK_TEMPERATURE = 1.0
VLLM_DEEPSEEK_TOP_P = 1.0
VLLM_DEEPSEEK_TOP_LOGPROBS = 20
VLLM_DEEPSEEK_REASONING_EFFORT = "high"
RESULT_SCHEMA_VERSION = 1
SCORING_PROTOCOL = "all-task-per-task-seed-v1"

BACKEND_DEFAULTS = {
    "codex-text": {
        "model": "gpt-5.6-luna",
        "workers": 1,
        "scoring_mode": "text-argmax-20bin",
    },
    "deepseek-logprob": {
        "model": "deepseek-v4-flash",
        # Upstream terminal_bench_2.1 benchmark setting. A single local task
        # has only ~30 jobs, so this is an upper bound rather than 2,000
        # simultaneous requests in this post-hoc runner.
        "workers": 2_000,
        "scoring_mode": "token-logprob-expectation",
    },
    "gemini-logprob": {
        "model": "gemini-2.5-flash",
        "workers": 6,
        "scoring_mode": "token-logprob-expectation",
    },
    "vllm-logprob": {
        # This is the served name in the Hephaestus vLLM launch script.  It
        # remains overrideable because vLLM's --served-model-name is local
        # deployment configuration.
        "model": "deepseek-v4-flash-0731-h200-high",
        "workers": 8,
        "scoring_mode": "token-logprob-expectation",
    },
    "openrouter-logprob": {
        "model": "deepseek/deepseek-v4-flash",
        "workers": 4,
        "scoring_mode": "token-logprob-expectation",
    },
}

# Require the exact two-line score block at the end. A malformed or truncated
# answer is an error, not a synthetic 0.5 tie.
FINAL_SCORE_BLOCK = re.compile(
    r"<score_A>\s*([A-T])\s*</score_A>\s*\n\s*"
    r"<score_B>\s*([A-T])\s*</score_B>\s*\Z",
    re.IGNORECASE,
)


class RetryableVerifierSampleError(RuntimeError):
    """A completed verifier sample lacks the data needed for strict scoring."""


def codex_request_config(reasoning_effort,
                         service_tier=DEFAULT_SERVICE_TIER):
    """Return score-affecting Codex CLI request settings."""
    return {
        "reasoning_effort": reasoning_effort,
        "service_tier": normalize_service_tier(service_tier),
    }


def vllm_deepseek_request_config(openrouter_provider=None):
    """Return every explicit score-affecting OpenAI-compatible field."""
    config = {
        "max_tokens": VLLM_DEEPSEEK_MAX_TOKENS,
        "temperature": VLLM_DEEPSEEK_TEMPERATURE,
        "top_p": VLLM_DEEPSEEK_TOP_P,
        "logprobs": True,
        "top_logprobs": VLLM_DEEPSEEK_TOP_LOGPROBS,
        "thinking": {"type": "enabled"},
        "reasoning_effort": VLLM_DEEPSEEK_REASONING_EFFORT,
        "chat_template_kwargs": {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_effort": VLLM_DEEPSEEK_REASONING_EFFORT,
        },
    }
    if openrouter_provider:
        config["openrouter_provider"] = {
            "order": [openrouter_provider],
            "allow_fallbacks": False,
        }
    return config


def _object_extra(value, name, default=None):
    """Read a normal or SDK-preserved extra response field."""
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, dict) and extra.get(name) is not None:
        return extra[name]
    if isinstance(value, dict) and value.get(name) is not None:
        return value[name]
    return default


def _append_call_usage(
    client,
    prompt,
    measured,
    *,
    call_id=None,
    response_model=None,
    response_provider=None,
    finish_reason=None,
    cost=0.0,
):
    """Durably append one normalized provider-usage tuple."""
    context = getattr(client, "_gv_usage_context", None)
    path = getattr(client, "_gv_usage_path", None)
    if not context or path is None:
        return
    try:
        cost = float(cost or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    record = {
        "schema_version": 1,
        **context,
        "call_id": call_id,
        "recorded_at_unix_ns": time.time_ns(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_model": response_model,
        "response_provider": response_provider,
        "finish_reason": finish_reason,
        "input_tokens": measured[0],
        "cached_input_tokens": measured[1],
        "output_tokens": measured[2],
        "reasoning_tokens": measured[3],
        "cost_usd": cost,
    }
    usage_path = Path(path)
    lock = getattr(client, "_gv_usage_lock", None)
    if lock is None:
        lock = threading.Lock()
        client._gv_usage_lock = lock
    with lock:
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def record_call_usage(client, response, prompt, finish_reason):
    """Append one credential-free, crash-resilient provider usage record."""
    counter = getattr(fine_reward, "USAGE", None)
    measured = counter.record(response) if counter is not None else (0, 0, 0, 0)
    usage = getattr(response, "usage", None)
    cost = _object_extra(usage, "cost", 0.0) if usage is not None else 0.0
    _append_call_usage(
        client,
        prompt,
        measured,
        call_id=_object_extra(response, "id", None),
        response_model=_object_extra(response, "model", None),
        response_provider=_object_extra(response, "provider", None),
        finish_reason=finish_reason,
        cost=cost,
    )


def record_codex_call_usage(client, prompt, usage, call_id=None):
    """Record one Codex CLI call without estimating any missing counters."""
    if usage is None:
        raise RuntimeError("Codex CLI completed without reporting token usage")
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    if any(
        not isinstance(usage.get(field), int)
        or isinstance(usage.get(field), bool)
        or usage[field] < 0
        for field in fields
    ):
        raise RuntimeError("Codex CLI reported invalid token usage")
    measured = tuple(usage[field] for field in fields)
    counter = getattr(fine_reward, "USAGE", None)
    add = getattr(counter, "add", None)
    if callable(add):
        add(
            input_tokens=measured[0],
            cached_input_tokens=measured[1],
            output_tokens=measured[2],
            reasoning_tokens=measured[3],
        )
    _append_call_usage(
        client,
        prompt,
        measured,
        call_id=call_id,
        response_model=getattr(client, "model", None),
        response_provider="codex-cli",
        finish_reason="stop",
    )


def aggregate_call_usage(path, run_id, task):
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    path = Path(path)
    if not path.is_file():
        return totals
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id") != run_id or record.get("task") != task:
                continue
            totals["calls"] += 1
            for field in (
                "input_tokens", "cached_input_tokens", "output_tokens",
                "reasoning_tokens",
            ):
                totals[field] += int(record.get(field, 0) or 0)
            totals["cost_usd"] += float(record.get("cost_usd", 0.0) or 0.0)
    totals["uncached_input_tokens"] = (
        totals["input_tokens"] - totals["cached_input_tokens"]
    )
    return totals


def reward_of(result):
    return float(result["verifier_result"]["rewards"]["reward"])


def load_trace(trial_dir):
    return json.loads(
        (trial_dir / "agent" / "trajectory.json").read_text(encoding="utf-8")
    )


def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(value)


def get_problem(trajectory, task):
    # Codex trajectories contain a user-role metadata preamble (for example
    # <recommended_plugins>) before the benchmark prompt. Harbor supplies the
    # actual task last, so use the final non-empty user message.
    messages = []
    for step in trajectory.get("steps", []):
        if step.get("source") == "user":
            message = text(step.get("message", "")).strip()
            if message:
                messages.append(message)
    return messages[-1] if messages else f"Terminal-Bench task: {task}"


def format_trace(trajectory, max_chars=DEFAULT_TRACE_CHARS):
    out = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue

        out.append(f"--- Agent Step {step.get('step_id', '?')} ---")

        message = text(step.get("message", ""))
        if message:
            out.append(message)

        for tool_call in step.get("tool_calls", []) or []:
            name = tool_call.get("function_name", "tool")
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, dict) and "keystrokes" in arguments:
                out.append(f"[Command] {str(arguments['keystrokes']).rstrip()}")
            else:
                out.append(
                    f"[Tool Call] {name} "
                    f"{json.dumps(arguments, ensure_ascii=False)}"
                )

        observation = step.get("observation") or {}
        for result in observation.get("results", []) or []:
            content = text(result.get("content", ""))
            if content:
                out.append(f"[Output]\n{content}")

        out.append("")

    rendered = "\n".join(out)
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    marker = "[... earlier trace characters omitted ...]\n"
    tail_chars = max(0, max_chars - len(marker))
    return marker + rendered[-tail_chars:]


def letter_score(letter):
    """Map A (best) ... T (worst) linearly into [1, 0]."""
    value = letter.upper()
    if len(value) != 1 or not "A" <= value <= "T":
        raise ValueError(f"not an A-T score letter: {letter!r}")
    return (ord("T") - ord(value)) / 19.0


def parse_text_scores(raw):
    match = FINAL_SCORE_BLOCK.search(raw or "")
    if not match:
        raise RuntimeError(
            "verifier did not end with a valid two-line A-T score block"
        )
    return tuple(letter_score(letter) for letter in match.groups())


def score_pair_text(
    client,
    problem,
    trace_a,
    trace_b,
    criterion,
    ground_truth_note,
    model="gpt-5.6-luna",
    images=None,
):
    """One text-only A-T rating, bypassing llm-verifier's logprob prefill."""
    if fine_reward.as_image_list(images):
        raise ValueError("codex-text does not support attached images")
    prompt = fine_reward.build_prompt(
        problem,
        trace_a,
        trace_b,
        criterion,
        ground_truth_note,
        n_images=0,
    )
    complete_with_usage = getattr(client, "complete_with_usage", None)
    if callable(complete_with_usage):
        raw, usage, call_id = complete_with_usage(prompt)
        record_codex_call_usage(client, prompt, usage, call_id)
    else:
        # Keep the scorer's lightweight client duck type for tests and custom
        # text backends; CodexRunner itself always takes the measured path.
        raw = client.complete(prompt)
    return parse_text_scores(raw)


def _tag_logprobs(tokens, position_logprobs, tag):
    """Find the alternatives immediately after the final score tag."""
    if not tokens or not position_logprobs:
        return None
    for suffix in (tag, tag[:-1]):
        found = None
        text_so_far = ""
        for index, token in enumerate(tokens):
            text_so_far += token
            # A whitespace token after a tag must not shadow the actual
            # distribution captured at the preceding position.
            if not token.strip():
                continue
            if (
                text_so_far.rstrip().endswith(suffix)
                and index + 1 < len(position_logprobs)
            ):
                found = position_logprobs[index + 1]
        if found is not None:
            return found
    return None


def expected_logprob_score(tokens, position_logprobs, tag):
    """Strict A-T expectation; missing logprob mass raises instead of tying."""
    alternatives = _tag_logprobs(tokens, position_logprobs, tag) or []
    if not alternatives:
        raise RetryableVerifierSampleError(
            f"score-tag token boundary not found at {tag}"
        )
    probabilities = {}
    for token, logprob in alternatives:
        normalized = token.strip()
        if normalized.startswith(">"):
            normalized = normalized[1:].strip()
        normalized = normalized.upper()
        if len(normalized) != 1 or not "A" <= normalized <= "T":
            continue
        probability = math.exp(float(logprob))
        if not math.isfinite(probability) or probability <= 0:
            continue
        # Some tokenizers expose duplicate spellings (for example A and
        # leading-space A); mirror upstream behavior and keep the larger mass.
        score = letter_score(normalized)
        probabilities[score] = max(probabilities.get(score, 0.0), probability)

    total = sum(probabilities.values())
    if total <= 0:
        visible = [str(token)[:40] for token, _logprob in alternatives[:25]]
        raise RetryableVerifierSampleError(
            f"no A-T token-logprob mass found at {tag}; "
            f"available tokens={visible!r}"
        )
    return sum(score * prob for score, prob in probabilities.items()) / total


def call_vllm_deepseek(
    client,
    prompt,
    model,
    images=None,
):
    """Call local vLLM through its OpenAI API using DeepSeek reasoning.

    Upstream's generic OpenAI-compatible route intentionally disables hybrid
    thinking and caps output at 4096 tokens before doing score-tag prefill.
    That is useful for generic open models but is not the published
    DeepSeek-v4-flash verifier configuration.  Keep this call explicit so the
    local backend always requests high thinking and a 32768-token budget.
    """
    if fine_reward.as_image_list(images):
        raise ValueError("OpenAI-compatible DeepSeek backend is text-only")

    request_config = vllm_deepseek_request_config(
        getattr(client, "_gv_openrouter_provider", None)
    )
    extra_body = {
        "thinking": request_config["thinking"],
        "reasoning_effort": request_config["reasoning_effort"],
        # vLLM forwards these to the DeepSeek-v4 tokenizer.  Supplying
        # them per request makes the scorer correct even if the server
        # was launched without --default-chat-template-kwargs.
        "chat_template_kwargs": request_config["chat_template_kwargs"],
    }
    if "openrouter_provider" in request_config:
        extra_body["provider"] = request_config["openrouter_provider"]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=request_config["max_tokens"],
        temperature=request_config["temperature"],
        top_p=request_config["top_p"],
        logprobs=request_config["logprobs"],
        top_logprobs=request_config["top_logprobs"],
        extra_body=extra_body,
    )
    choice = response.choices[0]
    record_call_usage(client, response, prompt, choice.finish_reason)
    raw = choice.message.content or ""
    content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content_logprobs:
        raise RetryableVerifierSampleError(
            "DeepSeek vLLM returned no answer logprobs "
            f"(finish_reason={choice.finish_reason!r}, reasoning consumed the "
            f"{VLLM_DEEPSEEK_MAX_TOKENS}-token budget)"
        )

    tokens = []
    position_logprobs = []
    for position in content_logprobs:
        tokens.append(position.token)
        alternatives = [
            (alternative.token, alternative.logprob)
            for alternative in (position.top_logprobs or [])
        ]
        # Some OpenAI-compatible providers omit the actually sampled token
        # from top_logprobs.  Its own logprob is still authoritative and must
        # be part of the available distribution; otherwise an emitted A-T
        # score can be impossible to integrate even though the response is
        # structurally valid.  This restores provider-supplied data only -- it
        # does not synthesize missing letters or a fallback tie.
        if not any(token == position.token for token, _value in alternatives):
            alternatives.append((position.token, position.logprob))
        position_logprobs.append(alternatives)
    return raw, tokens, position_logprobs


def score_pair_logprob(
    client,
    problem,
    trace_a,
    trace_b,
    criterion,
    ground_truth_note,
    model="gemini-2.5-flash",
    images=None,
):
    """One probability-weighted A-T rating with strict logprob validation."""
    imgs = fine_reward.as_image_list(images)
    prompt = fine_reward.build_prompt(
        problem,
        trace_a,
        trace_b,
        criterion,
        ground_truth_note,
        n_images=len(imgs),
    )
    # Reasoning and the emitted score block are stochastic at temperature 1.0.
    # Resample a response that has no strictly usable A-T distribution (for
    # example, an output-budget overflow or a malformed/missing score-tag
    # boundary). Provider/API failures still raise immediately, and no missing
    # probability mass or fallback tie is fabricated.
    attempts = LOGPROB_SAMPLE_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            if getattr(client, "_gv_vllm_deepseek", False):
                _raw, tokens, position_logprobs = call_vllm_deepseek(
                    client, prompt, model, images=imgs
                )
            else:
                _raw, tokens, position_logprobs = fine_reward.call_verifier(
                    client, prompt, model, images=imgs
                )
            return (
                expected_logprob_score(
                    tokens, position_logprobs, "<score_A>"
                ),
                expected_logprob_score(
                    tokens, position_logprobs, "<score_B>"
                ),
            )
        except RetryableVerifierSampleError as error:
            if attempt == attempts:
                raise
            print(
                f"[llm-verifier] unusable logprob sample ({error}); "
                f"resampling ({attempt}/{attempts - 1} retries used)",
                file=sys.stderr,
                flush=True,
            )
    raise AssertionError("unreachable logprob sample retry loop")


@contextmanager
def patched_pair_scorer(scorer):
    """Use a strict scorer inside llm-verifier's unchanged PPT machinery."""
    original = fine_reward.score_pair_criterion
    fine_reward.score_pair_criterion = scorer
    try:
        yield
    finally:
        fine_reward.score_pair_criterion = original


def slug(value, limit=80):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (cleaned or "item")[:limit]


def stable_digest(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scorer_source_hash():
    """Fingerprint this driver and the imported llm_verifier implementation."""
    package_root = Path(llm_verifier.__file__).resolve().parent
    sources = {
        "experiments/score_llm_verifier.py": file_digest(Path(__file__).resolve()),
        "experiments/common/codex_endpoint.py": file_digest(
            HERE / "common" / "codex_endpoint.py"),
        "verifier/llm/codex_service.py": file_digest(
            REPO_ROOT / "verifier" / "llm" / "codex_service.py"),
    }
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        sources[f"llm_verifier/{relative}"] = file_digest(path)
    return stable_digest({
        "schema": "llm-verifier-scorer-source-v1",
        "sources": sources,
    })


def endpoint_identity(backend, base_url=None):
    """Return a stable endpoint identifier that cannot contain credentials."""
    if backend in {"vllm-logprob", "openrouter-logprob"}:
        raw = base_url if base_url is not None else os.environ.get(
            "OPENAI_BASE_URL", ""
        )
        if not raw:
            return {"kind": "openai-compatible", "endpoint": "unconfigured"}
        parsed = urlsplit(raw)
        hostname = parsed.hostname or "unknown-host"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        origin = f"{parsed.scheme.lower()}://{hostname.lower()}"
        if parsed.port is not None:
            origin += f":{parsed.port}"
        # The path distinguishes multiple servers behind one origin, but is
        # hashed in case an operator put a credential-like value there.  URL
        # userinfo, query parameters, and fragments are never retained.
        canonical = f"{origin}{parsed.path.rstrip('/')}"
        return {
            "kind": (
                "openrouter-openai-compatible"
                if backend == "openrouter-logprob"
                else "openai-compatible"
            ),
            "origin": origin,
            "endpoint_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }
    if backend == "deepseek-logprob":
        return {"kind": "deepseek-hosted", "origin": "https://api.deepseek.com"}
    if backend == "gemini-logprob":
        return {"kind": "gemini-vertex"}
    if backend == "codex-text":
        return {"kind": "codex-cli"}
    raise AssertionError(f"unhandled backend: {backend}")


def task_trace_identity(task, rows):
    """Fingerprint only inputs visible to the verifier, never gold reward."""
    return stable_digest(
        {
            "task": task,
            "rows": [
                {
                    "trial": row["trial"],
                    "problem": row["problem"],
                    "trace": row["trace"],
                    "source": row.get("source"),
                    "source_sha256": row.get("source_sha256"),
                }
                for row in sorted(rows, key=lambda item: item["trial"])
            ],
        }
    )


def dataset_identity(kind, tasks):
    """Location-independent content-tree identity for a loaded dataset."""
    tree = {
        task: [
            {
                "trial": row["trial"],
                "source": row.get("source"),
                "source_sha256": row.get("source_sha256"),
            }
            for row in sorted(rows, key=lambda item: item["trial"])
        ]
        for task, rows in sorted(tasks.items())
    }
    return {
        "kind": kind,
        "name": kind,
        "tree_sha256": stable_digest(tree),
    }


def run_identity_payload(
    dataset,
    tasks,
    backend,
    model,
    served_model_identity,
    endpoint,
    scoring_mode,
    scorer_source_sha256,
    request_config,
    n_evaluations,
    pivots,
    seed,
    trace_chars,
):
    """All score-affecting run inputs, excluding every provider secret."""
    return {
        "dataset": dataset,
        "task_traces": {
            task: task_trace_identity(task, rows)
            for task, rows in sorted(tasks.items())
        },
        "backend": backend,
        "model": model,
        "served_model_identity": served_model_identity,
        "endpoint": endpoint,
        "scoring_mode": scoring_mode,
        "scorer_source_sha256": scorer_source_sha256,
        "request_config": request_config,
        "criteria_sha256": file_digest(OFFICIAL_CRITERIA_PATH),
        "n_evaluations": n_evaluations,
        "pivots": pivots,
        "seed": seed,
        "trace_chars": trace_chars,
        "protocol": SCORING_PROTOCOL,
    }


def cache_file_for(
    cache_dir,
    dataset,
    task,
    rows,
    backend,
    model,
    served_model_identity,
    endpoint,
    scorer_source_sha256,
    request_config,
):
    # llm-verifier.select internally names every synthetic task simply
    # "task", and its cache key does not include prompt/model/backend. Keep a
    # separate, content-addressed cache file to prevent cross-task score reuse.
    payload = {
        "dataset": dataset,
        "task": task,
        "trace_identity": task_trace_identity(task, rows),
        "criteria_file": OFFICIAL_CRITERIA_PATH.read_text(encoding="utf-8"),
        "backend": backend,
        "model": model,
        "served_model_identity": served_model_identity,
        "endpoint": endpoint,
        "scorer_source_sha256": scorer_source_sha256,
        "request_config": request_config,
    }
    digest = stable_digest(payload)[:16]
    return (
        cache_dir
        / slug(dataset["name"])
        / slug(backend)
        / slug(model)
        / f"{slug(task)}-{digest}.json"
    )


def load_harbor_tasks(job_dir, trace_chars, selected_tasks):
    tasks = defaultdict(list)
    for result_file in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        trial_dir = result_file.parent
        trajectory_file = trial_dir / "agent" / "trajectory.json"
        if not trajectory_file.exists():
            print(f"WARNING: missing trajectory: {trajectory_file}", file=sys.stderr)
            continue

        trajectory = load_trace(trial_dir)
        task = result.get("task_name", trial_dir.name).split("/", 1)[-1]
        if selected_tasks and task not in selected_tasks:
            continue
        tasks[task].append(
            {
                "trial": result.get("trial_name", trial_dir.name),
                "reward": reward_of(result),
                "problem": get_problem(trajectory, task),
                "trace": format_trace(trajectory, trace_chars),
                "source": str(result_file.relative_to(job_dir)),
                # Harbor's top-level result contains held-out reward.  Only
                # the solver trajectory is a verifier input and cache key.
                "source_sha256": file_digest(trajectory_file),
            }
        )
    return dict(tasks)


def parse_candidate_path_maps(values):
    """Parse unambiguous absolute ``OLD=NEW`` relocation roots."""
    mappings = []
    for value in values or ():
        old_text, separator, new_text = value.partition("=")
        if not separator or not old_text.strip() or not new_text.strip():
            raise ValueError(
                "candidate path maps must use non-empty absolute OLD=NEW roots"
            )
        old = Path(os.path.normpath(old_text.strip())).expanduser()
        new = Path(os.path.normpath(new_text.strip())).expanduser()
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("candidate path map roots must be absolute")
        if old == Path(old.anchor):
            raise ValueError("candidate path map OLD may not be a filesystem root")
        mappings.append((old, new))
    for index, (old, _new) in enumerate(mappings):
        for other_index, (other, _other_new) in enumerate(mappings):
            if index == other_index:
                continue
            if old == other or old.is_relative_to(other):
                raise ValueError(
                    "candidate path map OLD roots must be unique and non-overlapping"
                )
    return tuple(mappings)


def remap_candidate_spec_paths(spec, path_maps):
    """Return a copy with only top-level ``*_path`` locations relocated."""
    relocated = dict(spec)
    for field, value in spec.items():
        if not field.endswith("_path") or not isinstance(value, str):
            continue
        source = Path(value).expanduser()
        matches = []
        if source.is_absolute():
            for old, new in path_maps:
                if source == old or source.is_relative_to(old):
                    matches.append((old, new))
        if len(matches) > 1:
            raise candidate_seal.SealError(
                f"candidate path matches multiple relocation roots: {value}"
            )
        if matches:
            old, new = matches[0]
            relocated[field] = str(new / source.relative_to(old))
    return relocated


def load_frozen_candidate_tasks(
    specs_path,
    seals_path,
    verifier_inputs_path,
    trace_chars,
    selected_tasks,
    candidate_path_maps=(),
):
    """Load one immutable, reward-hidden Harbor candidate manifest.

    ``verifier_inputs_path`` is the public join boundary.  It contains no
    held-out reward and must equal :func:`candidate_seal.verifier_view` for
    every private seal.  ``specs_path`` and ``seals_path`` are read only to
    resolve the already-bound trajectory path and to retain reward for the
    post-hoc BoN report; neither object is passed to ``llm_verifier.select``.

    Recomputing each seal also makes a trajectory, task config, or artifact
    mutation fail closed before any model call.  The returned dataset identity
    is the pinned Terminal-Bench dataset digest rather than a digest of the
    incrementally growing manifest.  Per-task cache identity remains bound to
    the exact candidate IDs and rendered traces, so calls made for a complete
    five-way pool are reusable when other task pools are appended later.
    """
    paths = [
        Path(specs_path).expanduser().resolve(),
        Path(seals_path).expanduser().resolve(),
        Path(verifier_inputs_path).expanduser().resolve(),
    ]
    if len(set(paths)) != len(paths):
        raise candidate_seal.SealError(
            "candidate specs, seals, and verifier inputs must be distinct files"
        )
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise candidate_seal.SealError(
                f"expected a regular frozen-candidate JSONL file: {path}"
            )
    before = {path: candidate_seal.file_digest(path) for path in paths}
    specs = candidate_seal.load_jsonl(paths[0])
    seals = candidate_seal.load_jsonl(paths[1])
    verifier_inputs = candidate_seal.load_jsonl(paths[2])
    if not specs or not seals or not verifier_inputs:
        raise candidate_seal.SealError("frozen candidate manifest is empty")

    def source_key(value):
        identity = value.get("identity") if isinstance(value, dict) else None
        if not isinstance(identity, dict):
            raise candidate_seal.SealError(
                "frozen candidate has no identity object"
            )
        dataset = identity.get("dataset")
        task = identity.get("task")
        if not isinstance(dataset, dict) or not isinstance(task, dict):
            raise candidate_seal.SealError(
                "frozen candidate has malformed dataset/task identity"
            )
        return dataset.get("name"), task.get("name"), identity.get("trial")

    seals_by_id = {}
    seals_by_source = {}
    for seal in seals:
        candidate_seal.validate_seal(seal)
        candidate_id = seal["candidate_id"]
        key = source_key(seal)
        if candidate_id in seals_by_id or key in seals_by_source:
            raise candidate_seal.SealError(
                "duplicate candidate in private seals"
            )
        seals_by_id[candidate_id] = seal
        seals_by_source[key] = seal

    views_by_id = {}
    for view in verifier_inputs:
        candidate_id = view.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in views_by_id:
            raise candidate_seal.SealError(
                "duplicate or missing candidate_id in verifier inputs"
            )
        seal = seals_by_id.get(candidate_id)
        if seal is None or view != candidate_seal.verifier_view(seal):
            raise candidate_seal.SealError(
                "verifier input does not exactly match its reward-free seal view"
            )
        views_by_id[candidate_id] = view
    if set(views_by_id) != set(seals_by_id):
        raise candidate_seal.SealError(
            "private seals and reward-free verifier inputs have different candidates"
        )

    specs_by_source = {}
    for spec in specs:
        if spec.get("schema") != "harbor-rec-codex-candidate/v1":
            raise candidate_seal.SealError(
                "candidate spec is not a frozen Harbor Codex candidate"
            )
        key = (spec.get("dataset"), spec.get("task"), spec.get("trial"))
        if key in specs_by_source:
            raise candidate_seal.SealError("duplicate frozen candidate spec")
        specs_by_source[key] = spec
    if set(specs_by_source) != set(seals_by_source):
        raise candidate_seal.SealError(
            "candidate specs and private seals have different sources"
        )

    tasks = defaultdict(list)
    datasets = set()
    pools_by_task = defaultdict(set)
    for key, seal in sorted(seals_by_source.items()):
        spec = specs_by_source[key]
        resolved_spec = remap_candidate_spec_paths(spec, candidate_path_maps)
        recomputed = candidate_seal.seal_from_spec(resolved_spec)
        if recomputed != seal:
            raise candidate_seal.SealError(
                f"frozen candidate no longer matches its seal: {key[1]}/{key[2]}"
            )
        view = views_by_id[seal["candidate_id"]]
        identity = view["identity"]
        dataset_identity_value = identity["dataset"]
        datasets.add((
            dataset_identity_value["name"],
            dataset_identity_value["sha256"],
        ))
        full_task = identity["task"]["name"]
        task = full_task.split("/", 1)[-1]
        if selected_tasks and task not in selected_tasks \
                and full_task not in selected_tasks:
            continue
        pools_by_task[task].add(view["pool_id"])
        trajectory_path = Path(
            resolved_spec["trajectory_path"]
        ).expanduser().resolve()
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        tasks[task].append(
            {
                "candidate_id": seal["candidate_id"],
                "trial": identity["trial"],
                # Evaluation metadata stays outside the public verifier view
                # and is used only after pairwise scoring.
                "reward": float(seal["evaluation"]["reward"]),
                "problem": get_problem(trajectory, task),
                "trace": format_trace(trajectory, trace_chars),
                "source": f"candidate:{seal['candidate_id']}",
                "source_sha256": identity["trajectory"]["sha256"],
            }
        )

    if len(datasets) != 1:
        raise candidate_seal.SealError(
            "frozen candidates must belong to exactly one pinned dataset"
        )
    for task, pool_ids in pools_by_task.items():
        if len(pool_ids) != 1:
            raise candidate_seal.SealError(
                f"task {task} mixes candidates from different solver pools"
            )
    if any(candidate_seal.file_digest(path) != digest
           for path, digest in before.items()):
        raise candidate_seal.SealError(
            "frozen candidate manifest changed while loading"
        )
    dataset_name, dataset_digest = next(iter(datasets))
    dataset = {
        "kind": "frozen-harbor-candidates",
        "name": dataset_name,
        "dataset_sha256": dataset_digest,
    }
    return dict(tasks), dataset


def official_problem(trajectory, task):
    """Match the problem extraction used by the official TB 2.1 loader."""
    for step in trajectory.get("steps", []):
        if step.get("source") != "user":
            continue
        message = text(step.get("message", ""))
        if message and not (message.startswith("$") and len(message) < 5):
            return message

    initial_analysis = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        message = text(step.get("message", ""))
        if message:
            initial_analysis.append(message)
        if len(initial_analysis) >= 2:
            break
    if initial_analysis:
        return (
            f"[Task: {task}]\nThe original task instruction was not captured. "
            "Below is the agent's initial analysis:\n\n"
            + "\n\n".join(initial_analysis)
        )
    return f"(Task: {task})"


def load_official_tasks(trajectory_root, trace_chars, selected_tasks):
    """Load the official ``*_trajectory.json`` Terminal-Bench release."""
    tasks = defaultdict(list)
    seen_trials = set()
    files = sorted(trajectory_root.rglob("*_trajectory.json"))
    if not files:
        return {}

    for trajectory_file in files:
        try:
            payload = json.loads(trajectory_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid official trajectory JSON: {trajectory_file}: {error}"
            ) from error

        trajectory = payload.get("trajectory")
        task = payload.get("task_name")
        trial = payload.get("trial_name")
        if not isinstance(trajectory, dict) or not task or not trial:
            raise ValueError(
                f"official trajectory is missing task/trial/trajectory: "
                f"{trajectory_file}"
            )
        if selected_tasks and task not in selected_tasks:
            continue
        duplicate_key = (task, trial)
        if duplicate_key in seen_trials:
            raise ValueError(
                f"duplicate official trajectory for {task}/{trial}: "
                f"{trajectory_file}"
            )
        seen_trials.add(duplicate_key)
        try:
            reward = float(payload["reward"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"official trajectory has invalid reward: {trajectory_file}"
            ) from error

        tasks[task].append(
            {
                "trial": trial,
                "reward": reward,
                "problem": official_problem(trajectory, task),
                "trace": format_trace(trajectory, trace_chars),
                "source": str(trajectory_file.relative_to(trajectory_root)),
                # Gold reward is evaluation metadata, never a scoring input.
                # Hash canonical JSON so formatting and reward edits do not
                # invalidate verifier caches or run identity.
                "source_sha256": stable_digest(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "reward"
                    }
                ),
            }
        )
    return dict(tasks)


def default_results_path(cache_dir, dataset, backend, model, run_id):
    return (
        cache_dir
        / "task_results"
        / f"{slug(dataset['name'])}-{slug(backend)}-{slug(model)}-{run_id[:16]}.jsonl"
    )


def load_completed_records(path, run_id):
    """Read valid completed-task records; tolerate a crash-truncated line."""
    completed = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: ignoring malformed JSONL line "
                    f"{path}:{line_number}",
                    file=sys.stderr,
                )
                continue
            if (
                record.get("schema_version") != RESULT_SCHEMA_VERSION
                or record.get("run_id") != run_id
                or not isinstance(record.get("task"), str)
            ):
                continue
            if not (
                isinstance(record.get("scores"), list)
                and isinstance(record.get("ranking"), list)
            ):
                print(
                    f"WARNING: ignoring incomplete task record "
                    f"{path}:{line_number}",
                    file=sys.stderr,
                )
                continue
            completed[record["task"]] = record
    return completed


def append_task_record(path, record):
    """Durably append one task, preserving a prior crash-truncated record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        json.dump(record, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def print_tsv_rows(task, rows, scores, ranking, backend, model, scoring_mode):
    rank = {
        candidate_index: position + 1
        for position, candidate_index in enumerate(ranking)
    }
    for index, row in enumerate(rows):
        print(
            f"{task}\t"
            f"{row['trial']}\t"
            f"{row['reward']:.0f}\t"
            f"{scores[index]:.6f}\t"
            f"{rank[index]}\t"
            f"{backend}\t"
            f"{model}\t"
            f"{scoring_mode}\t"
            f"{SCORING_PROTOCOL}",
            flush=True,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc LLM-as-a-Verifier scoring for a Harbor job or the "
            "official Terminal-Bench trajectory release; does not run Harbor "
            "or regenerate trajectories."
        )
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        nargs="?",
        help=(
            "Harbor job directory containing one subdirectory per trial; "
            "mutually exclusive with --trajectory-root"
        ),
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        help=(
            "official Terminal-Bench release root containing recursive "
            "*_trajectory.json files (for TB 2.1: 445 files, five per task)"
        ),
    )
    parser.add_argument(
        "--candidate-specs",
        type=Path,
        help=(
            "private frozen Harbor candidate specs; requires "
            "--candidate-seals and --candidate-verifier-inputs and is "
            "mutually exclusive with job_dir/--trajectory-root"
        ),
    )
    parser.add_argument(
        "--candidate-seals",
        type=Path,
        help="private immutable candidate seals for --candidate-specs",
    )
    parser.add_argument(
        "--candidate-verifier-inputs",
        type=Path,
        help=(
            "reward-free candidate views forming the public verifier join "
            "boundary for --candidate-specs"
        ),
    )
    parser.add_argument(
        "--candidate-path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "relocate absolute *_path fields in frozen candidate specs "
            "without changing content identity (repeatable; only valid with "
            "--candidate-specs)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_DEFAULTS),
        required=True,
        help=(
            "explicit verifier path: Codex text-only A-T, DeepSeek logprobs, "
            "local vLLM DeepSeek logprobs, or Gemini Vertex logprobs"
        ),
    )
    parser.add_argument(
        "--model",
        help="model id (default depends on --backend)",
    )
    parser.add_argument(
        "--served-model-identity",
        help=(
            "immutable checkpoint/deployment identity behind --model; required "
            "for vllm/openrouter logprob backends so a reused served alias "
            "cannot hit old scores"
        ),
    )
    parser.add_argument(
        "--openrouter-provider",
        help=(
            "force one OpenRouter provider and disable fallbacks; only valid "
            "with --backend openrouter-logprob and recorded in request "
            "identity"
        ),
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=2,
        help=(
            "repeated ratings K per criterion (default: 2, published TB 2.1 "
            "Bo3/Bo5 preset)"
        ),
    )
    parser.add_argument(
        "--pivots",
        type=int,
        default=1,
        help=(
            "PPT pivot count (default: 1, published TB 2.1 Bo3/Bo5 preset)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "parallel verifier calls (default: 1 for Codex, 2000 for the "
            "official hosted DeepSeek preset, 8 for local vLLM, 6 for Gemini)"
        ),
    )
    parser.add_argument(
        "--codex-effort",
        default="low",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="reasoning effort for --backend codex-text (default: low)",
    )
    parser.add_argument(
        "--codex-service-tier",
        default=DEFAULT_SERVICE_TIER,
        choices=SERVICE_TIERS,
        help="service tier for --backend codex-text (default: standard)",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=600,
        help="seconds allowed for each codex-text call (default: 600)",
    )
    parser.add_argument(
        "--trace-chars",
        type=int,
        default=DEFAULT_TRACE_CHARS,
        help="keep this many trailing characters per trace (default: 0, no cap)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory for isolated raw pair-score caches",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="score only this task (repeatable; default: all tasks)",
    )
    parser.add_argument(
        "--expected-candidates",
        type=int,
        help=(
            "warn when a task has a different candidate count (for the "
            "official TB 2.1 release use 5); default: accept any count"
        ),
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        help=(
            "durable per-task result log used for resume (default: a "
            "content-addressed path under --cache-dir)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the planned workload without scoring",
    )
    args = parser.parse_args(argv)

    sources = (
        args.job_dir is not None,
        args.trajectory_root is not None,
        args.candidate_specs is not None,
    )
    if sum(sources) != 1:
        parser.error(
            "provide exactly one of job_dir, --trajectory-root, or "
            "--candidate-specs"
        )
    frozen_companions = (
        args.candidate_seals is not None,
        args.candidate_verifier_inputs is not None,
    )
    if args.candidate_specs is not None and not all(frozen_companions):
        parser.error(
            "--candidate-specs requires --candidate-seals and "
            "--candidate-verifier-inputs"
        )
    if args.candidate_specs is None and any(frozen_companions):
        parser.error(
            "--candidate-seals/--candidate-verifier-inputs require "
            "--candidate-specs"
        )
    if args.candidate_specs is None and args.candidate_path_map:
        parser.error("--candidate-path-map requires --candidate-specs")
    try:
        args.candidate_path_map = parse_candidate_path_maps(
            args.candidate_path_map
        )
    except ValueError as error:
        parser.error(str(error))
    if args.n_evaluations < 1:
        parser.error("--n-evaluations must be at least 1")
    if args.pivots < 0:
        parser.error("--pivots must be non-negative")
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.codex_timeout < 1:
        parser.error("--codex-timeout must be at least 1")
    if (args.codex_service_tier == "fast"
            and args.backend != "codex-text"):
        parser.error("--codex-service-tier fast requires --backend codex-text")
    if args.trace_chars < 0:
        parser.error("--trace-chars must be non-negative")
    if args.expected_candidates is not None and args.expected_candidates < 1:
        parser.error("--expected-candidates must be at least 1")
    if args.served_model_identity is not None:
        args.served_model_identity = args.served_model_identity.strip()
        if not args.served_model_identity:
            parser.error("--served-model-identity may not be empty")
    if args.backend in {"vllm-logprob", "openrouter-logprob"} \
            and not args.served_model_identity:
        parser.error(
            "--served-model-identity is required for vllm/openrouter logprob"
        )
    if args.openrouter_provider is not None:
        args.openrouter_provider = args.openrouter_provider.strip()
        if not args.openrouter_provider:
            parser.error("--openrouter-provider may not be empty")
        if args.backend != "openrouter-logprob":
            parser.error(
                "--openrouter-provider requires openrouter-logprob"
            )
    if args.backend == "openrouter-logprob" and not args.openrouter_provider:
        parser.error(
            "--openrouter-provider is required for openrouter-logprob"
        )
    return args


def configure_backend(args, model, workers):
    if args.backend == "codex-text":
        # Import only for this backend. It invokes the already-authenticated
        # local Codex CLI directly; no OpenAI API key or proxy is involved.
        from common.codex_endpoint import CodexRunner

        client = CodexRunner(
            model=model,
            effort=args.codex_effort,
            timeout=args.codex_timeout,
            concurrency=workers,
            service_tier=args.codex_service_tier,
        )
        client._gv_usage_lock = threading.Lock()
        return client, score_pair_text
    if args.backend == "deepseek-logprob":
        return fine_reward.create_deepseek_client(model=model), score_pair_logprob
    if args.backend == "gemini-logprob":
        return fine_reward.create_gemini_client(), score_pair_logprob
    if args.backend in {"vllm-logprob", "openrouter-logprob"}:
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not base_url:
            raise RuntimeError(
                "set OPENAI_BASE_URL to the local vLLM /v1 endpoint"
            )
        if not api_key:
            raise RuntimeError(
                "set OPENAI_API_KEY to the key configured by the local vLLM server"
            )
        client = fine_reward.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # score_pair_logprob checks this private tag and bypasses upstream's
        # generic 4096-token, thinking-disabled OpenAI-compatible path.
        client._gv_vllm_deepseek = True
        client._gv_usage_lock = threading.Lock()
        client._gv_openrouter_provider = (
            args.openrouter_provider
            if args.backend == "openrouter-logprob"
            else None
        )
        client._llm_verifier_deepseek = True
        client._llm_verifier_model = model
        return client, score_pair_logprob
    raise AssertionError(f"unhandled backend: {args.backend}")


def main(argv=None):
    args = parse_args(argv)
    if not OFFICIAL_CRITERIA_PATH.is_file():
        raise SystemExit(
            f"Official Terminal-Bench criteria file is missing: "
            f"{OFFICIAL_CRITERIA_PATH}"
        )

    defaults = BACKEND_DEFAULTS[args.backend]
    model = args.model or defaults["model"]
    served_model_identity = args.served_model_identity or model
    workers = args.max_workers or defaults["workers"]
    scoring_mode = defaults["scoring_mode"]
    source_sha256 = scorer_source_hash()
    if args.backend == "codex-text":
        request_config = codex_request_config(
            args.codex_effort, args.codex_service_tier)
    elif args.backend in {"vllm-logprob", "openrouter-logprob"}:
        request_config = vllm_deepseek_request_config(args.openrouter_provider)
    else:
        request_config = None
    cache_dir = args.cache_dir.expanduser().resolve()

    if args.candidate_specs is not None:
        source_root = args.candidate_verifier_inputs.expanduser().resolve()
        try:
            loaded_tasks, dataset = load_frozen_candidate_tasks(
                args.candidate_specs,
                args.candidate_seals,
                args.candidate_verifier_inputs,
                args.trace_chars,
                set(),
                args.candidate_path_map,
            )
        except (OSError, ValueError, candidate_seal.SealError) as error:
            raise SystemExit(str(error)) from error
    elif args.trajectory_root is not None:
        source_root = args.trajectory_root.expanduser().resolve()
        if not source_root.is_dir():
            raise SystemExit(
                f"Trajectory root does not exist: {source_root}"
            )
        dataset_kind = "official-terminal-bench-trajectories"
        try:
            loaded_tasks = load_official_tasks(
                source_root, args.trace_chars, set()
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        dataset = dataset_identity(dataset_kind, loaded_tasks)
    else:
        source_root = args.job_dir.expanduser().resolve()
        if not source_root.is_dir():
            raise SystemExit(f"Job directory does not exist: {source_root}")
        dataset_kind = "harbor-job"
        loaded_tasks = load_harbor_tasks(
            source_root, args.trace_chars, set()
        )
        dataset = dataset_identity(dataset_kind, loaded_tasks)
    selected_tasks = set(args.task)
    tasks = {
        task: rows
        for task, rows in loaded_tasks.items()
        if not selected_tasks or task in selected_tasks
    }
    if not tasks:
        suffix = f" matching {args.task}" if args.task else ""
        raise SystemExit(
            f"No usable trajectories found under {source_root}{suffix}"
        )

    trial_count = sum(len(rows) for rows in tasks.values())
    count_distribution = defaultdict(int)
    for rows in tasks.values():
        count_distribution[len(rows)] += 1
    all_pass_count = sum(
        1 for rows in tasks.values() if all(row["reward"] == 1 for row in rows)
    )
    all_fail_count = sum(
        1 for rows in tasks.values() if all(row["reward"] == 0 for row in rows)
    )
    swing_count = len(tasks) - all_pass_count - all_fail_count
    counts_text = ", ".join(
        f"{candidate_count}x{task_count}"
        for candidate_count, task_count in sorted(count_distribution.items())
    )
    if args.expected_candidates is not None:
        for task, rows in sorted(tasks.items()):
            if len(rows) != args.expected_candidates:
                print(
                    f"WARNING: {task} has {len(rows)} traces, expected "
                    f"{args.expected_candidates}",
                    file=sys.stderr,
                )

    max_pairs = 0
    max_pair_chars = 0
    for rows in tasks.values():
        n = len(rows)
        if n < 2:
            continue
        longest = sorted((len(row["trace"]) for row in rows), reverse=True)
        max_pair_chars = max(max_pair_chars, longest[0] + longest[1])
        k = min(args.pivots, max(0, n - 1))
        max_pairs += n + k * (n - k) + k * (k - 1) // 2
    max_calls = max_pairs * OFFICIAL_CRITERIA_COUNT * args.n_evaluations
    endpoint = endpoint_identity(args.backend)
    identity_payload = run_identity_payload(
        dataset=dataset,
        tasks=tasks,
        backend=args.backend,
        model=model,
        served_model_identity=served_model_identity,
        endpoint=endpoint,
        scoring_mode=scoring_mode,
        scorer_source_sha256=source_sha256,
        request_config=request_config,
        n_evaluations=args.n_evaluations,
        pivots=args.pivots,
        seed=args.seed,
        trace_chars=args.trace_chars,
    )
    run_id = stable_digest(identity_payload)
    results_path = (
        args.results_jsonl.expanduser().resolve()
        if args.results_jsonl is not None
        else default_results_path(
            cache_dir, dataset, args.backend, model, run_id
        )
    )
    usage_path = Path(f"{results_path}.usage.jsonl")
    completed = load_completed_records(results_path, run_id)
    for task, record in list(completed.items()):
        rows = tasks.get(task, [])
        if (
            len(record["scores"]) != len(rows)
            or sorted(record["ranking"]) != list(range(len(rows)))
            or record.get("task_trace_identity")
            != task_trace_identity(task, rows)
        ):
            print(
                f"WARNING: ignoring stale/incomplete JSONL record for {task}",
                file=sys.stderr,
            )
            del completed[task]

    print(
        f"[llm-verifier] backend={args.backend} model={model} "
        f"served={served_model_identity} "
        f"scoring={scoring_mode} protocol={SCORING_PROTOCOL} "
        f"K={args.n_evaluations} "
        f"pivots={args.pivots} workers={workers} "
        f"criteria={OFFICIAL_CRITERIA_PATH} "
        f"endpoint={json.dumps(endpoint, sort_keys=True)} "
        f"scorer_source={source_sha256[:16]} "
        f"request={json.dumps(request_config, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[llm-verifier] loaded {trial_count} traces in {len(tasks)} tasks; "
        f"candidate-counts={counts_text}; reward-pools="
        f"all-pass:{all_pass_count},swing:{swing_count},all-fail:{all_fail_count}; "
        f"max-pair-chars={max_pair_chars}; at most {max_calls} verifier calls "
        f"(cache hits reduce this)",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[llm-verifier] protocol note: full audit scores unanimous pools and "
        "resets the tournament seed per task; this is not the paper's exact "
        "global-RNG/swing-only run_bo5 protocol",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[llm-verifier] run_id={run_id} results={results_path}; "
        f"{len(completed)}/{len(tasks)} tasks already complete",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        print("[llm-verifier] dry run: no verifier calls made", file=sys.stderr)
        return

    print(
        "task\ttrial\ttb_reward\tllm_verifier_ppt_score\trank\t"
        "backend\tmodel\tscoring_mode\tprotocol",
        flush=True,
    )

    pending = [task for task in tasks if task not in completed]
    client = None
    scorer = None
    if pending:
        client, scorer = configure_backend(args, model, workers)
    scorer_context = patched_pair_scorer(scorer) if scorer else nullcontext()

    with scorer_context:
        for task in sorted(tasks):
            rows = sorted(tasks[task], key=lambda row: row["trial"])
            if task in completed:
                record = completed[task]
                print(
                    f"[llm-verifier] resumed {task} from {results_path}",
                    file=sys.stderr,
                    flush=True,
                )
                print_tsv_rows(
                    task,
                    rows,
                    record["scores"],
                    record["ranking"],
                    args.backend,
                    model,
                    scoring_mode,
                )
                continue

            cache_file = cache_file_for(
                cache_dir,
                dataset,
                task,
                rows,
                args.backend,
                model,
                served_model_identity,
                endpoint,
                source_sha256,
                request_config,
            )
            print(
                f"[llm-verifier] scoring {task} ({len(rows)} traces); "
                f"cache={cache_file}",
                file=sys.stderr,
                flush=True,
            )
            client._gv_usage_path = usage_path
            client._gv_usage_context = {
                "run_id": run_id,
                "task": task,
                "backend": args.backend,
                "model": model,
                "served_model_identity": served_model_identity,
                "service_tier": (
                    request_config.get("service_tier")
                    if request_config is not None else None
                ),
            }
            result = llm_verifier.select(
                problem=rows[0]["problem"],
                candidates=[row["trace"] for row in rows],
                criteria=str(OFFICIAL_CRITERIA_PATH),
                n_evaluations=args.n_evaluations,
                pivots=min(args.pivots, len(rows) - 1),
                seed=args.seed,
                max_workers=workers,
                model=model,
                cache=str(cache_file),
                progress=False,
                # Never fabricate 0.5 ties when a provider or parser fails.
                on_error="raise",
                client=client,
            )

            scores = [float(value) for value in result.scores]
            ranking = [int(value) for value in result.ranking]
            record = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "run_identity": identity_payload,
                "task": task,
                "task_trace_identity": task_trace_identity(task, rows),
                "served_model_identity": served_model_identity,
                "scorer_source_sha256": source_sha256,
                "request_config": request_config,
                "trials": [
                    {
                        "trial": row["trial"],
                        "reward": row["reward"],
                        "source": row.get("source"),
                        "source_sha256": row.get("source_sha256"),
                    }
                    for row in rows
                ],
                "scores": scores,
                "ranking": ranking,
                "selected_index": int(result.index),
                "n_comparisons": int(result.n_comparisons),
                "usage": aggregate_call_usage(usage_path, run_id, task),
            }
            append_task_record(results_path, record)
            print_tsv_rows(
                task,
                rows,
                scores,
                ranking,
                args.backend,
                model,
                scoring_mode,
            )

    if args.backend == "codex-text" and client is not None:
        print(
            f"[llm-verifier] Codex calls={client.calls} "
            f"empty/failed={client.failures}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
