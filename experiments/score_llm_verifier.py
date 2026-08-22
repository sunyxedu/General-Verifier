"""Post-hoc LLM-as-a-Verifier scoring for an existing Harbor job.

The scorer reads saved ``result.json`` and ``agent/trajectory.json`` files.
It never invokes Harbor and never regenerates a solver trajectory.

Two deliberately explicit verifier paths are supported:

* ``codex-text`` asks a ChatGPT-subscription Codex model for one literal A-T
  rating per candidate and criterion. This is a discrete 20-bin baseline;
  token logprobs are not available through the Codex CLI.
* ``deepseek-logprob`` / ``gemini-logprob`` require real token logprobs and
  compute the probability-weighted A-T expectation from them.

TSV results go to stdout. Diagnostics go to stderr, so stdout can be piped
through ``tee`` without contaminating the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import llm_verifier
import llm_verifier.fine_grained_reward as fine_reward


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = (
    HERE / "terminal_bench" / "results" / "llm_verifier_cache"
)
OFFICIAL_CRITERIA_PATH = (
    HERE / "vendor" / "llm-as-a-verifier" / "criteria" / "terminal_bench.md"
)
OFFICIAL_CRITERIA_COUNT = 3
DEFAULT_TRACE_CHARS = 0
# Total tries for a logprob call whose reasoning overflowed the output
# budget before the failure is re-raised.
BUDGET_OVERFLOW_ATTEMPTS = 3

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
}

# Require the exact two-line score block at the end. A malformed or truncated
# answer is an error, not a synthetic 0.5 tie.
FINAL_SCORE_BLOCK = re.compile(
    r"<score_A>\s*([A-T])\s*</score_A>\s*\n\s*"
    r"<score_B>\s*([A-T])\s*</score_B>\s*\Z",
    re.IGNORECASE,
)


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
                out.append(f"[Command] {arguments['keystrokes']}")
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
        raise RuntimeError(f"no A-T token-logprob mass found at {tag}")
    return sum(score * prob for score, prob in probabilities.items()) / total


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
    # Reasoning length is stochastic at temperature 1.0, so a sample that
    # overflows the output budget (finish_reason='length', no answer
    # logprobs) is resampled a few times before giving up. Every other
    # failure still raises immediately; ties are never fabricated.
    attempts = BUDGET_OVERFLOW_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            _raw, tokens, position_logprobs = fine_reward.call_verifier(
                client, prompt, model, images=imgs
            )
            break
        except RuntimeError as error:
            if "no answer logprobs" not in str(error) or attempt == attempts:
                raise
            print(
                f"[llm-verifier] reasoning overflowed the output budget; "
                f"resampling ({attempt}/{attempts - 1} retries used)",
                file=sys.stderr,
                flush=True,
            )
    return (
        expected_logprob_score(tokens, position_logprobs, "<score_A>"),
        expected_logprob_score(tokens, position_logprobs, "<score_B>"),
    )


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


def cache_file_for(cache_dir, job_dir, task, rows, backend, model):
    # llm-verifier.select internally names every synthetic task simply
    # "task", and its cache key does not include prompt/model/backend. Keep a
    # separate, content-addressed cache file to prevent cross-task score reuse.
    payload = {
        "job": str(job_dir),
        "task": task,
        "problem": rows[0]["problem"],
        "traces": [row["trace"] for row in rows],
        "criteria_file": OFFICIAL_CRITERIA_PATH.read_text(encoding="utf-8"),
        "backend": backend,
        "model": model,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return (
        cache_dir
        / slug(job_dir.name)
        / slug(backend)
        / slug(model)
        / f"{slug(task)}-{digest}.json"
    )


def load_tasks(job_dir, trace_chars, selected_tasks):
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
            }
        )
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc LLM-as-a-Verifier scoring for an existing Harbor job; "
            "does not run Harbor or regenerate trajectories."
        )
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        help="Harbor job directory containing one subdirectory per trial",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_DEFAULTS),
        required=True,
        help=(
            "explicit verifier path: Codex text-only A-T, DeepSeek logprobs, "
            "or Gemini Vertex logprobs"
        ),
    )
    parser.add_argument(
        "--model",
        help="model id (default depends on --backend)",
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=2,
        help="repeated ratings K per criterion (default: 2, upstream Bo3)",
    )
    parser.add_argument(
        "--pivots",
        type=int,
        default=1,
        help="PPT pivot count (default: 1, upstream Bo3)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "parallel verifier calls (default: 1 for Codex, 2000 for the "
            "official DeepSeek Bo3 preset, 6 for Gemini)"
        ),
    )
    parser.add_argument(
        "--codex-effort",
        default="low",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="reasoning effort for --backend codex-text (default: low)",
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
        "--dry-run",
        action="store_true",
        help="validate inputs and print the planned workload without scoring",
    )
    args = parser.parse_args()

    if args.n_evaluations < 1:
        parser.error("--n-evaluations must be at least 1")
    if args.pivots < 0:
        parser.error("--pivots must be non-negative")
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.codex_timeout < 1:
        parser.error("--codex-timeout must be at least 1")
    if args.trace_chars < 0:
        parser.error("--trace-chars must be non-negative")
    return args


def configure_backend(args, model, workers):
    if args.backend == "codex-text":
        # Import only for this backend. It invokes the already-authenticated
        # local Codex CLI directly; no OpenAI API key or proxy is involved.
        from common.codex_endpoint import CodexRunner

        return (
            CodexRunner(
                model=model,
                effort=args.codex_effort,
                timeout=args.codex_timeout,
                concurrency=workers,
            ),
            score_pair_text,
        )
    if args.backend == "deepseek-logprob":
        return fine_reward.create_deepseek_client(model=model), score_pair_logprob
    if args.backend == "gemini-logprob":
        return fine_reward.create_gemini_client(), score_pair_logprob
    raise AssertionError(f"unhandled backend: {args.backend}")


def main():
    args = parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"Job directory does not exist: {job_dir}")
    if not OFFICIAL_CRITERIA_PATH.is_file():
        raise SystemExit(
            f"Official Terminal-Bench criteria file is missing: "
            f"{OFFICIAL_CRITERIA_PATH}"
        )

    defaults = BACKEND_DEFAULTS[args.backend]
    model = args.model or defaults["model"]
    workers = args.max_workers or defaults["workers"]
    scoring_mode = defaults["scoring_mode"]
    cache_dir = args.cache_dir.expanduser().resolve()

    tasks = load_tasks(job_dir, args.trace_chars, set(args.task))
    if not tasks:
        suffix = f" matching {args.task}" if args.task else ""
        raise SystemExit(
            f"No Harbor trials with trajectories found under {job_dir}{suffix}"
        )

    trial_count = sum(len(rows) for rows in tasks.values())
    max_pairs = 0
    for rows in tasks.values():
        n = len(rows)
        k = min(args.pivots, max(0, n - 1))
        max_pairs += n + k * (n - k) + k * (k - 1) // 2
    max_calls = max_pairs * OFFICIAL_CRITERIA_COUNT * args.n_evaluations
    print(
        f"[llm-verifier] backend={args.backend} model={model} "
        f"scoring={scoring_mode} K={args.n_evaluations} "
        f"pivots={args.pivots} workers={workers} "
        f"criteria={OFFICIAL_CRITERIA_PATH}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[llm-verifier] loaded {trial_count} traces in {len(tasks)} tasks; "
        f"at most {max_calls} verifier calls (cache hits reduce this)",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        print("[llm-verifier] dry run: no verifier calls made", file=sys.stderr)
        return

    client, scorer = configure_backend(args, model, workers)

    print(
        "task\ttrial\ttb_reward\tllm_verifier_ppt_score\trank\t"
        "backend\tmodel\tscoring_mode",
        flush=True,
    )

    with patched_pair_scorer(scorer):
        for task in sorted(tasks):
            rows = sorted(tasks[task], key=lambda row: row["trial"])
            if len(rows) != 3:
                print(
                    f"WARNING: {task} has {len(rows)} traces, expected 3",
                    file=sys.stderr,
                )

            cache_file = cache_file_for(
                cache_dir, job_dir, task, rows, args.backend, model
            )
            print(
                f"[llm-verifier] scoring {task} ({len(rows)} traces); "
                f"cache={cache_file}",
                file=sys.stderr,
                flush=True,
            )
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

            rank = {
                candidate_index: position + 1
                for position, candidate_index in enumerate(result.ranking)
            }
            for index, row in enumerate(rows):
                print(
                    f"{task}\t"
                    f"{row['trial']}\t"
                    f"{row['reward']:.0f}\t"
                    f"{result.scores[index]:.6f}\t"
                    f"{rank[index]}\t"
                    f"{args.backend}\t"
                    f"{model}\t"
                    f"{scoring_mode}",
                    flush=True,
                )

    if args.backend == "codex-text":
        print(
            f"[llm-verifier] Codex calls={client.calls} "
            f"empty/failed={client.failures}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
