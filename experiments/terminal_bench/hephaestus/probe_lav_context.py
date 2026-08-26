#!/usr/bin/env python3
"""Measure the largest exact LLM-as-a-Verifier prompt before a vLLM run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import score_llm_verifier as scorer
from llm_verifier.prompts import load_prompts


def largest_prompt(tasks, criteria, ground_truth_note):
    largest = None
    for task, task_rows in sorted(tasks.items()):
        rows = sorted(task_rows, key=lambda row: len(row["trace"]), reverse=True)
        if len(rows) < 2:
            continue
        row_a, row_b = rows[:2]
        for criterion in criteria:
            prompt = scorer.fine_reward.build_prompt(
                row_a["problem"],
                row_a["trace"],
                row_b["trace"],
                criterion,
                ground_truth_note,
            )
            candidate = {
                "criterion": criterion["id"],
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "task": task,
                "trial_a": row_a["trial"],
                "trial_b": row_b["trial"],
            }
            if largest is None or candidate["prompt_chars"] > largest["prompt_chars"]:
                largest = candidate
    if largest is None:
        raise ValueError("no task contains at least two trajectories")
    return largest


def count_tokens(base_url, model, api_key, prompt):
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages/count_tokens",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"count_tokens returned HTTP {error.code}: {detail}"
        ) from error
    count = int(result.get("input_tokens", 0))
    if count < 1:
        raise RuntimeError(f"invalid token count response: {result!r}")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8371")
    parser.add_argument(
        "--model", default="deepseek-v4-flash-0731-h200-high"
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--trace-chars", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--max-output-tokens", type=int, default=32_768)
    args = parser.parse_args(argv)

    if args.trace_chars < 0:
        parser.error("--trace-chars must be non-negative")
    if args.max_model_len <= args.max_output_tokens:
        parser.error("--max-model-len must exceed --max-output-tokens")
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get("VLLM_API_KEY", "").strip()
    if not api_key:
        parser.error("set VLLM_API_KEY or pass --api-key-file")

    selected = set(args.task)
    tasks = scorer.load_official_tasks(
        args.trajectory_root.resolve(), args.trace_chars, selected
    )
    missing = sorted(selected - set(tasks))
    if missing:
        parser.error(f"unknown tasks: {', '.join(missing)}")
    ground_truth_note, criteria = load_prompts(
        str(scorer.OFFICIAL_CRITERIA_PATH)
    )
    largest = largest_prompt(tasks, criteria, ground_truth_note)
    input_tokens = count_tokens(
        args.base_url, args.model, api_key, largest.pop("prompt")
    )
    input_limit = args.max_model_len - args.max_output_tokens
    result = {
        **largest,
        "input_limit": input_limit,
        "input_tokens": input_tokens,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "fits": input_tokens <= input_limit,
        "headroom_tokens": input_limit - input_tokens,
        "task_count": len(tasks),
        "trace_count": sum(len(rows) for rows in tasks.values()),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["fits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
