#!/usr/bin/env python3
"""Protocol smoke test for the DeepSeek-v4-flash vLLM service.

The verifier uses both of vLLM's compatibility surfaces: OpenAI chat
completions for the LLM-as-a-Verifier baseline and Anthropic Messages for the
General Verifier seats.  This script exercises the features that those two
paths specifically need before an expensive benchmark run is started.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def request_json(base_url: str, path: str, api_key: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def smoke(base_url: str, model: str, api_key: str):
    models = request_json(base_url, "/v1/models", api_key)
    served = {item.get("id") for item in models.get("data", [])}
    require(model in served, f"served model missing: {model}; got {sorted(served)}")
    print("ok models")

    count = request_json(
        base_url,
        "/v1/messages/count_tokens",
        api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count this short prompt."}],
        },
    )
    require(int(count.get("input_tokens", 0)) > 0, "invalid Anthropic token count")
    print("ok anthropic count_tokens")

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    structured = request_json(
        base_url,
        "/v1/messages",
        api_key,
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Return 6 times 7."}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema}
            },
        },
    )
    text_blocks = [
        block.get("text", "")
        for block in structured.get("content", [])
        if block.get("type") == "text"
    ]
    parsed = json.loads("".join(text_blocks))
    require(parsed == {"answer": 42}, f"bad structured response: {parsed!r}")
    print("ok anthropic structured output")

    tool = request_json(
        base_url,
        "/v1/messages",
        api_key,
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Use the add tool for 19 plus 23."}
            ],
            "tools": [
                {
                    "name": "add",
                    "description": "Add two integers.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "add"},
        },
    )
    tool_uses = [block for block in tool.get("content", []) if block.get("type") == "tool_use"]
    require(tool_uses, f"Anthropic response contained no tool_use: {tool!r}")
    require(tool_uses[0].get("name") == "add", f"wrong tool: {tool_uses[0]!r}")
    values = tool_uses[0].get("input", {})
    require(values.get("a") + values.get("b") == 42, f"bad tool input: {values!r}")
    print("ok anthropic tool_use")

    logprobs = request_json(
        base_url,
        "/v1/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "End with exactly these two lines, choosing one letter A-T "
                        "for each:\n<score_A>A</score_A>\n<score_B>B</score_B>"
                    ),
                }
            ],
            "max_tokens": 2048,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": 20,
            # These are the fields produced when the OpenAI Python client's
            # ``extra_body`` argument is merged into the HTTP request.
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        },
    )
    choice = logprobs.get("choices", [{}])[0]
    positions = ((choice.get("logprobs") or {}).get("content") or [])
    require(positions, f"OpenAI response contained no answer logprobs: {choice!r}")
    require(
        any(position.get("top_logprobs") for position in positions),
        "OpenAI response contained no top_logprobs",
    )
    print("ok openai top_logprobs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8371")
    parser.add_argument(
        "--model", default="deepseek-v4-flash-0731-h200-high"
    )
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get("VLLM_API_KEY", "").strip()
    if not api_key:
        parser.error("set VLLM_API_KEY or pass --api-key-file")
    smoke(args.base_url, args.model, api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
