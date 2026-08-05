"""Append-only JSONL result files with resume support.

Every runner records one JSON object per unit of work and skips units whose
key is already present — re-running a crashed experiment continues where it
left off.
"""
import json
import os


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recorded(path: str, *key_fields: str) -> set[tuple]:
    return {tuple(r.get(k) for k in key_fields) for r in load(path)}


def append(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
