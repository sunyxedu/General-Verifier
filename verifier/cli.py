"""Consumer-side certificate gate.

The solver is never forced to verify; the CONSUMER refuses to merge without a
certificate. This CLI is that refusal, shaped for CI:

  uv run python -m verifier.cli certify --events trace.jsonl --state-dir .gv-ci \
      [--final F] [--model m]
  -> exit 0 iff the verdict is `verified`; prints the certificate summary.

  uv run python -m verifier.cli replay --events trace.jsonl
  -> LLM-free: rebuilds the claim graph and prints label -> status.

  uv run python -m verifier.cli score --state-dir .gv [--model m]
  -> offline ORM path: the state dir holds a vanilla trace (hook_feed.jsonl
     + task.md + baseline/ + final.txt); extract the claim graph, certify,
     print the feature dict. Exit 0 iff the verdict is `verified`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Optional

from .stream.builder import Builder
from .engine.certificate import request_certificate
from .core.events import EventLog
from .mcp_server import Service
from .core.store import ClaimStore, TRIGGER_ON_FIRST_DEPENDENT


def certify(events_path: str, state_dir: str, final_label: Optional[str] = None,
            model: Optional[str] = None, kernel_factory=None) -> int:
    os.makedirs(state_dir, exist_ok=True)
    target = os.path.join(state_dir, "events.jsonl")
    if os.path.abspath(events_path) != os.path.abspath(target):
        shutil.copyfile(events_path, target)   # the gate works on its own copy

    # the gate replays a finished trace: there is no live solver to warn early,
    # and the verdict depends only on the final's slice — so verify lazily and
    # let the certificate pull in the residue, instead of checking every claim
    # in the trace including the ones no final rests on.
    sc = Service(state_dir, kernel_factory=kernel_factory, model=model,
                 trigger=TRIGGER_ON_FIRST_DEPENDENT)
    if final_label is not None:
        final_id = sc.builder.resolve(final_label)
    elif sc.builder.finals:
        final_id = sc.builder.finals[-1]
    else:
        print(json.dumps({"error": "no final claim in trace"}))
        return 2

    cert = request_certificate(sc.store, sc.kernel, sc.log, sc.artifacts,
                               final_id)
    print(json.dumps(
        {
            "verdict": cert["verdict"],
            "certificate_id": cert["certificate_id"],
            "slice": [
                {"text": e["text"], "status": e["status"]}
                for e in cert["slice"]
            ],
        },
        ensure_ascii=False, indent=1,
    ))
    return 0 if cert["verdict"] == "verified" else 1


def replay(events_path: str) -> int:
    log = EventLog(events_path)
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    out = {
        label: builder.store.status(cid)
        for label, cid in builder.labels.items()
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def score(state_dir: str, model: Optional[str] = None,
          max_rounds: int = 3, critic_rounds: Optional[int] = None) -> int:
    from . import offline

    features = offline.score(state_dir, model=model, max_rounds=max_rounds,
                             critic_rounds=critic_rounds)
    print(json.dumps(features, ensure_ascii=False, indent=1))
    return 0 if features.get("verdict") == "verified" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verifier.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("certify")
    c.add_argument("--events", required=True)
    c.add_argument("--state-dir", required=True)
    c.add_argument("--final", default=None)
    c.add_argument("--model", default=None)

    r = sub.add_parser("replay")
    r.add_argument("--events", required=True)

    s = sub.add_parser("score")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--max-rounds", type=int, default=3)
    s.add_argument("--critic-rounds", type=int, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "certify":
        return certify(args.events, args.state_dir, args.final, args.model)
    if args.cmd == "score":
        return score(args.state_dir, args.model, args.max_rounds,
                     args.critic_rounds)
    return replay(args.events)


if __name__ == "__main__":
    sys.exit(main())
