#!/usr/bin/env python3
"""Re-run the observer accounting over a retained worker output, read-only.

The observer ledger is the evidence; the analyzer's rules are code.  When a
rule is corrected (for example, recognising a verifier-owned scratch cwd)
the retained ledgers can be re-accounted without re-running any model
call.  This prints the analyzer's result and never modifies the output
directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium",
           "service_tier": "standard"}
# The evaluation workers pin TMPDIR to this host temporary root; verifier
# scratch directories therefore live directly under it.
WORKER_TMPDIR = "/private/tmp"


def _analyzer():
    path = Path(__file__).with_name("analyze_codex_observer.py")
    spec = importlib.util.spec_from_file_location("_gv_reaccount_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reaccount(output: Path, tmpdir: str = WORKER_TMPDIR) -> dict:
    start = json.loads((output / "worker-start.json").read_text())
    plan = start["plan"]
    result_path = output / "worker-result.json"
    summary = None
    if result_path.is_file():
        summary = json.loads(result_path.read_text()).get("observer_summary")
    state = output / "state"
    contexts = {str(state): {"arm": plan["arm"], "worker_id": start["worker_id"]}}
    temp = os.path.realpath(tmpdir)
    return _analyzer().analyze_file(
        output / "observer.jsonl", context_by_cwd=contexts, **RUNTIME,
        session_summary=summary,
        authority_scratch_prefix=os.path.join(temp, "gv-authority-"),
        typed_judger_scratch_prefix=os.path.join(temp, "gv-typed-judger-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+")
    parser.add_argument("--tmpdir", default=WORKER_TMPDIR,
                        help="the TMPDIR the worker ran with")
    args = parser.parse_args()
    for output in args.outputs:
        accounting = reaccount(Path(output), args.tmpdir)
        print(json.dumps({
            "output": output,
            "accounting_complete": accounting.get("accounting_complete"),
            "findings": accounting.get("findings"),
            "attempt_count": accounting.get("attempt_count"),
            "tokens": {key: value.get("known_sum")
                       for key, value in accounting.get("tokens", {}).items()},
            "authority_scratch_attempts": accounting.get("authority_scratch_attempts"),
            "typed_judger_scratch_attempts": accounting.get("typed_judger_scratch_attempts"),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
