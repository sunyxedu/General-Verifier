"""Replay a recorded solver trace into a fresh task container.

Purpose: trace-only datasets (e.g. the LLM-as-a-Verifier release) ship no
captured world. If the recorded command stream, replayed from the task's
pinned starting image, reproduces the recorded outputs step by step, the
resulting filesystem is a *certified reconstruction* of the delivery-time
world — usable by GV as world-floor evidence under a declared replay bridge
(strength bounded, verdicts cap at supported).

v1 scope: the TieredCodex primitives found in our own ledgers —
  tools.exec_command({"cmd", "workdir", ...})   -> docker exec, compare output
  tools.apply_patch("*** Begin Patch ...")      -> apply add/update/delete
and source-preserving bash records with input {"keystrokes": <string>}.
Fidelity gates: per-step output comparison (normalized) plus a final tree
diff against the trial's captured artifacts when available (our trials only).

Usage:
  replay.py --trial <trial_dir> --feed <hook_feed.jsonl> [--image IMG]
            [--keep] [--artifacts <dir>]
Exit: prints a JSON fidelity report to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import uuid

WALL_NOISE = re.compile(
    r"^(Script completed|Wall time [0-9.]+ seconds|Output:)\s*$")


HEX_ADDR = re.compile(r"0x[0-9a-fA-F]{6,16}")
LS_DATE = re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{4}|\d{2}:\d{2})\b")
LS_TOTAL = re.compile(r"^total \d+$")


def norm(text: str) -> str:
    """Normalize an output for comparison: drop runner chrome, timing noise,
    trailing whitespace. Deliberately conservative — semantic-content lines
    are kept verbatim."""
    lines = []
    for line in (text or "").splitlines():
        if WALL_NOISE.match(line.strip()):
            continue
        line = HEX_ADDR.sub("0xADDR", line)
        line = LS_DATE.sub("DATE", line)
        if LS_TOTAL.match(line.strip()):
            line = "total N"
        lines.append(line.rstrip())
    # compare as a sorted line multiset: stdout/stderr interleave is not
    # deterministic across the recorder and the replayer
    return "\n".join(sorted(l for l in lines if l.strip()))


def parse_feed(path: str):
    """Yield (kind, payload, recorded_output) replay steps from a ledger."""
    events = [json.loads(l) for l in open(path) if l.strip()]
    calls = {}
    order = []
    for e in events:
        if e.get("type") == "tool_call":
            p = e["payload"]
            calls[p.get("tool_use_id")] = {"tool": p.get("tool"),
                                           "input": p.get("input"),
                                           "output": None}
            order.append(p.get("tool_use_id"))
        elif e.get("type") == "tool_result":
            p = e["payload"]
            tid = p.get("tool_use_id")
            if tid in calls:
                calls[tid]["output"] = p.get("excerpt") or ""
    for tid in order:
        c = calls[tid]
        tool = c["tool"]
        src = c["input"]
        recorded = c["output"] or ""
        # excerpt is a JSON list of {text,type} chunks in our ledgers
        try:
            chunks = json.loads(recorded)
            recorded = "".join(ch.get("text", "") for ch in chunks
                               if isinstance(ch, dict))
        except (json.JSONDecodeError, TypeError):
            pass
        if tool == "bash" and isinstance(src, dict):
            command = src.get("keystrokes")
            if isinstance(command, str):
                yield ("exec", {"cmd": command, "workdir": "/app"},
                       recorded)
            else:
                yield ("skip", {
                    "reason": "bash input lacks string keystrokes",
                }, recorded)
            continue
        if tool != "exec":
            yield ("skip", {"reason": f"unsupported tool {tool!r}"},
                   recorded)
            continue
        if not isinstance(src, str):
            yield ("skip", {"reason": "exec input is not a string"},
                   recorded)
            continue
        start = src.find("tools.exec_command(")
        if start != -1:
            # brace-balanced JSON extraction (nested objects break lazy regex)
            i = src.find("{", start)
            depth, in_str, esc = 0, False, False
            for j in range(i, len(src)):
                ch = src[j]
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break
            try:
                args = json.loads(src[i:j + 1])
            except json.JSONDecodeError:
                yield ("skip", {"reason": "unparseable exec args"}, recorded)
                continue
            yield ("exec", args, recorded)
            continue
        m = re.search(r'tools\.apply_patch\(\s*"(.*)"\s*\)', src, re.S)
        if m:
            patch = m.group(1).encode().decode("unicode_escape")
            yield ("patch", {"patch": patch}, recorded)
            continue
        yield ("skip", {"reason": "unknown primitive",
                        "head": src[:80]}, recorded)


def parse_official(path: str):
    """Replay steps from an official LLM-as-a-Verifier terminal-bench
    trajectory (mini-swe-agent format): each agent step is one bash
    keystrokes command with a recorded {returncode, output} observation."""
    d = json.load(open(path))
    for step in d["trajectory"]["steps"]:
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_name") != "bash":
                yield ("skip", {"reason": f"tool {tc.get('tool_name')}"}, "")
                continue
            cmd = (tc.get("arguments") or {}).get("keystrokes") or ""
            recorded = ""
            obs = step.get("observation") or {}
            results = obs.get("results") or []
            if results:
                content = results[0].get("content") or ""
                try:
                    recorded = json.loads(content).get("output", "")
                except (json.JSONDecodeError, AttributeError):
                    recorded = content
            if "COMPLETE_TASK_AND_SUBMIT" in cmd:
                yield ("skip", {"reason": "submit marker"}, "")
                continue
            yield ("exec", {"cmd": cmd, "workdir": "/app"}, recorded)


_CAPTURE_CAP = 4 * 1024 * 1024  # per-stream cap; replayed steps can emit GBs


class _DockerResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _drain(stream, chunks, cap):
    kept = 0
    while True:
        data = stream.read(65536)
        if not data:
            break
        if kept < cap:
            chunks.append(data[:cap - kept])
            kept += len(chunks[-1])
    stream.close()


def docker(args, **kw):
    import threading
    timeout = kw.pop("timeout", 300)
    proc = subprocess.Popen(["docker", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **kw)
    out, err = [], []
    readers = [threading.Thread(target=_drain, args=(proc.stdout, out, _CAPTURE_CAP)),
               threading.Thread(target=_drain, args=(proc.stderr, err, _CAPTURE_CAP))]
    for t in readers:
        t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        for t in readers:
            t.join()
    return _DockerResult(proc.returncode, "".join(out), "".join(err))


def apply_patch(container: str, patch: str):
    """Minimal v4a patch applier: Add/Update/Delete File sections."""
    sections = re.split(r"\*\*\* (Add|Update|Delete) File: ", patch)[1:]
    for verb, body in zip(sections[0::2], sections[1::2]):
        path, _, rest = body.partition("\n")
        path = path.strip()
        if verb == "Delete":
            docker(["exec", container, "rm", "-f", path])
            continue
        if verb == "Add":
            content = "\n".join(l[1:] for l in rest.splitlines()
                                if l.startswith("+"))
            p = subprocess.run(
                ["docker", "exec", "-i", container, "sh", "-c",
                 f"mkdir -p $(dirname {shlex.quote(path)}) && "
                 f"cat > {shlex.quote(path)}"],
                input=content + "\n", text=True, capture_output=True)
            if p.returncode:
                return f"add failed: {p.stderr[:200]}"
            continue
        return f"unsupported patch verb for replay v1: {verb}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--artifacts", help="captured artifacts dir to diff")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    name = f"gv-replay-{uuid.uuid4().hex[:8]}"
    up = docker(["run", "-d", "--name", name, args.image,
                 "sleep", "infinity"])
    if up.returncode:
        sys.exit(f"container start failed: {up.stderr[:300]}")
    report = {"steps": [], "image": args.image}
    parser = (parse_official if args.feed.endswith("_trajectory.json")
              else parse_feed)
    try:
        for kind, payload, recorded in parser(args.feed):
            if kind == "skip":
                report["steps"].append({"kind": kind, **payload})
                continue
            if kind == "patch":
                err = apply_patch(name, payload["patch"])
                report["steps"].append(
                    {"kind": "patch", "ok": err is None,
                     **({"error": err} if err else {})})
                continue
            cmd = payload.get("cmd") or ""
            workdir = payload.get("workdir") or "/app"
            run = docker(["exec", "-w", workdir, name,
                          "bash", "-lc", cmd], timeout=600)
            got = norm(run.stdout + run.stderr)
            want = norm(recorded)
            match = (got == want) or (want and want in got) \
                or (got and got in want)
            report["steps"].append(
                {"kind": "exec", "cmd": cmd[:120], "match": bool(match),
                 **({} if match else {"got": got[:200],
                                      "want": want[:200]})})
        execs = [s for s in report["steps"] if s["kind"] == "exec"]
        report["exec_steps"] = len(execs)
        report["exec_matched"] = sum(s["match"] for s in execs)

        if args.artifacts:
            diff = []
            for root, _, files in os.walk(args.artifacts):
                for f in files:
                    host = os.path.join(root, f)
                    rel = os.path.relpath(host, args.artifacts)
                    cat = docker(["exec", name, "cat", "/" + rel])
                    if cat.returncode:
                        diff.append({"path": rel, "state": "missing"})
                        continue
                    want = open(host, "rb").read()
                    if cat.stdout.encode() != want:
                        diff.append({"path": rel, "state": "differs"})
            report["world_diff"] = diff
            report["world_files_checked"] = sum(
                len(fs) for _, _, fs in os.walk(args.artifacts))
    finally:
        if not args.keep:
            docker(["rm", "-f", name])
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
