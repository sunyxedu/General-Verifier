"""End-to-end arms: ONE prompt to a real solver, the whole system measured.

No isolated claims, no pre-formed answers. Per problem:

--arm gv      a stock headless Claude Code session with the full General
              Verifier mounted (MCP sidecar + Pre/PostToolUse hooks + the
              solver protocol as CLAUDE.md). The solver works the problem,
              commits claims as it reasons, and requests the certificate on
              its final claim. Measured: the answer it settles on, the
              certificate verdict, and whether verification interfered
              (refuted claims it had to recover from).

--arm plain   the same solver model with the same problem and no harness,
              sampled --n times per problem. Yields pass@1 / the candidate
              pool for paper-style best-of-N selection (select_e2e.py).

Ground truth comes from problems_aime2026 (every answer brute-force
confirmed in labels_aime2026.py); the set post-dates the model cutoff.

Usage:
  uv run python experiments/aime2024/run_e2e.py --arm gv
  uv run python experiments/aime2024/run_e2e.py --arm plain --n 3
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from problems_aime2026 import PROBLEMS             # noqa: E402

TOOLCHAIN = os.path.expanduser("~/.cache/gv-toolchain")
TIMEOUT = 3600

MCP_JSON = {
    "mcpServers": {
        "gv": {
            "type": "stdio",
            "command": "uv",
            "args": ["run", "--project", ROOT, "python", "-m",
                     "verifier.sidecar", "--state-dir", ".gv",
                     "--toolchain-dir", TOOLCHAIN],
        }
    }
}

# the repo path contains a space — the hook command is shell-parsed, so the
# script path MUST be quoted or every tool call in the session gets blocked
# by a failing PreToolUse hook
SETTINGS_JSON = {
    "hooks": {
        "PreToolUse": [{"matcher": "*", "hooks": [
            {"type": "command",
             "command": f'python3 "{ROOT}/hooks/pre_tool_use.py"'}]}],
        "PostToolUse": [{"matcher": "*", "hooks": [
            {"type": "command",
             "command": f'python3 "{ROOT}/hooks/post_tool_use.py"'}]}],
    }
}

GV_PROMPT = (
    "{statement}\n\n"
    "Solve this problem; the answer is an integer. Follow the claim "
    "protocol in CLAUDE.md: commit your key intermediate results as claims "
    "while you work, commit a final claim whose text ends with 'The answer "
    "is <N>.', and request gv.certificate on it before you finish. End "
    "your reply with a line of exactly this form: FINAL ANSWER: <N>"
)

PLAIN_PROMPT = (
    "{statement}\n\n"
    "Solve this problem; the answer is an integer. End your reply with a "
    "line of exactly this form: FINAL ANSWER: <N>"
)


def recorded(results_path, key="pid"):
    done = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r[key], r.get("sample", 0)))
    return done


def final_claim(state_dir):
    """(text, status) of the final claim, plus refuted-claim count."""
    db_path = os.path.join(state_dir, "claims.sqlite")
    if not os.path.exists(db_path):
        return None, None, 0
    db = sqlite3.connect(db_path)
    row = db.execute(
        "SELECT text, status FROM claims WHERE kind='final' LIMIT 1"
    ).fetchone()
    (n_refuted,) = db.execute(
        "SELECT COUNT(*) FROM claims WHERE status='refuted' "
        "AND kind != 'obligation'").fetchone()
    return (row[0], row[1], n_refuted) if row else (None, None, n_refuted)


def parse_answer(text):
    if not text:
        return None
    m = re.findall(r"answer is\s*(-?\d+)", text, re.IGNORECASE)
    if not m:
        m = re.findall(r"FINAL ANSWER:\s*(-?\d+)", text)
    return int(m[-1]) if m else None


def run_solver(cwd, prompt, mcp=False):
    t0 = time.time()
    cmd = ["claude", "-p", prompt, "--model", "sonnet",
           "--dangerously-skip-permissions"]
    if mcp:
        # headless -p does not auto-trust a project .mcp.json; load it
        # explicitly or the whole harness silently stays unmounted
        cmd += ["--mcp-config", ".mcp.json", "--strict-mcp-config"]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=TIMEOUT)
        out = proc.stdout
    except subprocess.TimeoutExpired:
        out = "<timeout>"
    return out, round(time.time() - t0, 1)


def arm_gv(results_dir, results):
    done = recorded(results)
    todo = [p for p in PROBLEMS if (p["id"], 0) not in done]
    print(f"[e2e:gv] {len(todo)} problems ({len(done)} recorded)", flush=True)
    for i, p in enumerate(todo, 1):
        # workdir OUTSIDE the repo: inside it, Claude Code adopts the repo
        # as the project and the whole verifier tree leaks into the
        # solver's context
        work = os.path.join("/tmp/gv-e2e", p["id"], "work")
        os.makedirs(os.path.join(work, ".claude"), exist_ok=True)
        with open(os.path.join(work, ".mcp.json"), "w") as f:
            json.dump(MCP_JSON, f)
        with open(os.path.join(work, ".claude", "settings.json"), "w") as f:
            json.dump(SETTINGS_JSON, f)
        with open(os.path.join(ROOT, "protocol", "solver-protocol.md")) as f:
            protocol = f.read()
        with open(os.path.join(work, "CLAUDE.md"), "w") as f:
            f.write(protocol)
        out, seconds = run_solver(
            work, GV_PROMPT.format(statement=p["statement"]), mcp=True)
        state = os.path.join(work, ".gv")
        text, status, n_refuted = final_claim(state)
        answer = parse_answer(text) or parse_answer(out)
        events = 0
        ev = os.path.join(state, "events.jsonl")
        if os.path.exists(ev):
            events = sum(1 for line in open(ev) if line.strip())
        rec = {"pid": p["id"], "sample": 0, "answer": answer,
               "truth": p["true"], "solved": answer == p["true"],
               "verdict": status, "refuted_claims": n_refuted,
               "seconds": seconds, "events": events, "workdir": work,
               "final_text": (text or "")[:200],
               "output_tail": out[-300:]}
        with open(results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[e2e:gv {i}/{len(todo)}] {p['id']} answer={answer} "
              f"truth={p['true']} verdict={status} "
              f"refuted={n_refuted} ({seconds}s, {events} ev)", flush=True)


def arm_plain(results_dir, results, n):
    done = recorded(results)
    todo = [(p, s) for p in PROBLEMS for s in range(n)
            if (p["id"], s) not in done]
    print(f"[e2e:plain] {len(todo)} samples ({len(done)} recorded)",
          flush=True)
    for i, (p, s) in enumerate(todo, 1):
        work = os.path.join(results_dir, f"{p['id']}_s{s}")
        os.makedirs(work, exist_ok=True)
        out, seconds = run_solver(
            work, PLAIN_PROMPT.format(statement=p["statement"]))
        answer = parse_answer(out)
        rec = {"pid": p["id"], "sample": s, "answer": answer,
               "truth": p["true"], "solved": answer == p["true"],
               "seconds": seconds, "output_tail": out[-300:]}
        with open(results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[e2e:plain {i}/{len(todo)}] {p['id']}#{s} answer={answer} "
              f"truth={p['true']} ({seconds}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gv", "plain"])
    ap.add_argument("--n", type=int, default=3, help="plain samples/problem")
    args = ap.parse_args()
    results_dir = os.path.join(HERE, "results", f"e2e-{args.arm}")
    os.makedirs(results_dir, exist_ok=True)
    results = os.path.join(results_dir, "results.jsonl")
    if args.arm == "gv":
        arm_gv(results_dir, results)
    else:
        arm_plain(results_dir, results, args.n)
    print(f"[e2e:{args.arm}] done", flush=True)


if __name__ == "__main__":
    main()
