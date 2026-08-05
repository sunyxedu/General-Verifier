"""SWE-bench Pro — run a solver arm over the prepared subset.

Per instance the solver gets ONE prompt (issue + requirements + interface)
and a working checkout of the repo at base_commit (extracted from the
official instance image by prep.py --pull). The produced patch is
`git diff` against the base commit; evaluation happens later inside the
official harness (evaluate.py).

Arms:
  --arm plain    stock headless Claude Code           (--n samples/instance)
  --arm codex    stock headless Codex CLI
  --arm gv       Claude Code + full General Verifier mounted (MCP sidecar +
                 hooks + solver protocol). Just one prompt — the system is
                 used exactly as a normal user would use it.

Usage:
  uv run python experiments/swebench_pro/run.py --arm gv
  uv run python experiments/swebench_pro/run.py --arm plain --n 3
  uv run python experiments/swebench_pro/run.py --arm codex
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                       # noqa: E402
from common import solver                               # noqa: E402

CACHE = os.path.expanduser("~/.cache/gv-experiments/swebench_pro")
WORKROOT = "/tmp/gv-swebench"          # outside the repo (see common.solver)
GV_EXTRA_EXCLUDES = [".gv/", ".claude/", ".mcp.json", "CLAUDE.md"]

BASE_PROMPT = """\
You are working in a checkout of the repository {repo} at the commit where \
the issue below was reported. Fix the issue by modifying the repository in \
place. Do not commit; leave your changes in the working tree. The full test \
environment may not be runnable on this machine — reason from the code and \
verify what you can.

<issue>
{problem_statement}
</issue>

<requirements>
{requirements}
</requirements>

<interface>
{interface}
</interface>
"""

GV_SUFFIX = """
Follow the claim protocol in CLAUDE.md while you work: commit your key \
intermediate conclusions (root cause, intended behavior, properties of your \
fix) as claims, commit a final claim stating that the issue described above \
is fixed by the changes in the working tree, and request gv.certificate on \
it before you finish. Report the verdict verbatim.
"""


def load_instances():
    with open(os.path.join(HERE, "instances.json")) as f:
        return json.load(f)


def materialize(inst, work):
    """Copy the cached /app checkout into a fresh workdir."""
    src = os.path.join(CACHE, "app", inst["instance_id"])
    if not os.path.isdir(os.path.join(src, ".git")):
        raise RuntimeError(
            f"no cached checkout for {inst['instance_id']} — run "
            "`uv run experiments/swebench_pro/prep.py --pull` first")
    os.makedirs(os.path.dirname(work), exist_ok=True)
    # APFS clone (-c) is instant; fall back to plain copy elsewhere
    if subprocess.run(["cp", "-c", "-R", src, work],
                      capture_output=True).returncode != 0:
        subprocess.run(["cp", "-a", src, work], check=True)
    subprocess.run(["git", "config", "core.fileMode", "false"], cwd=work,
                   check=True)
    os.makedirs(os.path.join(work, ".git", "info"), exist_ok=True)
    with open(os.path.join(work, ".git", "info", "exclude"), "a") as f:
        f.write("\n".join(GV_EXTRA_EXCLUDES) + "\n")


def capture_patch(work):
    subprocess.run(["git", "add", "-A"], cwd=work, capture_output=True)
    proc = subprocess.run(["git", "diff", "--cached"], cwd=work,
                          capture_output=True, text=True)
    return proc.stdout


def build_prompt(inst, gv):
    p = BASE_PROMPT.format(
        repo=inst["repo"],
        problem_statement=inst["problem_statement"],
        requirements=inst.get("requirements") or "(none)",
        interface=inst.get("interface") or "(none)")
    return p + (GV_SUFFIX if gv else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gv", "plain", "codex"])
    ap.add_argument("--n", type=int, default=1, help="samples per instance "
                    "(use >1 on plain for best-of-N selection)")
    ap.add_argument("--model", default="sonnet", help="claude model for "
                    "plain/gv; ignored for codex unless --model given twice")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    instances = load_instances()
    results_path = os.path.join(HERE, "results", args.arm, "results.jsonl")
    patches_dir = os.path.join(HERE, "results", args.arm, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    done = res.recorded(results_path, "instance_id", "sample")
    todo = [(inst, s) for inst in instances for s in range(args.n)
            if (inst["instance_id"], s) not in done]
    print(f"[swebench:{args.arm}] {len(todo)} runs ({len(done)} recorded)",
          flush=True)

    for i, (inst, s) in enumerate(todo, 1):
        iid = inst["instance_id"]
        work = os.path.join(WORKROOT, args.arm, iid, f"s{s}")
        if not os.path.isdir(work):
            materialize(inst, work)
        prompt = build_prompt(inst, gv=args.arm == "gv")
        if args.arm == "gv":
            solver.mount_gv(work, model=args.model)
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             mcp=True, timeout=args.timeout)
        elif args.arm == "plain":
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             timeout=args.timeout)
        else:
            out, seconds = solver.run_codex(work, prompt,
                                            timeout=args.timeout)
        patch = capture_patch(work)
        patch_file = os.path.join(patches_dir, f"{iid}_s{s}.diff")
        with open(patch_file, "w") as f:
            f.write(patch)
        transcripts_dir = os.path.join(HERE, "results", args.arm,
                                       "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        transcript_file = os.path.join(transcripts_dir, f"{iid}_s{s}.txt")
        with open(transcript_file, "w") as f:
            f.write(out)
        rec = {"instance_id": iid, "sample": s, "arm": args.arm,
               "model": args.model, "patch_file": patch_file,
               "transcript_file": transcript_file,
               "patch_bytes": len(patch), "seconds": seconds,
               "workdir": work, "output_tail": out[-300:]}
        if args.arm == "gv":
            rec.update(solver.gv_outcome(work))
            rec["final_text"] = (rec["final_text"] or "")[:200]
        res.append(results_path, rec)
        extra = (f" verdict={rec.get('verdict')}"
                 f" refuted={rec.get('refuted_claims')}"
                 if args.arm == "gv" else "")
        print(f"[swebench:{args.arm} {i}/{len(todo)}] {iid}#{s} "
              f"patch={len(patch)}B{extra} ({seconds}s)", flush=True)

    print(f"[swebench:{args.arm}] done", flush=True)


if __name__ == "__main__":
    main()
