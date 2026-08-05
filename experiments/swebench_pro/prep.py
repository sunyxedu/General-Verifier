# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets"]
# ///
"""Prepare the SWE-bench Pro (public set) experiment.

1. Downloads the public dataset (ScaleAI/SWE-bench_Pro, split=test, 731
   instances) and selects a deterministic subset -> instances.json
2. Clones the official evaluation harness (scaleapi/SWE-bench_Pro-os)
   into vendor/ (gitignored)
3. --pull: pulls each instance's prebuilt Docker image
   (jefzda/sweap-images:<tag>, linux/amd64) and extracts the /app repo
   checkout into a local cache — that checkout is what solver agents work on

Run:  uv run experiments/swebench_pro/prep.py [--n 12] [--seed 0] [--pull]
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/gv-experiments/swebench_pro")
HARNESS_GIT = "https://github.com/scaleapi/SWE-bench_Pro-os"

# Default repo pool biases toward fast test suites (Go builds, NodeBB) so a
# subset run on emulated amd64 stays tractable; override with --repos.
DEFAULT_REPOS = ["navidrome/navidrome", "flipt-io/flipt", "NodeBB/NodeBB",
                 "internetarchive/openlibrary"]

FIELDS = ["instance_id", "repo", "base_commit", "problem_statement",
          "requirements", "interface", "repo_language", "fail_to_pass",
          "pass_to_pass", "selected_test_files_to_run",
          "before_repo_set_cmd", "dockerhub_tag", "issue_categories"]


def select_subset(n: int, seed: int, repos: list[str]) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    pool = [dict(r) for r in ds if not repos or r["repo"] in repos]
    by_repo: dict[str, list] = {}
    for r in sorted(pool, key=lambda r: r["instance_id"]):
        by_repo.setdefault(r["repo"], []).append(r)
    rng = random.Random(seed)
    for rows in by_repo.values():
        rng.shuffle(rows)
    # round-robin across repos for a stratified subset
    picked, i = [], 0
    while len(picked) < n and any(by_repo.values()):
        for repo in sorted(by_repo):
            if by_repo[repo] and len(picked) < n:
                picked.append(by_repo[repo].pop(0))
        i += 1
    return [{k: r.get(k) for k in FIELDS} for r in picked]


def clone_harness() -> None:
    dest = os.path.join(HERE, "vendor", "SWE-bench_Pro-os")
    if os.path.exists(dest):
        print(f"[prep] harness already at {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", HARNESS_GIT, dest],
                   check=True)


def image_of(inst: dict) -> str:
    return f"jefzda/sweap-images:{inst['dockerhub_tag']}"


def pull_and_extract(inst: dict) -> None:
    """Pull the instance image and extract /app (repo at base_commit)."""
    dest = os.path.join(CACHE, "app", inst["instance_id"])
    if os.path.exists(os.path.join(dest, ".git")):
        print(f"[prep] cached: {inst['instance_id']}")
        return
    img = image_of(inst)
    print(f"[prep] pulling {img} ...", flush=True)
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", img],
                   check=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cid = subprocess.run(
        ["docker", "create", "--platform", "linux/amd64", img],
        check=True, capture_output=True, text=True).stdout.strip()
    try:
        subprocess.run(["docker", "cp", f"{cid}:/app", dest + ".tmp"],
                       check=True)
        os.rename(dest + ".tmp", dest)
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)
    print(f"[prep] extracted -> {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS,
                    help="repo pool; pass nothing after --repos for all 11")
    ap.add_argument("--pull", action="store_true",
                    help="also pull docker images + extract /app checkouts")
    args = ap.parse_args()

    inst_path = os.path.join(HERE, "instances.json")
    if os.path.exists(inst_path):
        with open(inst_path) as f:
            subset = json.load(f)
        print(f"[prep] instances.json exists ({len(subset)} instances) — "
              "delete it to reselect")
    else:
        subset = select_subset(args.n, args.seed, args.repos)
        with open(inst_path, "w") as f:
            json.dump(subset, f, indent=1)
        print(f"[prep] selected {len(subset)}:")
        for r in subset:
            print(f"    {r['instance_id']}  ({r['repo_language']})")

    clone_harness()

    if args.pull:
        for inst in subset:
            try:
                pull_and_extract(inst)
            except subprocess.CalledProcessError as err:
                print(f"[prep] FAILED {inst['instance_id']}: {err}",
                      file=sys.stderr)
    else:
        print("[prep] skipped image pull (use --pull); run.py needs the "
              "extracted checkouts")


if __name__ == "__main__":
    main()
