"""SWE-bench Pro — evaluate an arm's patches with the OFFICIAL harness.

Wraps vendor/SWE-bench_Pro-os/swe_bench_pro_eval.py (cloned by prep.py):
builds the predictions file + subset CSV it expects, then runs it with
--use_local_docker against the prebuilt jefzda/sweap-images instance
images. On Apple Silicon the script auto-selects linux/amd64 emulation.

Arms:
  --arm gv | turbo | codex   evaluates sample 0 of that arm
  --arm plain                evaluates EVERY sample

Usage:
  uv run python experiments/swebench_pro/evaluate.py --arm gv
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402

VENDOR = os.path.join(HERE, "vendor", "SWE-bench_Pro-os")
EVAL_DEPS = "pandas,tqdm,docker,requests"


def write_subset_csv(instances, path):
    fields = list(instances[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for inst in instances:
            w.writerow(inst)


def patches_for(arm, sample):
    rows = res.load(os.path.join(HERE, "results", arm, "results.jsonl"))
    rows = [r for r in rows if r["sample"] == sample]
    out = []
    for r in rows:
        with open(r["patch_file"]) as f:
            patch = f.read()
        if patch.strip():
            out.append({"instance_id": r["instance_id"], "patch": patch,
                        "prefix": f"{arm}_s{sample}" if arm == "plain"
                                  else arm})
    return out


def run_eval(tag, patches, instances, workers):
    out_dir = os.path.join(HERE, "results", "eval", tag)
    os.makedirs(out_dir, exist_ok=True)
    patch_path = os.path.join(out_dir, "patches.json")
    with open(patch_path, "w") as f:
        json.dump(patches, f)
    subset_csv = os.path.join(out_dir, "subset.csv")
    iids = {p["instance_id"] for p in patches}
    write_subset_csv([i for i in instances if i["instance_id"] in iids],
                     subset_csv)
    cmd = ["uv", "run", "--no-project", "--with", EVAL_DEPS, "python",
           os.path.join(VENDOR, "swe_bench_pro_eval.py"),
           f"--raw_sample_path={subset_csv}",
           f"--patch_path={patch_path}",
           f"--output_dir={out_dir}",
           "--scripts_dir=run_scripts",
           "--dockerhub_username=jefzda",
           "--use_local_docker",
           f"--num_workers={workers}"]
    print(f"[swebench:eval:{tag}] {len(patches)} patches -> {out_dir}",
          flush=True)
    proc = subprocess.run(cmd, cwd=VENDOR)
    if proc.returncode != 0:
        print(f"[swebench:eval:{tag}] harness exit {proc.returncode}",
              file=sys.stderr)
    results_file = os.path.join(out_dir, "eval_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            r = json.load(f)
        ok = sum(1 for v in r.values() if v)
        print(f"[swebench:eval:{tag}] resolved {ok}/{len(r)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["gv", "turbo", "plain", "codex"])
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    if not os.path.isdir(VENDOR):
        sys.exit("vendor harness missing — run prep.py first")
    with open(os.path.join(HERE, "instances.json")) as f:
        instances = json.load(f)

    if args.arm == "plain":
        rows = res.load(os.path.join(HERE, "results", "plain",
                                     "results.jsonl"))
        samples = sorted({r["sample"] for r in rows})
        for s in samples:
            run_eval(f"plain_s{s}", patches_for("plain", s), instances,
                     args.num_workers)
    else:
        run_eval(args.arm, patches_for(args.arm, 0), instances,
                 args.num_workers)


if __name__ == "__main__":
    main()
