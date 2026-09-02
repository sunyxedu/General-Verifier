"""Terminal-Bench — offline ORM scoring of a recorder arm's trials.

Two trace sources, one pipeline:
  - `rec` (claude-code): the container's recording hooks wrote the ledger
    directly; harbor pulled /gv-state as a trial artifact.
  - `rec-codex`: codex has no hooks, but harbor copies its own rollout log
    to trial_dir/agent/sessions/. verifier.stream.codex rewrites that into
    the same feed format, so both arms score identically.

ISOLATION: scoring runs in a sanitized state dir OUTSIDE the repository
($GV_SCORING_ROOT, default ~/.gv-scoring/terminal_bench, then
<job>/<config>/<rep>/<trial>/gv), holding only the verifier's legitimate
inputs — task.md, the raw ledger feed, the final output, the pre-task
baseline, and the delivered world (APFS clonefile when available).
Harbor's gold (result.json, verifier/reward.txt, the official test
output) stays outside the boundary, no path handed to any seat points
back at it, and the seats' cwd has nothing above it but the user's home
— out of reach of the emergent "look around the repo for expected
output" failure mode. The seats are still unsandboxed processes, so
this is isolation by construction of the inputs, not a jail; the trust
anchor for verified certificates is the auditable artifact chain (the
recomputation script and its rerunnable output), and a container is the
optional hardening beyond that. The dir is per CONFIG: two models never
share extraction or settled state.

Each trial is scored independently: its own extraction, its own claim
graph, its own certificate. Scores append to results/rec_scores.jsonl
(resume by (job, trial, config)). All pending trials share ONE worker
pool across tasks (serializing tasks would idle workers on every task's
straggler); a task's trials are ranked by rank_key (best-of-N selection
is per-task) and emitted as TSV rows on stdout the moment its last
pending trial lands — diagnostics go to stderr, so stdout pipes through
`tee` clean. A rerun re-emits cached rows too, so the TSV is always the
complete table.

Usage:
  uv run python experiments/terminal_bench/score.py [--job rec] [--model m]
      [--seats codex] [--effort medium] [--task t ...] [--trial name ...]
"""
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EXPERIMENTS)
for _path in (REPO_ROOT, EXPERIMENTS, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import harborjobs                                        # noqa: E402
from common import harbor_artifacts                      # noqa: E402
from common import results as res                        # noqa: E402
from common import scoring                                # noqa: E402
from verifier import offline                             # noqa: E402
from verifier.stream import codex as codex_stream        # noqa: E402


SANITIZE_MARKER = ".sanitized"
# the verifier's legitimate inputs — everything else in a trial dir
# (result.json, verifier/reward.txt, official test output) is gold and
# stays outside the scoring boundary
SANITIZE_FILES = ("hook_feed.jsonl", "task.md", "final.txt")


def find_state(trial_dir):
    """The pulled /gv-state artifact inside a trial dir (source only —
    scoring itself runs in the sanitized copy)."""
    for root, _dirs, files in os.walk(trial_dir):
        if os.path.basename(root) == "gv-state" or "hook_feed.jsonl" in files:
            return root
    return None


def _cow_copytree(src, dst):
    """Copy a tree with APFS clonefile when available (instant, shares
    blocks); plain copy elsewhere. Copied, never symlinked: find/os.walk
    do not follow directory symlinks, so a symlinked mount would hide the
    tree from search-based checks."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", "-R", src, dst], check=True)
    else:
        shutil.copytree(src, dst, symlinks=True)


def _copy_captured_world(trial_dir, state):
    """Copy every successfully collected Harbor artifact root into world/.

    Harbor mirrors absolute artifact sources beneath ``artifacts/`` and
    records the mapping and collection status in ``manifest.json``.  Reading
    that manifest keeps the offline verifier honest about exactly which roots
    were captured; a failed or absent artifact never becomes negative world
    evidence.  Older /app-only jobs without a manifest retain their legacy
    fallback.
    """
    captured = []
    for capture in harbor_artifacts.successful_captures(trial_dir):
        source = capture.source
        host = os.fspath(capture.host_path)
        target = os.path.join(state, "world", source.lstrip("/"))
        if os.path.islink(host):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.symlink(os.readlink(host), target)
        elif os.path.isdir(host):
            _cow_copytree(host, target)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(host, target)
        captured.append(source)
    return sorted(set(captured))


def sanitized_state(trial, root, job, config):
    """Build (or reuse) the isolated per-config scoring state dir for a
    trial: <root>/<job>/<config>/<trial>/gv. Only SANITIZE_FILES, the
    baseline/ snapshot, and the delivered world (trial artifacts/app
    mounted at world/app, mirroring the run's filesystem root) are copied
    in. rec trials source from the pulled gv-state; codex trials convert
    their rollout log into the same feed format. Returns None when the
    trial has neither source."""
    state = os.path.join(root, job, config, trial["trial"], "gv")
    marker = os.path.join(state, SANITIZE_MARKER)
    if os.path.exists(marker):
        # Never adopt an old/interrupted sanitized tree as the immutable
        # source for a new verification.  The adapter seal must already have
        # been published before the completion marker became visible.
        offline.verify_immutable_input_anchor(state)
        return state
    if os.path.isdir(state):
        shutil.rmtree(state)   # interrupted build: start over

    source = find_state(trial["trial_dir"])
    if source is not None:
        os.makedirs(state, exist_ok=True)
        for name in SANITIZE_FILES:
            src = os.path.join(source, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(state, name))
        baseline = os.path.join(source, "baseline")
        if os.path.isdir(baseline):
            _cow_copytree(baseline, os.path.join(state, "baseline"))
    else:
        # codex trial: rollout -> feed; task.md from the trial config when
        # harbor records it there, else from the rollout's first user
        # message. No baseline — a vanilla codex container carries no GV
        # payload to snapshot with.
        sessions = os.path.join(trial["trial_dir"], "agent", "sessions")
        rollout = codex_stream.newest_rollout(sessions) \
            if os.path.isdir(sessions) else None
        if rollout is None:
            return None
        codex_stream.convert(rollout, state)
        instruction = harborjobs.instruction_of(trial["trial_dir"]) \
            or codex_stream.task_instruction(rollout)
        if instruction:
            with open(os.path.join(state, "task.md"), "w") as f:
                f.write(instruction)

    captured = _copy_captured_world(trial["trial_dir"], state)
    # The capture manifest keeps absence honest: a path outside these roots is
    # uncaptured (its absence proves nothing), never a counterexample.
    with open(os.path.join(state, "world_manifest.json"), "w") as f:
        json.dump({"captured_roots": captured}, f)

    # Some recorder backends omit final.txt even though Harbor retained the
    # final assistant message in agent.log.  Finalize that legitimate input
    # before publishing the immutable adapter anchor; mutating it afterwards
    # would correctly invalidate the state.
    final_path = os.path.join(state, "final.txt")
    final_missing = not os.path.exists(final_path)
    if not final_missing:
        with open(final_path, encoding="utf-8") as f:
            final_missing = not f.read().strip()
    if final_missing:
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(harborjobs.log_tail(trial["agent_log"]))

    offline.write_immutable_input_anchor(state)
    with open(marker, "w") as f:
        f.write("")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="rec",
                    help="job name under results/jobs (default: rec)")
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="extraction retry rounds (default 3)")
    ap.add_argument("--critic-rounds", type=int, default=None,
                    help="decomposition-game rounds before a claim fails "
                    "closed (default: the kernel's MAX_CRITIC_ROUNDS)")
    ap.add_argument("--workers", type=int, default=4,
                    help="traces scored concurrently (they share nothing)")
    ap.add_argument("--task", default=None, nargs="+",
                    help="score only these tasks (verification is the "
                    "expensive stage; spend it where selection differs)")
    ap.add_argument("--trial", default=None, nargs="+",
                    help="score only these trials (exact trial names; "
                    "combines with --task)")
    ap.add_argument("--rep", default="r1",
                    help="repetition label (default r1). Same config, "
                    "fresh sampling: each rep gets its own state dirs and "
                    "its own resume slot, so verdict variance is real "
                    "variance, not cache reuse")
    ap.add_argument("--seats", default=None, choices=["codex"],
                    help="codex: run every verifier seat through the Codex "
                    "CLI on the ChatGPT subscription (default model "
                    "gpt-5.6-luna) instead of the SDK/provider path")
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="codex seats' reasoning effort (default: medium)")
    ap.add_argument(
        "--service-tier", default="standard", choices=["standard", "fast"],
        help=("Codex service tier (default: standard; fast explicitly uses "
              "Codex Fast mode and has different credit accounting)"),
    )
    ap.add_argument(
        "--parallel-obligations", type=int, default=None,
        help=("discharges run concurrently within one claim (default: "
              "GV_PARALLEL_OBLIGATIONS or 4); an execution strategy, not "
              "part of the config identity"))
    ap.add_argument(
        "--parallel-claims", type=int, default=None,
        help=("claims of a certificate slice verified concurrently "
              "(default: GV_PARALLEL_CLAIMS or 4)"))
    ap.add_argument(
        "--no-shared-toolchain", action="store_true",
        help=("keep every trial's toolchain and install caches state-local "
              "(default: candidates of one task share "
              "<scoring-root>/<job>/<config>/_shared/<task>, so heavy "
              "instruments install once per task)"))
    ap.add_argument(
        "--scores-path", default=None,
        help=("JSONL result ledger (default: results/rec_scores.jsonl). "
              "Use an isolated path for concurrently running A/B shards."),
    )
    args = ap.parse_args()
    if args.service_tier == "fast" and args.seats != "codex":
        ap.error("--service-tier fast requires --seats codex")

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    scores_path = os.path.abspath(os.path.expanduser(
        args.scores_path or os.path.join(HERE, "results", "rec_scores.jsonl")))
    usage_path = f"{scores_path}.usage.jsonl"
    scoring_root = os.environ.get("GV_SCORING_ROOT") or os.path.expanduser(
        os.path.join("~", ".gv-scoring", "terminal_bench"))
    # resume is keyed on the full CONFIG identity — verifier code+prompts
    # fingerprint plus model/seats/effort/max_rounds. A record produced by
    # a different config is a different verifier's judgment and must not
    # shadow a rescore (offline.score also wipes stale per-trace derived
    # state itself); the plaintext fields ride along for analysis
    cfg = offline.config_fields(model=args.model, seats=args.seats,
                                effort=args.effort,
                                max_rounds=args.max_rounds,
                                critic_rounds=args.critic_rounds,
                                service_tier=args.service_tier)
    done = res.recorded(scores_path, "job", "trial", "config", "rep")
    # cached feature records, so already-scored trials still rank in the TSV
    scored = {r["trial"]: r for r in res.load(scores_path)
              if r.get("job") == args.job
              and r.get("config") == cfg["config"]
              and r.get("rep") == args.rep}

    by_task = {}
    for t in harborjobs.trials(job_dir):
        if args.task and t["task"] not in args.task:
            continue
        if args.trial and t["trial"] not in args.trial:
            continue
        by_task.setdefault(t["task"], []).append(t)
    total = sum(len(ts) for ts in by_task.values())
    todo = sum(1 for ts in by_task.values() for t in ts
               if (args.job, t["trial"], cfg["config"], args.rep) not in done)
    print(f"[tb:score] backend={cfg['seats']} model={cfg['model']} "
          f"effort={cfg['effort']} service_tier={cfg['service_tier']} "
          f"scoring=claim-graph-certificate max_rounds={args.max_rounds} "
          f"critic_rounds={cfg['critic_rounds']} "
          f"workers={args.workers} job={args.job} "
          f"config={cfg['config']} rep={args.rep}",
          file=sys.stderr, flush=True)
    print(f"[tb:score] loaded {total} trials in {len(by_task)} tasks; "
          f"{todo} to score ({len(scored)} cached in "
          f"{os.path.relpath(scores_path, HERE)})",
          file=sys.stderr, flush=True)

    # TSV to stdout (tee-able), diagnostics to stderr. Trials from ALL
    # tasks share one global pool — traces are independent, and the
    # per-task grouping exists only for ranking/emission, so serializing
    # tasks would idle workers on every task's straggler.
    # piped stdout gets TSV rows live, task by task; on a bare terminal
    # the rich board owns the screen, so the table prints once at the end
    # (harbor-style) — any mid-run stdout write would displace the live
    # region's in-place redraw into stacked stale frames
    live_tsv = not sys.stdout.isatty()
    header = ("task\ttrial\ttb_reward\tgv_verdict\tgv_score\t"
              "gv_contrary_signals\trank")
    if live_tsv:
        print(header, flush=True)
    finished = []

    def emit(task):
        finished.append(task)
        if live_tsv:
            render(task)

    def render(task):
        trials = sorted(by_task[task], key=lambda t: t["trial"])
        rows = [(t, scored[t["trial"]]) for t in trials
                if t["trial"] in scored]
        ranked = sorted(rows, key=lambda tr: (
            tuple(-c for c in offline.rank_key(tr[1].get("features"))),
            tr[0]["trial"]))
        rank = {t["trial"]: i + 1 for i, (t, _r) in enumerate(ranked)}
        for t, r in rows:
            features = r.get("features") or {}
            key = offline.rank_key(features)
            reward = "?" if t["reward"] is None else f"{t['reward']:.0f}"
            print(f"{task}\t{t['trial']}\t{reward}\t"
                  f"{features.get('verdict') or 'unscoreable'}\t"
                  f"{scoring.scalar(features):.6f}\t"
                  f"{-key[1]}\t{rank[t['trial']]}", flush=True)

    all_jobs, remaining, emitted = [], {}, set()
    for task in sorted(by_task):
        trials = sorted(by_task[task], key=lambda t: t["trial"])
        jobs = []
        for t in trials:
            if (args.job, t["trial"], cfg["config"], args.rep) in done:
                continue
            state = sanitized_state(
                t, scoring_root, args.job,
                os.path.join(cfg["config"], args.rep))
            if state is None:
                print(f"[tb:score] {t['trial']} no ledger (neither gv-state "
                      "nor a codex rollout) — skipped",
                      file=sys.stderr, flush=True)
                continue
            extra = {"parallel_obligations": args.parallel_obligations,
                     "parallel_claims": args.parallel_claims}
            if not args.no_shared_toolchain:
                extra["shared_dir"] = os.path.join(
                    scoring_root, args.job, cfg["config"], "_shared", task)
            jobs.append(({"job": args.job, "task": task, "trial": t["trial"],
                          "reward": t["reward"], "rep": args.rep,
                          **cfg}, state, extra))
        if jobs:
            all_jobs.extend(jobs)
            remaining[task] = len(jobs)
        else:
            emitted.add(task)
            emit(task)   # fully cached: rows re-emit immediately

    def on_record(rec):
        # runs under score_many's write lock, so emission is serialized
        scored[rec["trial"]] = rec
        remaining[rec["task"]] -= 1
        if remaining[rec["task"]] == 0 and rec["task"] not in emitted:
            emitted.add(rec["task"])
            emit(rec["task"])

    if all_jobs:
        usage_capture = contextlib.nullcontext()
        if args.seats == "codex":
            from verifier.llm.codex_backend import codex_usage_ledger

            usage_contexts = {
                os.path.realpath(state): {
                    key: record[key]
                    for key in ("job", "task", "trial", "config", "rep")
                }
                for record, state in all_jobs
            }
            usage_capture = codex_usage_ledger(
                usage_path, context_by_cwd=usage_contexts)
        print(f"[tb:score] scoring {len(all_jobs)} traces across "
              f"{len(remaining)} tasks (workers={args.workers})"
              + (f" usage={os.path.relpath(usage_path, HERE)}"
                 if args.seats == "codex" else ""),
              file=sys.stderr, flush=True)
        with usage_capture:
            scoring.score_many(
                all_jobs, scores_path, workers=args.workers,
                model=args.model, max_rounds=args.max_rounds,
                seats=args.seats, effort=args.effort,
                critic_rounds=args.critic_rounds,
                service_tier=args.service_tier,
                label=lambda r: (
                    f"{r['task']}/{r['trial']} reward={r['reward']}"
                ),
                tag="tb:score", on_record=on_record)
        # a trial that errored never reaches scored: emit its task's
        # surviving rows so the TSV stays the complete table
        for task in sorted(remaining):
            if task not in emitted:
                emit(task)
    if not live_tsv:
        print(header, flush=True)
        for task in finished:
            render(task)
    print("[tb:score] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
