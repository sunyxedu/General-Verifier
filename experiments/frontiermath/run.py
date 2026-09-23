"""FrontierMath (official public sample, 12 problems) — solver arms.

Per problem the solver gets ONE prompt in a fresh scratch directory (it may
write and run code there). Answers are mechanically checked against the
official published answers (check.py).

Arms — every arm is ONE prompt to a solver; only the attached harness
differs:
  --arm gv       Claude Code + full General Verifier mounted (MCP server +
                 hooks + solver protocol)
  --arm gv-codex Codex CLI + the same General Verifier, mounted through a
                 run-scoped CODEX_HOME (gv MCP server, hooks, protocol as
                 AGENTS.md). Use for a codex-family solver such as
                 gpt-5.6-luna.
  --arm turbo    Claude Code + TurboAgent, the official LLM-as-a-Verifier
                 Claude Code plugin, as shipped (proxy fans out candidates
                 per model call, logprob verifier picks; needs
                 VERTEX_API_KEY)
  --arm plain    stock headless Claude Code
  --arm codex    stock headless Codex CLI (--model names a codex model,
                 e.g. gpt-5.6-terra; default = the CLI's default). The
                 session rollout is captured and converted into the same
                 ledger layout as `rec`, so codex traces also score
                 through the offline ORM path (score.py --arm codex)
  --arm rec      stock headless Claude Code + recording hooks only (no MCP,
                 no protocol — the solver's distribution is untouched);
                 leaves a ledger for the offline ORM path (score.py)

Usage:
  uv run python experiments/frontiermath/run.py --arm gv
  uv run python experiments/frontiermath/run.py --arm turbo
  uv run python experiments/frontiermath/run.py --arm plain
  uv run python experiments/frontiermath/run.py --arm codex
  uv run python experiments/frontiermath/run.py --arm rec --n 4
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _path not in sys.path:            # experiments/, then the repo root
        sys.path.insert(0, _path)

from common import progress                              # noqa: E402
from common import results as res                        # noqa: E402
from common import solver                                # noqa: E402
from common import turbo                                 # noqa: E402
from check import matches, parse_final                   # noqa: E402

# outside the repo (see common.solver) and outside /tmp: a reboot wipes
# /tmp, and with it every in-flight trial's ledger and evidence
WORKROOT = os.path.expanduser("~/.cache/gv-experiments/frontiermath-work")

ANSWER_SPEC = {
    "integer": "The answer is a single (possibly very large) integer.",
    "sympy": ("The answer is an exact closed-form expression; write it as "
              "a sympy-parseable expression (e.g. sqrt(3)/36 + ...)."),
}

BASE_PROMPT = """\
{statement}

{answer_spec} You may write and run code in this directory. End your reply \
with a line of exactly this form: FINAL ANSWER: <answer>
"""

GV_SUFFIX = """
Follow the claim protocol in CLAUDE.md while you work: commit your key \
intermediate results as claims, commit a final claim whose text ends with \
'The answer is <answer>.', and request gv.certificate on it before you \
finish. Report the verdict verbatim.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["gv", "gv-codex", "turbo", "plain", "codex", "rec"])
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-problem seconds; 0 = no timeout (default)")
    ap.add_argument("--candidates", type=int, default=3,
                    help="turbo: parallel candidates per model call")
    ap.add_argument("--pid", action="append", default=None,
                    help="run only this problem id (e.g. fm-bmo-optimization)")
    ap.add_argument("--workers", type=int, default=1,
                    help="samples run concurrently (default 1 = serial); "
                    "codex rollout capture pairs by workdir cwd, so "
                    "parallel codex runs cannot cross-capture")
    ap.add_argument("--effort", default="max",
                    help="the solver's reasoning effort (codex arms)")
    ap.add_argument("--seat-effort", default="max",
                    help="the verifier seats' reasoning effort (gv-codex)")
    ap.add_argument("--seat-tier", default="standard",
                    choices=["standard", "fast"],
                    help="the verifier seats' service tier (gv-codex)")
    ap.add_argument("--web", action="store_true",
                    help="allow solver web access (default OFF: the sample "
                    "problems' answers are published — a codex solver was "
                    "caught fetching the FrontierMath paper and the "
                    "epoch-research solution repo mid-run)")
    args = ap.parse_args()

    with open(os.path.join(HERE, "problems.json")) as f:
        problems = json.load(f)
    results_path = os.path.join(HERE, "results", args.arm, "results.jsonl")
    done = res.recorded(results_path, "pid", "sample")
    # sample-major order: s0 of every problem, then s1, ... — concurrent
    # workers land on DIFFERENT problems instead of five-way hammering the
    # first hard one, and early coverage spans the whole problem set
    todo = [(p, s) for s in range(args.n) for p in problems
            if (p["id"], s) not in done]
    if args.pid:
        wanted = set(args.pid)          # repeatable: --pid a --pid b
        todo = [(p, s) for p, s in todo if p["id"] in wanted]
    print(f"[fm:{args.arm}] {len(todo)} runs ({len(done)} recorded), "
          f"web={'ON' if args.web else 'off'}", flush=True)
    bar = progress.Board(len(todo))

    # --model's default targets the Claude arms; for codex, the default
    # means "the Codex CLI's default model", and the effective name is
    # recorded so results rows never claim a codex run used sonnet
    codex_model, effective_model = None, args.model
    if args.arm in ("codex", "gv-codex"):
        from verifier.llm.codex_backend import DEFAULT_MODEL
        codex_model = None if args.model == "sonnet" else args.model
        effective_model = codex_model or DEFAULT_MODEL

    proxy_url = None
    if args.arm == "turbo":
        turbo.check_keys()
        turbo_dir = os.path.join(HERE, "results", "turbo")
        turbo.write_config(turbo_dir, solver_model=args.model,
                           num_candidates=args.candidates)
        proxy = turbo.Proxy(turbo_dir)
        proxy_url = proxy.__enter__()

    append_lock = threading.Lock()

    def attempt(p, s):
        work = os.path.join(WORKROOT, args.arm, p["id"], f"s{s}")
        os.makedirs(work, exist_ok=True)
        prompt = BASE_PROMPT.format(statement=p["statement"],
                                    answer_spec=ANSWER_SPEC[p["answer_type"]])
        if args.arm == "gv":
            task = prompt                  # clean statement for task.md
            prompt += GV_SUFFIX
            solver.mount_gv(work, model=args.model)
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             mcp=True, timeout=args.timeout,
                                             task=task, web=args.web)
        elif args.arm == "rec":
            solver.mount_recorder(work)
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             record=True,
                                             timeout=args.timeout,
                                             web=args.web)
            solver.record_final(work, out)
        elif args.arm == "gv-codex":
            # Same system as the gv arm, driven by a codex-family solver:
            # the harness is mounted through a run-scoped CODEX_HOME, and
            # the task statement is the contract the final claim is
            # adjudicated against.
            from verifier import baseline
            task = prompt
            prompt += GV_SUFFIX
            state = os.path.join(work, ".gv")
            os.makedirs(state, exist_ok=True)
            with open(os.path.join(state, "task.md"), "w") as f:
                f.write(task)
            baseline.snapshot(work, os.path.join(state, "baseline"))
            home = solver.mount_gv_codex(
                work, model=codex_model, effort=args.seat_effort,
                service_tier=args.seat_tier,
                # this benchmark publishes its problems with their answers
                seat_search=args.web, web=args.web)
            out, seconds = solver.run_codex(work, prompt, model=codex_model,
                                            timeout=args.timeout,
                                            web=args.web, codex_home=home,
                                            effort=args.effort)
        elif args.arm == "codex":
            # Ground truth + rollout capture so a codex trace scores through
            # the same offline ORM path as `rec`: codex has no recording
            # hooks, but its rollout log carries the same substance and is
            # converted into the hook-feed layout right here. Rollouts pair
            # by the session_meta cwd (this run's unique workdir), so
            # parallel runs — or a concurrently scoring verifier sharing
            # the same codex install — cannot cross-capture.
            import shutil
            import time as _time
            from verifier import baseline
            from verifier.stream import codex as codex_stream
            state = os.path.join(work, ".gv")
            os.makedirs(state, exist_ok=True)
            with open(os.path.join(state, "task.md"), "w") as f:
                f.write(prompt)
            baseline.snapshot(work, os.path.join(state, "baseline"))
            since = _time.time()
            out, seconds = solver.run_codex(work, prompt, model=codex_model,
                                            timeout=args.timeout,
                                            web=args.web, effort=args.effort)
            rollout = solver.rollout_for(work, since=since - 5)
            if rollout is None:
                bar.log(f"[fm:codex] WARNING {p['id']}#{s}: no rollout with "
                        f"cwd={work} — trace will be unscoreable offline")
            else:
                shutil.copy2(rollout, os.path.join(state, "rollout.jsonl"))
                codex_stream.convert(os.path.join(state, "rollout.jsonl"),
                                     state)
        else:  # plain / turbo — same stock solver, ± the proxy
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             timeout=args.timeout,
                                             base_url=proxy_url,
                                             web=args.web)
        transcripts_dir = os.path.join(HERE, "results", args.arm,
                                       "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        transcript_file = os.path.join(transcripts_dir,
                                       f"{p['id']}_s{s}.txt")
        with open(transcript_file, "w") as f:
            f.write(out)
        answer = parse_final(out)
        rec = {"pid": p["id"], "sample": s, "arm": args.arm,
               "model": effective_model, "answer": answer,
               "solved": matches(answer, p), "seconds": seconds,
               "workdir": work, "transcript_file": transcript_file,
               "output_tail": out[-300:]}
        if args.arm in ("gv", "gv-codex"):
            gvo = solver.gv_outcome(work)
            # prefer the certified final claim's answer when present
            claim_answer = parse_final(gvo["final_text"] or "")
            if claim_answer is None and gvo["final_text"]:
                import re
                m = re.findall(r"answer is\s*([^.]+)\.?\s*$",
                               gvo["final_text"], re.IGNORECASE)
                claim_answer = m[-1].strip() if m else None
            if claim_answer is not None:
                rec["answer"] = claim_answer
                rec["solved"] = matches(claim_answer, p)
            rec.update({k: v for k, v in gvo.items() if k != "final_text"})
            rec["final_text"] = (gvo["final_text"] or "")[:200]
        with append_lock:
            res.append(results_path, rec)
        return rec

    def one(i, p, s):
        # the spinner row appears when a worker actually picks the run up,
        # and is dropped again on any exit path before the error propagates
        job = bar.begin(f"{p['id']}#{s}")
        try:
            rec = attempt(p, s)
        except BaseException:
            bar.advance(job, 0.0)
            raise
        extra = (f" verdict={rec.get('verdict')}"
                 f" refuted={rec.get('refuted_claims')}"
                 if args.arm in ("gv", "gv-codex") else "")
        bar.advance(job, 1.0 if rec["solved"] else 0.0)
        bar.log(f"[fm:{args.arm} {i}/{len(todo)}] {p['id']}#{s} "
                f"answer={str(rec['answer'])[:40]} solved={rec['solved']}"
                f"{extra} ({rec['seconds']}s)")

    # one pool for serial and parallel alike: submission order = execution
    # order at workers=1, and a failed run is reported without killing the
    # batch either way
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one, i, p, s): (p, s)
                   for i, (p, s) in enumerate(todo, 1)}
        for future in as_completed(futures):
            p, s = futures[future]
            try:
                future.result()
            except Exception as err:      # noqa: BLE001 - reported, batch continues
                bar.log(f"[fm:{args.arm}] FAILED {p['id']}#{s}: "
                        f"{type(err).__name__}: {err}")

    bar.close()
    if args.arm == "turbo":
        proxy.__exit__(None, None, None)
    print(f"[fm:{args.arm}] done", flush=True)


if __name__ == "__main__":
    main()
