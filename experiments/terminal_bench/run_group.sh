#!/bin/zsh
# One-command group runner for the all89 rollout (user policy 2026-08-28):
#   generation -> completeness gate -> LAV baseline -> GV scoring -> report.
# Never re-runs a task that any earlier config already picked correctly.
#
# Usage: ./run_group.sh <group-name> <task ...>
#   e.g. ./run_group.sh g2 adaptive-rejection-sampler bn-fit-modify ...
# Env knobs: GEN_N (harbor concurrency, default 2), GV_WORKERS (default 10),
#            MODEL (default gpt-5.6-terra), EFFORT (default medium)
set -euo pipefail
setopt null_glob
# Locked-keychain guard: a shim credential helper reports "no stored
# credentials" so docker pulls public images anonymously instead of dying
# on the non-interactive login keychain.
export PATH=$HOME/bin-shim:$PATH
HERE=${0:a:h}
GROUP=$1; shift
TASKS=($@)
JOB="rec-codex-terra-medium-tb21-${GROUP}-k5-r1"
MODEL=${MODEL:-gpt-5.6-terra}
EFFORT=${EFFORT:-medium}
GEN_N=${GEN_N:-2}
GV_WORKERS=${GV_WORKERS:-10}
PY=$HERE/../../.venv/bin/python3
RESULTS=$HERE/results

echo "[group] $GROUP: ${#TASKS} tasks -> job $JOB"

# 0. quota preflight: fail fast instead of burning a half-broken batch
if ! timeout 120 codex exec --model $MODEL --skip-git-repo-check --json \
      -c model_reasoning_effort=low -- "Reply with the single word: ok" \
      2>/dev/null | grep -q turn.completed; then
  echo "[group] ABORT: codex quota/capacity preflight failed" >&2
  exit 3
fi

# 1. generation (skipped when the job dir already has 5 result.json per task)
need_gen=0
for t in $TASKS; do
  files=($RESULTS/jobs/$JOB/${t}__*/result.json(N))
  (( ${#files} < 5 )) && need_gen=1
done
if [[ $need_gen == 1 ]]; then
  echo "[group] generating solver traces (k=5, n=$GEN_N)"
  $PY $HERE/run.py --arm rec-codex --model $MODEL --effort $EFFORT \
      --k 5 --n-concurrent $GEN_N --job-name $JOB \
      --tasks-file $HERE/tasks_all89.txt --task $TASKS \
      >> $RESULTS/${GROUP}_gen.log 2>&1
fi

# 2. completeness gate: every task needs 5 non-errored trials
$PY - "$RESULTS/jobs/$JOB" $TASKS <<'EOF'
import json, glob, sys
jobdir, tasks = sys.argv[1], sys.argv[2:]
bad = []
for t in tasks:
    ok = 0
    for p in glob.glob(f"{jobdir}/{t}__*/result.json"):
        r = json.load(open(p))
        if not r.get("exception_info"):
            ok += 1
    if ok < 5:
        bad.append(f"{t}={ok}/5")
if bad:
    sys.exit("[group] generation incomplete: " + ", ".join(bad))
print("[group] generation complete: all tasks 5/5")
EOF

# 3. LAV baseline (skips tasks it already scored for this job)
$PY $HERE/lav.py --job $JOB --model $MODEL \
    >> $RESULTS/${GROUP}_lav.log 2>&1 || echo "[group] WARN: lav failed"

# 4. GV scoring — carry-forward: skip any task already picked correctly
#    (across every config) per the immutable-when-correct rule.
SKIP=$($PY - "$RESULTS/rec_scores.jsonl" $JOB <<'EOF'
import json, sys, collections
BAND={'verified':5,'supported':4,'partial':3,'unverified':2,'refuted':0}
path, job = sys.argv[1], sys.argv[2]
try: rows=[json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError: rows=[]
best={}
for r in rows:
    if r.get('job')==job and not (r.get('features') or {}).get('unscoreable'):
        best[r['trial']]=r
bytask=collections.defaultdict(list)
for r in best.values(): bytask[r['trial'].split('__')[0]].append(r)
done=[]
for task,rs in bytask.items():
    if len(rs)<5: continue
    if not any(x.get('reward')==1.0 for x in rs): done.append(task); continue
    def key(x):
        f=x.get('features') or {}
        fo=f.get('final_obligations') or {}; tot=f.get('final_obligations_total') or 0
        return (BAND.get(f.get('verdict'),2),(fo.get('verified',0)/tot if tot else 0),f.get('score') or 0)
    if max(rs,key=key).get('reward')==1.0: done.append(task)
print(' '.join(done))
EOF
)
RUN=()
for t in $TASKS; do
  [[ " $SKIP " == *" $t "* ]] || RUN+=$t
done
if (( ${#RUN} > 0 )); then
  echo "[group] GV scoring ${#RUN} tasks (skipping already-correct: ${SKIP:-none})"
  $PY $HERE/score.py --job $JOB --seats codex --model $MODEL --effort $EFFORT \
      --workers $GV_WORKERS --task $RUN \
      > $RESULTS/${GROUP}_gv.tsv 2> $RESULTS/${GROUP}_gv.log
else
  echo "[group] nothing to score (all tasks already correct)"
fi

# 5. report
$PY $HERE/bon_report.py --job $JOB --merge-configs
