#!/bin/bash
# Remote pipeline: prepull g5/g6, babysit g3-g6 replays, sequential GV chain (24 workers).
cd "$HOME/General-Verifier/experiments/terminal_bench"
PY=../../.venv/bin/python3
DS="../vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash"
log(){ echo "[orch $(date '+%m-%d %H:%M:%S')] $*"; }
count(){ ls "results/jobs/lav-official-g$1-k5"/*/result.json 2>/dev/null | wc -l; }
target(){ case $1 in 6) echo 45;; *) echo 80;; esac; }
rworkers(){ case $1 in 4) echo 32;; 6) echo 16;; *) echo 24;; esac; }
gtasks(){ awk -v pat="# group $1 " "index(\$0,pat)==1{f=1;next} /^# group/{f=0} f&&/^[a-z]/" official_groups.txt; }
clean_partial(){ for d in "results/jobs/lav-official-g$1-k5"/*__*; do [ -f "$d/result.json" ] || rm -rf "$d"; done; }

# --- prepull g5 then g6; flag files gate replay start ---
(
  for g in 5 6; do
    for t in $(gtasks $g); do
      docker image inspect "alexgshaw/$t:20251031" >/dev/null 2>&1 \
        || docker pull "alexgshaw/$t:20251031" >/dev/null 2>&1 || log "pull failed: $t"
    done
    touch "results/.prepull_g${g}_done"; log "g$g prepull done"
  done
) &

# --- babysitter: keep g3-g6 replay drivers alive until each group is complete ---
(
  while :; do
    remaining=0
    for g in 3 4 5 6; do
      c=$(count $g); t=$(target $g)
      [ "$c" -ge "$t" ] && continue
      remaining=1
      if [ $g -ge 5 ] && [ ! -f "results/.prepull_g${g}_done" ]; then continue; fi
      if ! pgrep -f "replay_lav.py.*--group $g " >/dev/null; then
        log "g$g driver not running at $c/$t - (re)starting workers=$(rworkers $g)"
        clean_partial $g
        nohup $PY replay_lav.py --dataset "$DS" --group $g \
          --workers $(rworkers $g) >> "results/replay_g$g.log" 2>&1 &
      fi
    done
    [ $remaining -eq 0 ] && { log "all replays complete"; break; }
    sleep 120
  done
) &

# --- sequential GV chain, 24 workers each ---
for g in 3 4 5 6; do
  while [ "$(count $g)" -lt "$(target $g)" ]; do sleep 120; done
  log "g$g replay complete - starting GV workers=24"
  $PY score.py --job "lav-official-g$g-k5" --seats codex --model gpt-5.6-luna \
    --effort medium --workers 24 > "results/g${g}_gv.tsv" 2> "results/g${g}_gv.log"
  log "g$g GV done"
done
log "REMOTE ALL DONE"
