#!/bin/bash
# v2: a manual g1 GV over the 13 full-pool tasks is already running.
# This script: babysit g1 replay to 80 -> g2 replay to 80; once the manual GV
# exits, top-up g1 GV (no --task; judged trials come from cache); then g2 GV.
cd "$(dirname "$0")"
export PATH="$HOME/bin-shim:$PATH"
PY=../../.venv/bin/python3
DS="../vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash"
log(){ echo "[orch2 $(date '+%m-%d %H:%M:%S')] $*"; }
count(){ ls "results/jobs/$1"/*/result.json 2>/dev/null | wc -l | tr -d ' '; }
clean_partial(){ for d in "results/jobs/$1"/*__*; do [ -f "$d/result.json" ] || rm -rf "$d"; done; }

# --- babysit g1 replay to 80 ---
while [ "$(count lav-official-g16-k5)" -lt 80 ]; do
  if ! pgrep -f "replay_lav.py.*--group 1 " >/dev/null; then
    log "g1 driver dead at $(count lav-official-g16-k5)/80 - restarting"
    clean_partial lav-official-g16-k5
    nohup $PY replay_lav.py --dataset "$DS" --group 1 --job lav-official-g16-k5 \
      --workers 4 >> results/replay_g1_fin.log 2>&1 &
  fi
  sleep 60
done
log "g1 replay 80/80"

# --- g2 replay with babysit (bounded attempts) ---
tries=0
while [ "$(count lav-official-g2-k5)" -lt 80 ] && [ $tries -lt 15 ]; do
  if ! pgrep -f "replay_lav.py.*--group 2 " >/dev/null; then
    clean_partial lav-official-g2-k5
    tries=$((tries+1))
    log "g2 replay attempt $tries, have $(count lav-official-g2-k5)/80"
    nohup $PY replay_lav.py --dataset "$DS" --group 2 --workers 4 \
      >> results/replay_g2.log 2>&1 &
  fi
  sleep 60
done
log "g2 replay done: $(count lav-official-g2-k5)/80"

# --- g1 GV top-up after the manual run exits ---
while pgrep -f "score.py --job lav-official-g16" >/dev/null; do sleep 120; done
log "manual g1 GV finished - running g1 top-up GV"
$PY score.py --job lav-official-g16-k5 --seats codex --model gpt-5.6-luna \
  --effort medium --workers 8 > results/g1_gv_full.tsv 2> results/g1_gv_full.log
log "g1 GV complete"

# --- g2 GV ---
$PY score.py --job lav-official-g2-k5 --seats codex --model gpt-5.6-luna \
  --effort medium --workers 8 > results/g2_gv.tsv 2> results/g2_gv.log
log "g2 GV finished - LOCAL ALL DONE"
