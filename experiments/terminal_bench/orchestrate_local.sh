#!/bin/bash
# Local pipeline: babysit g1 replay -> g1 GV(8) ; prepull g2 -> g2 replay -> g2 GV(8).
# GV runs are sequential (shared Luna account, ~8 concurrent sessions locally).
cd "$(dirname "$0")"
export PATH="$HOME/bin-shim:$PATH"
PY=../../.venv/bin/python3
DS="../vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash"
log(){ echo "[orch $(date '+%m-%d %H:%M:%S')] $*"; }
count(){ ls "results/jobs/$1"/*/result.json 2>/dev/null | wc -l | tr -d ' '; }
clean_partial(){ for d in "results/jobs/$1"/*__*; do [ -f "$d/result.json" ] || rm -rf "$d"; done; }

# --- prepull group-2 images now (background; replay pulls stragglers on demand) ---
(
  for t in $(awk '/^# group 2 /{f=1;next} /^# group/{f=0} f&&/^[a-z]/' official_groups.txt); do
    docker image inspect "alexgshaw/$t:20251031" >/dev/null 2>&1 \
      || docker pull "alexgshaw/$t:20251031" >/dev/null 2>&1 || log "pull failed: $t"
  done
  log "g2 prepull done"
) &
PREPULL=$!

# --- babysit g1 replay to 80 (restart driver if it dies) ---
while [ "$(count lav-official-g16-k5)" -lt 80 ]; do
  if ! pgrep -f "replay_lav.py.*--group 1 " >/dev/null; then
    log "g1 driver dead at $(count lav-official-g16-k5)/80 - restarting"
    clean_partial lav-official-g16-k5
    nohup $PY replay_lav.py --dataset "$DS" --group 1 --job lav-official-g16-k5 \
      --workers 4 >> results/replay_g1_fin.log 2>&1 &
  fi
  sleep 60
done
log "g1 replay 80/80 - starting g1 GV workers=8"
$PY score.py --job lav-official-g16-k5 --seats codex --model gpt-5.6-luna \
  --effort medium --workers 8 > results/g1_gv.tsv 2> results/g1_gv.log &
G1GV=$!

# --- g2 replay with babysit (bounded attempts) ---
wait $PREPULL 2>/dev/null
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

# --- g2 GV after g1 GV finishes ---
wait $G1GV
log "g1 GV finished - starting g2 GV workers=8"
$PY score.py --job lav-official-g2-k5 --seats codex --model gpt-5.6-luna \
  --effort medium --workers 8 > results/g2_gv.tsv 2> results/g2_gv.log
log "g2 GV finished - LOCAL ALL DONE"
