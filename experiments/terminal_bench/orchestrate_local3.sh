#!/bin/bash
# g6 pipeline on the local Mac (moved off the server): prepull 9 images,
# replay group 6 to 45 with babysit, then run g6 GV after the local2 chain
# (g1 GV -> g1 top-up -> g2 GV) has fully finished, keeping local GV at 8.
cd "$(dirname "$0")"
export PATH="$HOME/bin-shim:$PATH"
PY=../../.venv/bin/python3
DS="../vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash"
log(){ echo "[orch3 $(date '+%m-%d %H:%M:%S')] $*"; }
count(){ ls "results/jobs/lav-official-g6-k5"/*/result.json 2>/dev/null | wc -l | tr -d ' '; }
clean_partial(){ for d in "results/jobs/lav-official-g6-k5"/*__*; do [ -f "$d/result.json" ] || rm -rf "$d"; done; }

# --- prepull g6 images (sequential) ---
for t in $(awk '/^# group 6 /{f=1;next} /^# group/{f=0} f&&/^[a-z]/' official_groups.txt); do
  docker image inspect "alexgshaw/$t:20251031" >/dev/null 2>&1 && continue
  log "pulling $t"
  docker pull "alexgshaw/$t:20251031" >/dev/null 2>&1 || log "pull failed: $t"
done
log "g6 prepull done"

# --- replay group 6 with babysit ---
tries=0
while [ "$(count)" -lt 45 ] && [ $tries -lt 20 ]; do
  if ! pgrep -f "replay_lav.py.*--group 6 " >/dev/null; then
    clean_partial
    tries=$((tries+1))
    log "g6 replay attempt $tries, have $(count)/45"
    nohup $PY replay_lav.py --dataset "$DS" --group 6 --workers 4 \
      >> results/replay_g6_local.log 2>&1 &
  fi
  sleep 60
done
log "g6 replay done: $(count)/45"

# --- g6 GV after the local2 chain finishes ---
while pgrep -f "orchestrate_local2.sh" >/dev/null; do sleep 300; done
log "local2 chain finished - starting g6 GV workers=8"
$PY score.py --job lav-official-g6-k5 --seats codex --model gpt-5.6-luna \
  --effort medium --workers 8 > results/g6_gv.tsv 2> results/g6_gv.log
log "g6 GV finished - LOCAL g6 ALL DONE"
