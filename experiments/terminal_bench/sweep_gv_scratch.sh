#!/bin/bash
# Reclaim gv/work scratch (offline homes, pip deps, tmp) from FINISHED trials
# only — a trial is finished once gv/score.json exists. Certificates, events,
# claims.sqlite, world snapshot and score.json are never touched.
# Runs forever; sweeps every 10 minutes.
while :; do
  n=0
  for w in "$HOME"/.gv-scoring/terminal_bench/lav-official-*/*/r*/*/gv/work; do
    [ -d "$w" ] || continue
    [ -f "$(dirname "$w")/score.json" ] || continue
    rm -rf "$w" && n=$((n+1))
  done
  [ "$n" -gt 0 ] && echo "[sweep $(date '+%m-%d %H:%M')] cleared scratch of $n finished trials"
  sleep 600
done
