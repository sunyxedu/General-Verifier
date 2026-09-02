#!/bin/bash
# Local chain: verify the discriminative (mixed-reward) pools first, all three
# local groups in parallel, then sweep the remaining tasks of each group.
# Judged trials are served from results/rec_scores.jsonl, so the sweeps are cheap.
cd "$(dirname "$0")"
export PATH="$HOME/bin-shim:$PATH"
PY=../../.venv/bin/python3
log(){ echo "[mixed $(date '+%m-%d %H:%M:%S')] $*"; }
gv(){ # gv <job> <tag> <workers> [tasks...]
  local job=$1 tag=$2 w=$3; shift 3
  $PY score.py --job "$job" --seats codex --model gpt-5.6-luna --effort medium \
    --workers "$w" ${1:+--task "$@"} > "results/${tag}.tsv" 2> "results/${tag}.log"
}

G1_MIXED="bn-fit-modify cancel-async-tasks dna-assembly dna-insert extract-elf"
G2_MIXED="chess-best-move db-wal-recovery extract-moves-from-video gcode-to-text"
G6_MIXED="torch-pipeline-parallelism torch-tensor-parallelism train-fasttext video-processing winning-avg-corewars write-compressor"

log "phase 1: mixed pools (g1/g2/g6 in parallel)"
gv lav-official-g16-k5 g1_gv_mixed 7 $G1_MIXED &
P1=$!
gv lav-official-g2-k5  g2_gv_mixed 7 $G2_MIXED &
P2=$!
gv lav-official-g6-k5  g6_gv_mixed 7 $G6_MIXED &
P6=$!
wait $P1 $P2 $P6
log "phase 1 done - mixed pools of g1/g2/g6 judged"

log "phase 2: full sweeps (cached judgments skipped)"
gv lav-official-g16-k5 g1_gv_full 7 &
Q1=$!
gv lav-official-g2-k5  g2_gv_full 7 &
Q2=$!
gv lav-official-g6-k5  g6_gv_full 7 &
Q6=$!
wait $Q1 $Q2 $Q6
log "LOCAL ALL DONE"
