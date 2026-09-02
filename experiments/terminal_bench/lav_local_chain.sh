#!/bin/bash
# LAV baseline (Luna via codex endpoint, TEXT-FALLBACK mode) over the local
# groups' mixed tasks, sequential per job.
cd "$(dirname "$0")/../.."
export OPENAI_BASE_URL=http://127.0.0.1:8901/v1 OPENAI_API_KEY=dummy
LAV="uv run --no-project --with git+https://github.com/llm-as-a-verifier/llm-as-a-verifier python experiments/terminal_bench/lav.py"
R=experiments/terminal_bench/results
while pgrep -f "lav.py --job lav-official-g16" >/dev/null; do sleep 60; done
$LAV --job lav-official-g2-k5 --model gpt-5.6-luna --max-workers 4 \
  --task chess-best-move db-wal-recovery extract-moves-from-video gcode-to-text \
  > $R/lav_g2_mixed.log 2>&1
$LAV --job lav-official-g6-k5 --model gpt-5.6-luna --max-workers 4 \
  --task torch-pipeline-parallelism torch-tensor-parallelism train-fasttext video-processing winning-avg-corewars write-compressor \
  > $R/lav_g6_mixed.log 2>&1
echo "[lav-chain] local done"
