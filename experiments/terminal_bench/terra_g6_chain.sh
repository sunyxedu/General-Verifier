#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "gpt-5.6-terra --effort medium --workers 5 --task db-wal-recovery" >/dev/null 2>&1; do sleep 60; done
exec ../../.venv/bin/python3 score.py --job lav-official-g6-k5 --seats codex --model gpt-5.6-terra --effort medium --workers 5 --task video-processing torch-pipeline-parallelism > results/terra_g6.tsv 2>> results/terra_g6.log
