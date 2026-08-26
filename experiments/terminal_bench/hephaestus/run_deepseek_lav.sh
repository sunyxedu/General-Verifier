#!/bin/bash
# Run frozen-candidate DeepSeek LLM-as-a-Verifier scoring on a compute node.
#
# The default scope remains the six mixed-reward tasks used by the pilot:
#
#   ./run_deepseek_lav.sh
#
# Score all 20 frozen tasks with a separate, resumable result ledger while
# reusing the same per-comparison cache:
#
#   ./run_deepseek_lav.sh --scope all20

set -euo pipefail

RUN_ROOT="${GV_RUN_ROOT:-$HOME/gv-runs/rec-codex-terra-medium-tb21-strat20-k5-r1}"
CODE_ROOT="$RUN_ROOT/code"
CANDIDATE_ROOT="$RUN_ROOT/candidates"
HARBOR_ROOT="$RUN_ROOT/harbor-job"
OUTPUT_ROOT="$RUN_ROOT/lav-deepseek"
PYTHON="${GV_LAV_PYTHON:-$HOME/gv-vllm/bin/python}"
API_KEY_FILE="${GV_VLLM_API_KEY_FILE:-$HOME/.config/gv/vllm_api_key}"
BASE_URL="${GV_VLLM_BASE_URL:-http://127.0.0.1:8371/v1}"
MODEL="${GV_VLLM_MODEL:-deepseek-v4-flash-0731-h200-high}"
MODEL_REVISION="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SERVE_CONFIG_SHA256="8e60ecb0493f766abdefbc8966abb6fd7fa1514673fe3970c8f86e3a725f2012"
SOURCE_HARBOR_ROOT="/Users/hemlix/Documents/General Verifier/experiments/terminal_bench/results/jobs/rec-codex-terra-medium-tb21-strat20-k5-r1"

SCOPE="${GV_LAV_SCOPE:-swing6}"
if [[ "${1:-}" == "--scope" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--scope requires swing6 or all20" >&2
    exit 2
  fi
  SCOPE="$2"
  shift 2
elif [[ "${1:-}" == --scope=* ]]; then
  SCOPE="${1#--scope=}"
  shift
fi

TASK_ARGS=()
case "$SCOPE" in
  swing6)
    RESULTS_BASENAME="swing6.results.jsonl"
    TASK_ARGS=(
      --task chess-best-move
      --task count-dataset-tokens
      --task db-wal-recovery
      --task kv-store-grpc
      --task largest-eigenval
      --task sqlite-db-truncate
    )
    ;;
  all20)
    RESULTS_BASENAME="all20.results.jsonl"
    ;;
  *)
    echo "invalid LAV scope: $SCOPE (expected swing6 or all20)" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "missing LAV Python: $PYTHON" >&2
  exit 2
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "missing vLLM API key file: $API_KEY_FILE" >&2
  exit 2
fi
for path in \
  "$CODE_ROOT/experiments/score_llm_verifier.py" \
  "$CANDIDATE_ROOT/specs.jsonl" \
  "$CANDIDATE_ROOT/seals.jsonl" \
  "$CANDIDATE_ROOT/verifier-inputs.jsonl"; do
  if [[ ! -f "$path" ]]; then
    echo "missing frozen scoring input: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/cache"
LOCK_FILE="${GV_LAV_LOCK_FILE:-$OUTPUT_ROOT/driver.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another DeepSeek LAV driver holds $LOCK_FILE" >&2
  exit 3
fi
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/experiments/vendor/llm-as-a-verifier${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY
OPENAI_API_KEY="$(<"$API_KEY_FILE")"

SERVED_MODEL_IDENTITY="deepseek-ai/DeepSeek-V4-Flash-0731@$MODEL_REVISION;vllm=0.27.1;tp=2;expert-parallel=true;h200-nvl=2;serve=$SERVE_CONFIG_SHA256"
echo "[deepseek-lav] scope=$SCOPE results=$OUTPUT_ROOT/$RESULTS_BASENAME cache=$OUTPUT_ROOT/cache"

"$PYTHON" -u "$CODE_ROOT/experiments/score_llm_verifier.py" \
  --candidate-specs "$CANDIDATE_ROOT/specs.jsonl" \
  --candidate-seals "$CANDIDATE_ROOT/seals.jsonl" \
  --candidate-verifier-inputs "$CANDIDATE_ROOT/verifier-inputs.jsonl" \
  --candidate-path-map "$SOURCE_HARBOR_ROOT=$HARBOR_ROOT" \
  --backend vllm-logprob \
  --model "$MODEL" \
  --served-model-identity "$SERVED_MODEL_IDENTITY" \
  --n-evaluations 2 \
  --pivots 1 \
  --seed 0 \
  --max-workers 8 \
  --trace-chars 0 \
  --expected-candidates 5 \
  --cache-dir "$OUTPUT_ROOT/cache" \
  --results-jsonl "$OUTPUT_ROOT/$RESULTS_BASENAME" \
  "${TASK_ARGS[@]}" \
  "$@"
