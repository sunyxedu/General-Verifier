#!/bin/bash
#SBATCH --job-name=gv-seal-verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SNAPSHOT_ROOT="${GV_SNAPSHOT_ROOT:?set GV_SNAPSHOT_ROOT}"
CODE_ROOT="${GV_SEAL_CODE_ROOT:?set GV_SEAL_CODE_ROOT}"
VERIFY_ROOT="${GV_SEAL_VERIFY_ROOT:?set GV_SEAL_VERIFY_ROOT}"
PYTHON="${GV_SEAL_PYTHON:-python3}"

if [[ ! -d "$SNAPSHOT_ROOT" || -L "$SNAPSHOT_ROOT" ]]; then
  echo "invalid snapshot root: $SNAPSHOT_ROOT" >&2
  exit 2
fi
for path in \
  "$SNAPSHOT_ROOT/candidate-seals.jsonl" \
  "$SNAPSHOT_ROOT/candidate-verifier-inputs.jsonl" \
  "$CODE_ROOT/experiments/common/harbor_candidate_seal.py"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "missing regular input: $path" >&2
    exit 2
  fi
done

mkdir -p "$VERIFY_ROOT"
if [[ -e "$VERIFY_ROOT/candidate-specs.jsonl" \
      || -e "$VERIFY_ROOT/candidate-seals.jsonl" \
      || -e "$VERIFY_ROOT/candidate-verifier-inputs.jsonl" ]]; then
  echo "verification output already exists: $VERIFY_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "snapshot=$SNAPSHOT_ROOT"
echo "verify_root=$VERIFY_ROOT"

"$PYTHON" -m experiments.common.harbor_candidate_seal \
  --job-dir "$SNAPSHOT_ROOT" \
  --specs "$VERIFY_ROOT/candidate-specs.jsonl" \
  --seals "$VERIFY_ROOT/candidate-seals.jsonl" \
  --verifier-inputs "$VERIFY_ROOT/candidate-verifier-inputs.jsonl"

test "$(wc -l < "$VERIFY_ROOT/candidate-seals.jsonl")" -eq 138
test "$(wc -l < "$VERIFY_ROOT/candidate-verifier-inputs.jsonl")" -eq 138
cmp "$SNAPSHOT_ROOT/candidate-seals.jsonl" \
    "$VERIFY_ROOT/candidate-seals.jsonl"
cmp "$SNAPSHOT_ROOT/candidate-verifier-inputs.jsonl" \
    "$VERIFY_ROOT/candidate-verifier-inputs.jsonl"

sha256sum \
  "$SNAPSHOT_ROOT/candidate-seals.jsonl" \
  "$VERIFY_ROOT/candidate-seals.jsonl" \
  "$SNAPSHOT_ROOT/candidate-verifier-inputs.jsonl" \
  "$VERIFY_ROOT/candidate-verifier-inputs.jsonl"
echo "SEAL_VERIFY_OK candidates=138"
