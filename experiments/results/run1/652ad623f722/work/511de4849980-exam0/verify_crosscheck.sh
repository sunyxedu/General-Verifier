#!/bin/sh
# 义务：确认 2^11 - 1 = 2047，用 bc 独立重算并与 python3 交叉核对
set -e

echo "== bc computation of 2^11 - 1 =="
BC_RESULT=$(echo "2^11 - 1" | bc)
echo "bc result: $BC_RESULT"

echo "== python3 computation of 2^11 - 1 =="
PY_RESULT=$(python3 -c "print(2**11 - 1)")
echo "python3 result: $PY_RESULT"

echo "== cross-check =="
if [ "$BC_RESULT" = "2047" ] && [ "$PY_RESULT" = "2047" ] && [ "$BC_RESULT" = "$PY_RESULT" ]; then
  echo "PASS: both bc and python3 agree that 2^11 - 1 = 2047"
  exit 0
else
  echo "FAIL: mismatch. bc=$BC_RESULT python3=$PY_RESULT"
  exit 1
fi
