#!/bin/sh
cd "$(dirname "$0")"
/opt/homebrew/bin/lean Sum1To100.lean > lean_output.txt 2>&1
echo "EXIT_CODE=$?" >> lean_output.txt
cat lean_output.txt
