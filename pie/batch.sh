#!/bin/bash
# pie/batch.sh SET name1 "args1" name2 "args2" ...   up to 3 runs at once; one line each: phone errors, then UTMOS
set -u
SET=$1; shift
PY=$HOME/AI/PIE/tts/vc/.venv-eval/bin/python
run() {
  local a b
  a=$(OMP_NUM_THREADS=8 $PY -I pie/eval.py "$1" --set "$SET" --jobs 8 --extra "$2" 2>&1 | grep -v -i warn | tail -2 | head -1)
  b=""; [ -n "${MOS:-}" ] && b=$(OMP_NUM_THREADS=8 $PY -I pie/mos.py "$1" 2>&1 | tail -1 | sed "s/^[^:]*: //")
  echo "$a | $b"
}
while [ $# -gt 0 ]; do
  n=0
  while [ $# -gt 0 ] && [ $n -lt 3 ]; do run "$1" "$2" & shift 2; n=$((n+1)); done
  wait
done
