#!/bin/bash
# pie/batch2.sh SET name1 "args1" name2 "args2" ...   render each (12 jobs), then score with PhoneticXeus (pie/gop.py)
set -u
SET=$1; shift
PY=$HOME/AI/PIE/tts/vc/.venv-eval/bin/python
while [ $# -gt 0 ]; do
  n=$1; x=$2; shift 2
  $PY -I pie/eval.py "$n" --set "$SET" --jobs 14 --extra "$x" --render-only > /dev/null 2>&1
  $PY -I pie/gop.py "px-$n" --run "$n" --set "$SET" --threads 14 2>&1 | grep -v -i warn | tail -2
done
