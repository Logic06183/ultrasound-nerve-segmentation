#!/usr/bin/env bash
# 5-fold patient-grouped training. Resumable: a finished fold (train_log.json) is skipped.
set -euo pipefail
cd "$(dirname "$0")"
for f in 0 1 2 3 4; do
  if [ -f runs/fold$f/train_log.json ]; then echo "fold $f already done"; continue; fi
  echo "=== fold $f start $(date +%T)"
  PYTHONPATH=. .venv/bin/python -m nerveseg.train --val-fold $f --epochs 15 --out runs/fold$f
  echo "=== fold $f done $(date +%T)"
done
echo "ALL FOLDS DONE $(date +%T)"
