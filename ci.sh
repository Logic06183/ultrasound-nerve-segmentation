#!/usr/bin/env bash
# Local stand-in for .github/workflows/tests.yml: runs the test suite on
# Python 3.10, 3.11 and 3.13 in fresh-ish uv virtualenvs under .venv-ci/.
# Usage: ./ci.sh            (all versions)
#        ./ci.sh 3.11       (one version)
set -euo pipefail
cd "$(dirname "$0")"
versions=("${@:-3.10 3.11 3.13}")
status=0
for v in ${versions[@]}; do
  d=".venv-ci/$v"
  [ -x "$d/bin/python" ] || uv venv -q -p "$v" "$d"
  uv pip install -q -p "$d/bin/python" -r requirements.txt jax
  echo "== Python $v =="
  if KERAS_BACKEND=jax "$d/bin/python" -m pytest -q; then
    echo "== Python $v: PASS =="
  else
    echo "== Python $v: FAIL =="; status=1
  fi
done
if [ -x .venv/bin/python ]; then
  echo "== nerveseg (PyTorch) =="
  if PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_nerveseg.py; then echo "== nerveseg: PASS =="
  else echo "== nerveseg: FAIL =="; status=1; fi
fi
exit $status
