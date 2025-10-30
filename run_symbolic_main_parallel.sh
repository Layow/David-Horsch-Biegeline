#!/usr/bin/env bash
set -euo pipefail

if [ ! -x ./.venv/bin/python ]; then
  echo "Virtual environment not found. Run ./first-run.sh first."
  exit 1
fi

./.venv/bin/python symbolic_main_parallel.py "$@"

