#!/usr/bin/env bash
set -euo pipefail

# Create venv (safe if it already exists)
python3 -m venv .venv

# Install/upgrade tools and deps inside the venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements.txt

# Ensure runner scripts are executable (Linux/macOS)
if ls run_*.sh >/dev/null 2>&1; then
  chmod +x run_*.sh || true
fi

echo "Venv ready. Activate with: source .venv/bin/activate"
echo "You can run: ./run_numeric_main.sh (or other run_*.sh)"
