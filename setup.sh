#!/usr/bin/env bash
# One-time setup: creates a private Python environment in .venv and installs the libraries.
# Usage:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PY="$c"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9 or newer is required. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

echo "Using $("$PY" --version)"
"$PY" -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
chmod +x trader 2>/dev/null || true

echo
echo "Setup complete. Next:"
echo "  ./trader demo                      # offline demo on a synthetic market"
echo "  ./trader backtest                  # download real history and backtest"
echo "  ./trader run --capital 100000      # live recommendations + autonomous paper trading"
