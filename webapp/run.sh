#!/usr/bin/env bash
# Start Chart2Code: sets up a virtual environment on first run, trains the models if missing,
# then serves the site at http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  PY=""
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      PY="$candidate"; break
    fi
  done
  [ -n "$PY" ] || { echo "Python 3.10 or newer is required."; exit 1; }
  echo "Setting up the environment with $PY (first run only, a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

if [ ! -f model/models.joblib ]; then
  echo "Training the models (about 1 minute)..."
  .venv/bin/python train.py
fi

( sleep 4 && open "http://127.0.0.1:8000" 2>/dev/null || xdg-open "http://127.0.0.1:8000" 2>/dev/null || true ) &
echo "Chart2Code is starting at http://127.0.0.1:8000  (press Ctrl+C to stop)"
exec .venv/bin/python app.py
