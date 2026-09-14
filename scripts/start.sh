#!/usr/bin/env bash
# Starts XRouter in the background, writing logs to data/xrouter.log and its
# PID to data/xrouter.pid (used by `xrouter stop` / `xrouter status`).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data
PORT="${XROUTER_PORT:-20128}"

if [ -f data/xrouter.pid ] && kill -0 "$(cat data/xrouter.pid)" 2>/dev/null; then
  echo "XRouter already running (pid $(cat data/xrouter.pid))"
  exit 0
fi

PYTHON_BIN="python3"
if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
fi

nohup "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  > data/xrouter.log 2>&1 &
echo $! > data/xrouter.pid
echo "XRouter starting on http://localhost:${PORT} (pid $(cat data/xrouter.pid)), logs: data/xrouter.log"
