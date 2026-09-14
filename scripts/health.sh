#!/usr/bin/env bash
# Quick liveness check against a running XRouter instance.
set -euo pipefail
PORT="${XROUTER_PORT:-20128}"
curl -sS "http://localhost:${PORT}/health" | python3 -m json.tool
