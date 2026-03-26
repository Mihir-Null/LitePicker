#!/usr/bin/env bash
# shutdown.sh — stop litepicker and LiteLLM
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }

kill_port() {
    local port=$1 name=$2
    local pids
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        ok "$name stopped (port $port)"
    else
        warn "$name was not running"
    fi
}

echo "==> litepicker"
kill_port 8765 "litepicker"

echo "==> LiteLLM"
kill_port 4000 "LiteLLM"

echo ""
ok "done"
