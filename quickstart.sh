#!/usr/bin/env bash
# quickstart.sh — start Postgres, LiteLLM, and litepicker in order
set -euo pipefail

LITEPICKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITELLM_CONFIG="$HOME/.config/litellm/config.yaml"
CONDA_ENV="PAI"
LITEPICKER_PORT=8765
LITELLM_PORT=4000
POSTGRES_URL="postgresql://localhost/litellm"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

# ── 1. Postgres ───────────────────────────────────────────────────────────────
echo "==> Postgres"
if pg_isready -q 2>/dev/null; then
    ok "already running"
else
    warn "starting..."
    sudo systemctl start postgresql
    sleep 2
    pg_isready -q || die "Postgres failed to start"
    ok "started"
fi

# ensure litellm DB exists
psql "$POSTGRES_URL" -c '' 2>/dev/null || {
    warn "database 'litellm' not found, creating..."
    createdb litellm
    ok "database created"
}

# apply litepicker schema (idempotent — uses CREATE TABLE IF NOT EXISTS)
psql "$POSTGRES_URL" -f "$LITEPICKER_DIR/schema.sql" -q
ok "schema applied"

# ── 2. LiteLLM ────────────────────────────────────────────────────────────────
echo "==> LiteLLM"
litellm_up() { curl -so /dev/null -w "%{http_code}" "http://localhost:${LITELLM_PORT}/health" 2>/dev/null | grep -qE '^[24]'; }

if litellm_up; then
    ok "already running on :${LITELLM_PORT}"
else
    warn "starting in background..."
    [[ -f "$LITELLM_CONFIG" ]] || die "LiteLLM config not found at $LITELLM_CONFIG"
    conda run -n "$CONDA_ENV" litellm --config "$LITELLM_CONFIG" \
        > /tmp/litellm.log 2>&1 &
    LITELLM_PID=$!
    echo "  pid $LITELLM_PID — logs: /tmp/litellm.log"

    # wait up to 30s for LiteLLM to be ready
    for i in $(seq 1 30); do
        sleep 1
        if litellm_up; then
            ok "ready on :${LITELLM_PORT}"
            break
        fi
        [[ $i -eq 30 ]] && die "LiteLLM didn't respond after 30s — check /tmp/litellm.log"
    done
fi

# ── 3. litepicker ─────────────────────────────────────────────────────────────
echo "==> litepicker"
if curl -sf "http://localhost:${LITEPICKER_PORT}/health" >/dev/null 2>&1; then
    ok "already running on :${LITEPICKER_PORT}"
else
    warn "starting in background..."
    conda run -n "$CONDA_ENV" \
        uvicorn main:app --host 127.0.0.1 --port "$LITEPICKER_PORT" \
        --app-dir "$LITEPICKER_DIR" \
        > /tmp/litepicker.log 2>&1 &
    LITEPICKER_PID=$!
    echo "  pid $LITEPICKER_PID — logs: /tmp/litepicker.log"

    for i in $(seq 1 10); do
        sleep 1
        if curl -sf "http://localhost:${LITEPICKER_PORT}/health" >/dev/null 2>&1; then
            ok "ready on :${LITEPICKER_PORT}"
            break
        fi
        [[ $i -eq 10 ]] && die "litepicker didn't respond after 10s — check /tmp/litepicker.log"
    done
fi

# ── 4. smoke test ─────────────────────────────────────────────────────────────
echo "==> smoke test"
STATUS=$(curl -so /dev/null -w "%{http_code}" -X POST "http://localhost:${LITEPICKER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-local" \
    -d '{"model":"openai/auto","messages":[{"role":"user","content":"what time is it"}]}' \
    2>/dev/null)
[[ "$STATUS" =~ ^[245] ]] || die "smoke test got unexpected HTTP $STATUS — check /tmp/litepicker.log"
ok "request forwarded (HTTP $STATUS)"

echo ""
echo -e "${GREEN}All systems up.${NC}"
echo "  litepicker : http://localhost:${LITEPICKER_PORT}"
echo "  LiteLLM    : http://localhost:${LITELLM_PORT}"
echo "  logs       : /tmp/litepicker.log  /tmp/litellm.log"
