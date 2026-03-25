# LitePicker — Claude Code Project Instructions

## What This Project Is

litepicker is a lightweight Python FastAPI proxy that sits between nanobot and LiteLLM. Its sole job: classify each incoming OpenAI-compatible request, rewrite the `model` field to the appropriate LiteLLM alias, and forward it. Classification uses a port of ClawRouter's MIT-licensed scoring algorithm (pure arithmetic, no ML, <1ms).

```
nanobot (port any)  →  litepicker (port 8765)  →  LiteLLM (port 4000)  →  providers
                         classify + rewrite            route + track
```

## Architecture

**Four files:**
- `main.py` — FastAPI app: receive, classify, rewrite `body["model"]`, proxy to LiteLLM
- `classifier.py` — ported ClawRouter scoring (14 dimensions → sigmoid → tier) + `TIER_TO_MODEL` map
- `logger.py` — async Postgres logging of classification decisions (join with LiteLLM logs on `request_id`)
- `config.py` — pydantic-settings, env-driven

**Key separation:** litepicker does NOT import litellm. It only talks to LiteLLM proxy over HTTP. This is intentional — isolation from supply chain issues.

## Model Tier Map

| Tier | LiteLLM Alias | Actual Model |
|---|---|---|
| `heartbeat` | `ollama-default` | local Ollama |
| `simple` | `deepseek-local` | DeepSeek via Ollama |
| `medium` | `haiku-4-5` | claude-haiku-4-5 |
| `complex` | `sonnet-4-6` | claude-sonnet-4-6 |
| `reasoning` | `sonnet-reasoning` | claude-sonnet-4-6 (effort:high) |
| `local` | `r1-local` | DeepSeek R1 8B via Ollama |

Fallback chains live in LiteLLM's YAML, not here.

## Classifier Algorithm

ClawRouter port — read source before modifying:
```bash
git clone https://github.com/BlockRunAI/ClawRouter.git /tmp/clawrouter
# Key files: /tmp/clawrouter/src/router/rules.ts  /tmp/clawrouter/src/router/index.ts
```

Flow:
1. Score last 3 messages across 14 dimensions (most-recent gets 2x weight)
2. Weighted sum → sigmoid calibration → confidence [0.0, 1.0]
3. Hard override: 2+ reasoning markers → REASONING (confidence=0.97)
4. Confidence thresholds: <0.30=simple, <0.50=medium, <0.70=complex, else=reasoning
5. Bias correction: COMPLEX→MEDIUM if confidence < 0.80 (nanobot's large system prompt inflates scores)

`reasoning_depth` dimension only applies to user-turn messages — system prompt boilerplate must not inflate it.

## Environment

- Conda env: `openclaw`
- Port: 8765 (litepicker), 4000 (LiteLLM)
- DB: Postgres — same DB LiteLLM uses, table `litepicker_classifications`
- Systemd unit: `~/.config/systemd/user/litepicker.service`

## LiteLLM Safety — CRITICAL

**Safe version: litellm==1.82.6** — versions 1.82.7 and 1.82.8 were compromised (credential theft on import). Pin in requirements.txt. litepicker doesn't import litellm directly, but check before any pip operations in openclaw env:

```bash
pip show litellm | grep Version
find ~/.local -name "litellm_init.pth" 2>/dev/null
find / -name "sysmon.py" 2>/dev/null
# If either find prints anything: rotate ALL API keys immediately
```

## Development Workflow

```bash
conda activate openclaw
cd ~/litepicker

# Quick classifier sanity check
python -c "from classifier import classify; print(classify([{'role':'user','content':'prove the halting problem'}]))"
# Expected: ('reasoning', 'sonnet-reasoning', 0.97)

# Run dev server
uvicorn main:app --reload --port 8765

# Smoke test
curl -X POST localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/auto","messages":[{"role":"user","content":"what time is it"}]}'
# Verify LiteLLM receives model=deepseek-local, not openai/auto

# Check Postgres
psql $POSTGRES_URL -c "SELECT * FROM litepicker_classifications LIMIT 5;"
```

## nanobot Integration

nanobot config points at litepicker, not LiteLLM directly:
```json
{
  "providers": { "openai": { "apiKey": "litepicker-passthrough", "baseUrl": "http://localhost:8765/v1" } },
  "agents": { "defaults": { "model": "openai/auto" } }
}
```
`openai/auto` is always overwritten before forwarding — it never reaches a provider.

## Postgres Join Query (Self-Learning Signal)

```sql
SELECT lp.request_id, lp.tier, lp.model_alias, lp.confidence,
       ll.model AS actual_model_used, ll.status, ll.response_time_ms, ll.total_cost
FROM litepicker_classifications lp
LEFT JOIN litellm_spend_logs ll ON ll.request_id = lp.request_id
ORDER BY lp.created_at DESC;
```
Verify the LiteLLM table name with `\dt` in psql first.

## Open Questions (Resolve Before Coding)

1. **Keyword lists** — clone ClawRouter and verify `classifier.py` keyword sets match `src/router/rules.ts` exactly
2. **LiteLLM spend log table name** — run `\dt` in psql to confirm
3. **Heartbeat detection** — nanobot fires heartbeats from `HEARTBEAT.md` every 30min; consider detecting known system prompt pattern to hard-route to `ollama-default` before classifier runs
4. **Bandit weight updates** — `POST /admin/weights` endpoint to update `DIMENSION_WEIGHTS` in-place without restart

## Key Design Principles

- litepicker owns: classification, tier→alias mapping, classification logging
- LiteLLM owns: provider routing, fallbacks, retries, cost tracking, callbacks
- Never add provider auth logic, fallback chains, or cost math to litepicker
- `DIMENSION_WEIGHTS` dict is intentionally mutable at runtime for bandit updates
- Silent fallbacks in LiteLLM corrupt training data — ensure LiteLLM raises after exhausting fallbacks
