# litepicker

Lightweight FastAPI proxy between nanobot and LiteLLM. Classifies each incoming OpenAI-compatible request, rewrites the `model` field to the appropriate LiteLLM alias, and forwards it.

```
nanobot  →  litepicker :8765  →  LiteLLM :4000  →  providers
               classify + rewrite       route + track
```

## Model Tiers

| Tier | LiteLLM Alias | Model |
|---|---|---|
| `heartbeat` | `heart-model` | local Ollama |
| `simple` | `simple-model` | DeepSeek via Ollama |
| `medium` | `medium-model` | claude-haiku-4-5 |
| `complex` | `complex-model` | claude-sonnet-4-6 |
| `reasoning` | `reasoning-model` | claude-sonnet-4-6 (effort:high) |
| `local` | `local-model` | DeepSeek R1 8B via Ollama |

---

## Setup & Deployment

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — set LITELLM_URL and POSTGRES_URL
```

### 2. Create the classifications table

```bash
psql $POSTGRES_URL -f schema.sql
```

### 3. Verify classifier keyword sets (one-time)

View ClawRouter source online (`src/router/rules.ts`) and confirm these sets in `classifier.py` match:
`REASONING_KEYWORDS`, `COMPLEX_KEYWORDS`, `TOOL_USE_KEYWORDS`, `MATH_KEYWORDS`, `CODE_KEYWORDS`

### 4. Start litepicker

**Recommended — systemd user service:**

```bash
cp systemd/litepicker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now litepicker
systemctl --user status litepicker
```

**Or manually:**

```bash
conda activate PAI
uvicorn main:app --host 127.0.0.1 --port 8765
```

### 5. Smoke test

LiteLLM must be running on port 4000.

```bash
curl -X POST localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/auto","messages":[{"role":"user","content":"what time is it"}]}'
# Verify LiteLLM received model=deepseek-local, not openai/auto
```

### 6. Wire nanobot

In nanobot's config:

```json
{
  "providers": {
    "openai": {
      "apiKey": "litepicker-passthrough",
      "baseUrl": "http://localhost:8765/v1"
    }
  },
  "agents": {
    "defaults": { "model": "openai/auto" }
  }
}
```

`openai/auto` is always overwritten by the classifier — it never reaches a provider.

### 7. Verify the join (after traffic flows)

```bash
# Confirm LiteLLM spend log table name
psql $POSTGRES_URL -c '\dt'

# Join classification decisions with LiteLLM cost/latency data
psql $POSTGRES_URL -c "
SELECT lp.request_id, lp.tier, lp.confidence, ll.response_time_ms, ll.total_cost
FROM litepicker_classifications lp
LEFT JOIN litellm_spend_logs ll ON ll.request_id = lp.request_id
ORDER BY lp.created_at DESC LIMIT 10;"
```
