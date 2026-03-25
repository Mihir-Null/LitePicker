# LitePicker Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Scope:** v1 — core proxy + classifier + logger + unit tests

---

## Overview

litepicker is a lightweight FastAPI proxy (port 8765) that sits between nanobot and LiteLLM (port 4000). It classifies each incoming OpenAI-compatible request using a port of ClawRouter's MIT-licensed scoring algorithm, rewrites the `model` field to the appropriate LiteLLM alias for the classified tier, and forwards the request. Everything else — provider routing, fallbacks, retries, cost tracking — stays in LiteLLM.

```
nanobot  →  litepicker (8765)  →  LiteLLM (4000)  →  providers
              classify + rewrite       route + track
```

litepicker does **not** import litellm. It communicates with the LiteLLM proxy over HTTP only. This is intentional isolation from supply chain risk.

---

## File Structure

```
~/litepicker/
  config.py              ← pydantic-settings singleton; no project imports
  classifier.py          ← imports config; pure arithmetic, no I/O
  logger.py              ← imports config; async Postgres via asyncpg
  main.py                ← imports all three; FastAPI app + lifespan
  schema.sql             ← DDL for litepicker_classifications table; run once manually
  tests/
    test_classifier.py   ← imports classifier only; pytest, no mocking
  systemd/
    litepicker.service
  requirements.txt
  .env.example
```

**Dependency graph (strict DAG):**
```
config  →  classifier  ┐
config  →  logger      ├→  main
                       ┘
tests/test_classifier  →  classifier
```

No cycles. No module-level side effects beyond `settings = Settings()` in `config.py`.

---

## Component Designs

### `config.py`

Pydantic-v2 `BaseSettings` with `SettingsConfigDict(env_file=".env")`. One module-level singleton: `settings = Settings()`.

| Field | Type | Default | Env var |
|---|---|---|---|
| `litellm_url` | `str` | `http://localhost:4000` | `LITELLM_URL` |
| `litepicker_port` | `int` | `8765` | `LITEPICKER_PORT` |
| `postgres_url` | `str` | `postgresql://localhost/litellm` | `POSTGRES_URL` |
| `log_classifications` | `bool` | `True` | `LOG_CLASSIFICATIONS` |
| `default_tier` | `str` | `"medium"` | `DEFAULT_TIER` |

All fields have defaults — litepicker starts without a `.env` file.

---

### `classifier.py`

Port of ClawRouter's `src/router/rules.ts` + `src/router/index.ts`. Verify keyword sets against upstream before finalising.

#### `TIER_TO_MODEL`

Canonical tier-to-alias mapping. Must match `model_list` aliases in the LiteLLM YAML exactly.

| Tier | LiteLLM Alias | Actual Model |
|---|---|---|
| `heartbeat` | `ollama-default` | local Ollama |
| `simple` | `deepseek-local` | DeepSeek via Ollama |
| `medium` | `haiku-4-5` | claude-haiku-4-5 |
| `complex` | `sonnet-4-6` | claude-sonnet-4-6 |
| `reasoning` | `sonnet-reasoning` | claude-sonnet-4-6 (effort:high) |
| `local` | `r1-local` | DeepSeek R1 8B via Ollama |

**Note:** In v1, `classify()` only ever returns one of: `simple`, `medium`, `complex`, `reasoning`, or `settings.default_tier`. The `heartbeat` and `local` tiers exist in `TIER_TO_MODEL` for completeness and future use (heartbeat detection, explicit local routing) but no code path in v1 selects them. They are present so `TIER_TO_MODEL` is the single source of truth for alias strings.

#### Module-level constants (all mutable — bandit can update at runtime)

- `TIER_TO_MODEL: dict[str, str]` — as above
- `DIMENSION_WEIGHTS: dict[str, float]` — 14 named dimensions (see below); weights sum to 0.95 (the negative `factual_lookup` weight of -0.05 is intentional — it pulls requests with simple factual patterns toward cheaper tiers)
- Keyword sets: `REASONING_KEYWORDS`, `CODE_KEYWORDS`, `DOMAIN_KEYWORDS`, `FACTUAL_KEYWORDS`, `MULTI_STEP_KEYWORDS`
- Sigmoid parameters: `SIGMOID_K = 8.0`, `SIGMOID_MIDPOINT = 0.45`
- Tier boundaries:
  - `SIMPLE_BOUNDARY = 0.30` — confidence below this → `simple`
  - `MEDIUM_BOUNDARY = 0.50` — confidence below this → `medium`
  - `COMPLEX_BOUNDARY = 0.70` — confidence below this → `complex`; at or above → `reasoning`
- `REASONING_MARKERS_THRESHOLD = 2`

#### 14 Dimensions

`_score_dimensions()` returns a `dict[str, float]` with exactly these keys:

| Dimension | Score method | Notes |
|---|---|---|
| `vocabulary_complexity` | `min(_avg_word_length(text) / 10.0, 1.0)` | |
| `reasoning_depth` | `_keyword_score(text, REASONING_KEYWORDS)` | **0.0 for non-user turns** |
| `code_complexity` | `_keyword_score(text, CODE_KEYWORDS)` | |
| `domain_specificity` | `_keyword_score(text, DOMAIN_KEYWORDS)` | |
| `instruction_complexity` | `min(text.count(",") / 10.0, 1.0)` | |
| `context_dependency` | `1.0` if pronoun regex matches, else `0.0` | regex: `\b(it\|this\|that\|they\|them\|those)\b` |
| `output_format` | `1.0` if format keyword matches, else `0.0` | regex: `\b(json\|xml\|table\|markdown\|format\|structure)\b` |
| `ambiguity` | `1.0` if `"?" in text and word_count < 10`, else `0.0` | |
| `multi_step` | `_keyword_score(text, MULTI_STEP_KEYWORDS)` | |
| `creative_requirement` | `1.0` if creative verb regex matches, else `0.0` | regex: `\b(write\|compose\|create\|generate\|imagine\|story\|poem)\b` |
| `factual_lookup` | `_keyword_score(text, FACTUAL_KEYWORDS)` | **negative weight (-0.05)** |
| `message_length` | `min(word_count / 500.0, 1.0)` | |
| `tool_use_required` | `0.0` by default; set externally | see algorithm step 4 |
| `language_mix` | `0.0` always in v1 | placeholder for multilingual support |

#### Dimension weights

```python
DIMENSION_WEIGHTS: dict[str, float] = {
    "vocabulary_complexity":   0.20,
    "reasoning_depth":         0.25,
    "code_complexity":         0.15,
    "domain_specificity":      0.10,
    "instruction_complexity":  0.10,
    "context_dependency":      0.05,
    "output_format":           0.05,
    "ambiguity":               0.03,
    "multi_step":              0.03,
    "creative_requirement":    0.01,
    "factual_lookup":         -0.05,
    "message_length":          0.01,
    "tool_use_required":       0.01,
    "language_mix":            0.01,
}
```

#### Internal helpers

- `_sigmoid(x: float) -> float` — `1.0 / (1.0 + exp(-SIGMOID_K * (x - SIGMOID_MIDPOINT)))`
- `_keyword_score(text: str, keywords: set[str]) -> float` — lowercases `text` internally before matching; `min(hits / max(len(keywords) * 0.1, 1.0), 1.0)`
- `_avg_word_length(text: str) -> float` — mean character count of words; returns `0.0` for empty text
- `_score_dimensions(text: str, is_user_turn: bool) -> dict[str, float]` — returns all 14 dimension scores for one message. All regex patterns use `re.IGNORECASE` (`re.I`). The caller passes the original content string; lowercasing for keyword matching is handled internally by `_keyword_score` and by explicit `.lower()` calls in regex-free dimensions.

#### Public API

```python
def classify(messages: list[dict], tools: list | None = None) -> tuple[str, str, float]:
    """
    Classify a message list into a routing tier.

    Args:
        messages: OpenAI-format message list.
        tools:    Top-level 'tools' array from the request body, if present.
                  Used to set the tool_use_required dimension.

    Returns:
        (tier, model_alias, confidence)
    """
```

`tools` is a top-level field on the OpenAI request body, not on any message object. `main.py` extracts it with `body.get("tools")` and passes it as the second argument to `classify()`.

#### Algorithm

1. Slice `messages[-3:]`. If empty, return `(default_tier, TIER_TO_MODEL[default_tier], 0.5)`.
2. Initialise `combined = {dim: 0.0 for dim in DIMENSION_WEIGHTS}`, `reasoning_marker_count = 0`, `total_weight = 0.0`.
3. Loop over recent messages with index `i`. Most-recent (i == len(recent) - 1) gets `msg_weight = 2.0`, all others get `1.0`. For each message:
   - Extract `role = msg.get("role", "")` and `content = msg.get("content", "")`.
   - If `content` is a list (content-blocks), join all `.get("text", "")` parts with a space.
   - Skip if resulting content string is empty; do not add to `total_weight`.
   - Call `scores = _score_dimensions(content, is_user_turn=(role == "user"))`.
   - Accumulate: `combined[dim] += scores[dim] * msg_weight` for all 14 dimensions.
   - If `role == "user"`: increment `reasoning_marker_count` by the number of `REASONING_KEYWORDS` entries present in `content.lower()` for this message (`sum(1 for kw in REASONING_KEYWORDS if kw in content_lower)`). This is per-message, accumulated across turns — the same keyword appearing in two different messages increments the count twice. No cross-message deduplication.
   - Add `msg_weight` to `total_weight`.
4. If `tools` is non-empty (list with at least one element): **assign** `combined["tool_use_required"] = total_weight`. This is an overwrite, not an addition — `tool_use_required` is always 0.0 after the loop (the dimension table shows it as "0.0 by default; set externally"), so the overwrite is safe. After normalisation in step 6 it becomes 1.0.
5. If `total_weight == 0`: return `(default_tier, TIER_TO_MODEL[default_tier], 0.5)`.
6. Normalise: `normalised = {k: v / total_weight for k, v in combined.items()}`.
7. `raw_score = sum(normalised[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSION_WEIGHTS)` → `confidence = _sigmoid(raw_score)`.
8. Hard override: `reasoning_marker_count >= REASONING_MARKERS_THRESHOLD` → return `("reasoning", TIER_TO_MODEL["reasoning"], 0.97)`.
9. Map confidence to tier:
   - `confidence < SIMPLE_BOUNDARY` → `"simple"`
   - `confidence < MEDIUM_BOUNDARY` → `"medium"`
   - `confidence < COMPLEX_BOUNDARY` → `"complex"`
   - `confidence >= COMPLEX_BOUNDARY` → `"reasoning"`
10. Bias correction: `tier == "complex" and confidence < 0.80` → demote to `"medium"`.
    **Note:** Because `"complex"` is only assigned when `0.50 <= confidence < 0.70`, and `0.70 < 0.80`, this condition is always true in v1. The `"complex"` tier is intentionally suppressed — nanobot's large system prompt inflates raw scores, making the classifier over-predict `"complex"`. The threshold of 0.80 acts as a sentinel: `"complex"` stays dormant until bandit outcome data confirms the classifier's `"complex"` predictions are accurate, at which point the threshold should be lowered to ~0.65 (or removed) in a future update.
11. Return `(tier, TIER_TO_MODEL[tier], confidence)`.

---

### `logger.py`

Async Postgres logging via `asyncpg`. Module-level `_pool = None`, lazy-initialised on first call.

**Public API:**
```python
async def log_classification(
    request_id: str,
    tier: str,
    model_alias: str,
    confidence: float,
    message_count: int,
    preview: str,
) -> None
```

- If `settings.log_classifications` is `False`, return immediately.
- Entire body wrapped in `try/except Exception` — any failure logs to stderr and returns silently. Never raises to caller.
- `preview` truncated to 200 chars before insert.
- `ON CONFLICT (request_id) DO NOTHING` — idempotent.

**No shutdown hook** — pool cleanup handled by process exit. litepicker's lifespan does not own the pool.

---

### `main.py`

FastAPI app with lifespan context manager.

**Lifespan:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=120.0)
    yield
    await app.state.http.aclose()

app = FastAPI(title="litepicker", lifespan=lifespan)
```

asyncpg pool is created lazily in `logger.py` on first request.

**Routes:**

`POST /v1/chat/completions` (hot path):
1. `body = await request.json()`
2. `messages = body.get("messages", [])`, `tools = body.get("tools")` (may be `None` or a list)
3. `tier, model_alias, confidence = classify(messages, tools)` — synchronous, <1ms, no I/O
4. `body["model"] = model_alias`
5. Extract `preview`: iterate `reversed(messages)`, find first `role == "user"`, take `content` as str or join content-block texts
6. Fire-and-forget DB log. A module-level `_bg_tasks: set[asyncio.Task] = set()` prevents GC cancellation:
   ```python
   task = asyncio.create_task(log_classification(...))
   _bg_tasks.add(task)
   task.add_done_callback(_bg_tasks.discard)
   ```
   DB write never blocks the proxy response. If the process is killed mid-task, that log row is lost — accepted risk.
7. Build forward headers:
   ```python
   headers = {
       "Content-Type": "application/json",
       "X-Request-ID": request_id,          # fresh uuid4 str
       "X-Litepicker-Tier": tier,
       "X-Litepicker-Confidence": str(round(confidence, 3)),
   }
   if auth := request.headers.get("authorization"):
       headers["Authorization"] = auth
   ```
8. `client = request.app.state.http`
9. If `body.get("stream", False)`:
   ```python
   async def stream_gen():
       async with client.stream(
           "POST",
           f"{settings.litellm_url}/v1/chat/completions",
           json=body,
           headers=headers,
       ) as resp:
           async for chunk in resp.aiter_bytes():
               yield chunk

   return StreamingResponse(stream_gen(), status_code=200, media_type="text/event-stream")
   ```
   **Note:** The streaming path always returns HTTP 200. Because `httpx.AsyncClient.stream()` opens the connection lazily inside the generator, the upstream status code is not available before the `StreamingResponse` is constructed. For v1, 200 is the correct default — LiteLLM errors during streaming will surface as malformed SSE chunks, which nanobot's openai SDK handles. Propagating the upstream status code for streaming is deferred.
10. Else (non-streaming):
    ```python
    resp = await client.post(
        f"{settings.litellm_url}/v1/chat/completions",
        json=body,
        headers=headers,
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )
    ```

`GET /health`:
```python
return {"status": "ok", "litellm_url": settings.litellm_url}
```

**`request_id` policy:** Always generate a fresh `uuid4()` string at the top of the handler. Never read from incoming request headers (v1).

---

### `tests/test_classifier.py`

Pytest. No fixtures. No mocking. Calls `classify()` directly with hand-crafted `messages` lists.

| Test | Input | Expected |
|---|---|---|
| `test_simple_factual` | `[{"role":"user","content":"what time is it"}]` | tier == `"simple"` |
| `test_medium_request` | `[{"role":"user","content":"refactor this function to use dependency injection and return a structured result"}]` | tier == `"medium"` AND `0.30 <= confidence < 0.50`. Assert both tier and confidence range to make the test self-validating against threshold changes. |
| `test_reasoning_hard_override` | `[{"role":"user","content":"prove by induction that this theorem holds using contradiction"}]` | tier == `"reasoning"`, confidence == 0.97 (contains "prove", "induction", "theorem", "contradiction" — 4 REASONING_KEYWORDS, well above threshold of 2) |
| `test_empty_messages` | `[]` | tier == `settings.default_tier` |
| `test_tool_use_not_simple` | `messages=[{"role":"user","content":"refactor this function to use dependency injection and return a structured result"}]`, `tools=[{"type":"function","function":{"name":"foo","parameters":{}}}]` | tier != `"simple"`. Uses the same content as `test_medium_request` (known to score above `SIMPLE_BOUNDARY`) to guarantee the message itself is non-simple; tools arg exercises the `tool_use_required` code path. |
| `test_bias_correction` | messages crafted to produce confidence in `[0.50, 0.70)` (i.e. would map to `"complex"` via step 9) | tier == `"medium"` — verifies the intentional suppression of the `"complex"` tier |
| `test_returns_valid_model_alias` | any input | `model_alias in TIER_TO_MODEL.values()` |
| `test_confidence_bounds` | variety of inputs | `0.0 <= confidence <= 1.0` for all |

For `test_tool_use_not_simple`: pass the tools list as the second arg to `classify(messages, tools)`.

For `test_bias_correction`: use a message containing several CODE_KEYWORDS and DOMAIN_KEYWORDS to push confidence into `[0.50, 0.70)` (the `"complex"` range per step 9). Assert returned tier is `"medium"`, confirming the intentional suppression. Because `"complex"` is always suppressed in v1, any input that would naturally score `"complex"` is a valid input for this test — inspect `confidence` directly if needed to confirm it landed in `[0.50, 0.70)`.

---

### `schema.sql`

```sql
CREATE TABLE IF NOT EXISTS litepicker_classifications (
    request_id     TEXT PRIMARY KEY,
    tier           TEXT NOT NULL,
    model_alias    TEXT NOT NULL,
    confidence     FLOAT,
    message_count  INT,
    preview        TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
```

Run once: `psql $POSTGRES_URL -f schema.sql`

---

### `systemd/litepicker.service`

```ini
[Unit]
Description=litepicker LiteLLM wrapper
After=network.target litellm.service
Wants=litellm.service

[Service]
Type=simple
WorkingDirectory=%h/litepicker
EnvironmentFile=%h/litepicker/.env
ExecStart=%h/miniconda3/envs/openclaw/bin/uvicorn main:app \
    --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Install: `cp systemd/litepicker.service ~/.config/systemd/user/ && systemctl --user enable --now litepicker`

---

### `requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
asyncpg>=0.29.0
pydantic-settings>=2.0.0
pytest>=8.0.0
```

litellm is **not** a dependency.

---

### `.env.example`

```
# litepicker environment configuration
# Copy to .env and edit values

LITELLM_URL=http://localhost:4000
LITEPICKER_PORT=8765
POSTGRES_URL=postgresql://localhost/litellm
LOG_CLASSIFICATIONS=true
DEFAULT_TIER=medium
```

---

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| v1 scope | Core only | Heartbeat detection and `/admin/weights` deferred |
| Postgres error handling | Silent fail | Logging must never block proxying |
| Test scope | Classifier unit tests only | Pure arithmetic, no I/O — easiest to test and highest risk if wrong |
| `request_id` | Always generate fresh uuid4 | Simple; nanobot doesn't send one in v1 |
| httpx client | Shared via lifespan | Connection pooling; two lines over per-request client |
| Logging | `asyncio.create_task` fire-and-forget | DB write must not add latency to proxy response |
| `tools` parameter | Second arg to `classify()` | `tools` is a top-level body field, not a message field; must be extracted in `main.py` and passed explicitly |
| `heartbeat`/`local` tiers | In `TIER_TO_MODEL`, not in classifier paths | Keeps the map as single source of truth; selected by future code |

---

## Out of Scope (v1)

- Heartbeat detection (hard-route nanobot heartbeats before classifier)
- `POST /admin/weights` endpoint for runtime bandit weight updates
- ClawRouter keyword verification (flagged as a manual step before first production deploy)
- LiteLLM spend log table name verification (flagged as a manual step)
- Integration/smoke tests
- README
