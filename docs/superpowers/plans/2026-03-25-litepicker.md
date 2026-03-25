# LitePicker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build litepicker — a FastAPI proxy that classifies OpenAI-compatible requests using a ported ClawRouter scoring algorithm, rewrites the `model` field to the correct LiteLLM alias, and forwards to LiteLLM on port 4000.

**Architecture:** Single-process Python app with four modules in strict dependency order: `config.py` (pydantic-settings singleton) → `classifier.py` (14-dimension ClawRouter port, pure arithmetic) and `logger.py` (async Postgres via asyncpg) → `main.py` (FastAPI + lifespan + two routes). Shared `httpx.AsyncClient` via FastAPI lifespan for connection pooling. litellm is NOT imported — litepicker talks to the LiteLLM proxy over HTTP only.

**Tech Stack:** Python 3.11+ (conda env `openclaw`), FastAPI, uvicorn, httpx, asyncpg, pydantic-settings, pytest

---

### Task 1: Project bootstrap

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
asyncpg>=0.29.0
pydantic-settings>=2.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Write .env.example**

```
# litepicker environment configuration
# Copy to .env and edit values

LITELLM_URL=http://localhost:4000
LITEPICKER_PORT=8765
POSTGRES_URL=postgresql://localhost/litellm
LOG_CLASSIFICATIONS=true
DEFAULT_TIER=medium
```

- [ ] **Step 3: Create tests directory**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
conda activate openclaw
pip install -r requirements.txt
```

Expected: all packages install without errors. Verify `litellm` is NOT in requirements and not being installed as a transitive dep.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example tests/__init__.py
git commit -m "chore: bootstrap project structure and dependencies"
```

---

### Task 2: config.py

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    litellm_url: str = "http://localhost:4000"
    litepicker_port: int = 8765
    postgres_url: str = "postgresql://localhost/litellm"
    log_classifications: bool = True
    default_tier: str = "medium"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
conda activate openclaw
python -c "from config import settings; print(settings.litellm_url)"
```

Expected: `http://localhost:4000`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config settings module"
```

---

### Task 3: Classifier tests — red phase

**Files:**
- Create: `tests/test_classifier.py`

Write all 8 tests now. They will all fail with `ImportError` because `classifier.py` doesn't exist yet. That's correct — this is the TDD red phase.

- [ ] **Step 1: Write tests/test_classifier.py**

```python
from classifier import classify, TIER_TO_MODEL, SIMPLE_BOUNDARY, MEDIUM_BOUNDARY, COMPLEX_BOUNDARY
from config import settings


def test_simple_factual():
    """Short factual query should route to the cheapest tier."""
    messages = [{"role": "user", "content": "what time is it"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "simple"


def test_medium_request():
    """Moderate coding request should land in medium tier with confidence [0.30, 0.50)."""
    messages = [{"role": "user", "content": "debug and refactor this async function to use dependency injection, returning a structured JSON response"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "medium"
    assert SIMPLE_BOUNDARY <= confidence < MEDIUM_BOUNDARY, (
        f"Expected confidence in [{SIMPLE_BOUNDARY}, {MEDIUM_BOUNDARY}), got {confidence:.4f}"
    )


def test_reasoning_hard_override():
    """2+ reasoning markers always force reasoning tier at confidence 0.97."""
    messages = [{"role": "user", "content": "prove by induction that this theorem holds using contradiction"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "reasoning"
    assert confidence == 0.97


def test_empty_messages():
    """Empty message list returns the configured default tier."""
    tier, model_alias, confidence = classify([])
    assert tier == settings.default_tier


def test_tool_use_not_simple():
    """Requests with tools present should not route to the simple tier."""
    messages = [{"role": "user", "content": "debug and refactor this async function to use dependency injection, returning a structured JSON response"}]
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    tier, model_alias, confidence = classify(messages, tools)
    assert tier != "simple"


def test_bias_correction():
    """Messages scoring in the 'complex' confidence range [0.50, 0.70) are demoted to medium.

    The 'complex' tier is intentionally suppressed in v1. nanobot's large system prompt
    inflates scores. The bias correction threshold (0.80) is always true when tier == 'complex'
    (which only fires for confidence < 0.70 < 0.80), making 'complex' unreachable in v1.
    """
    messages = [{"role": "user", "content": "prove that implementing a neural network algorithm is efficient"}]
    tier, model_alias, confidence = classify(messages)
    # First: confirm the message actually scored in the 'complex' range
    assert MEDIUM_BOUNDARY <= confidence < COMPLEX_BOUNDARY, (
        f"Test input miscalibrated — expected confidence in [{MEDIUM_BOUNDARY}, {COMPLEX_BOUNDARY}), "
        f"got {confidence:.4f}. Adjust the message content to score in this range."
    )
    # Then: confirm bias correction demoted it to medium
    assert tier == "medium"


def test_returns_valid_model_alias():
    """Every returned model_alias must be a value in TIER_TO_MODEL."""
    messages = [{"role": "user", "content": "hello world"}]
    tier, model_alias, confidence = classify(messages)
    assert model_alias in TIER_TO_MODEL.values()


def test_confidence_bounds():
    """Confidence must always be in [0.0, 1.0] regardless of input."""
    inputs = [
        [{"role": "user", "content": "what is 2+2"}],
        [{"role": "user", "content": "refactor this function"}],
        [{"role": "user", "content": "prove by induction theorem contradiction"}],
        [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "help me write code"},
        ],
        [],
    ]
    for messages in inputs:
        tier, model_alias, confidence = classify(messages)
        assert 0.0 <= confidence <= 1.0, (
            f"confidence {confidence:.4f} out of bounds for input: {messages}"
        )
```

- [ ] **Step 2: Run tests — confirm they fail with ImportError**

```bash
conda activate openclaw
cd ~/litepicker
pytest tests/test_classifier.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'classifier'`

This is correct — `classifier.py` doesn't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_classifier.py
git commit -m "test: add classifier unit tests (all failing — red phase)"
```

---

### Task 4: Classifier — constants and keyword sets

**Files:**
- Create: `classifier.py`

Write constants only. No functions yet. Tests will still fail with `ImportError` on `classify` — still in the red phase.

- [ ] **Step 1: Write classifier.py with constants**

```python
import math
import re

from config import settings

# ── Tier → LiteLLM model alias ─────────────────────────────────────────────
# Must match model_list aliases in your LiteLLM config exactly.
# heartbeat and local tiers exist here for completeness; v1 classify() never
# returns them — reserved for future heartbeat detection and explicit local routing.
TIER_TO_MODEL: dict[str, str] = {
    "heartbeat": "ollama-default",
    "simple":    "deepseek-local",
    "medium":    "haiku-4-5",
    "complex":   "sonnet-4-6",
    "reasoning": "sonnet-reasoning",
    "local":     "r1-local",
}

# ── Dimension weights ───────────────────────────────────────────────────────
# Ported from ClawRouter rules.ts. Weights sum to 0.95 (factual_lookup is -0.05).
# These dicts are intentionally mutable — a contextual bandit can update them
# at runtime once outcome data accumulates.
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
    "factual_lookup":         -0.05,  # negative: pulls toward cheaper tier
    "message_length":          0.01,
    "tool_use_required":       0.01,
    "language_mix":            0.01,
}

# ── Tier boundaries ─────────────────────────────────────────────────────────
SIMPLE_BOUNDARY: float = 0.30   # confidence < this → simple
MEDIUM_BOUNDARY: float = 0.50   # confidence < this → medium
COMPLEX_BOUNDARY: float = 0.70  # confidence < this → complex; >= this → reasoning
REASONING_MARKERS_THRESHOLD: int = 2

# ── Sigmoid parameters ──────────────────────────────────────────────────────
SIGMOID_K: float = 8.0
SIGMOID_MIDPOINT: float = 0.45

# ── Keyword sets ─────────────────────────────────────────────────────────────
# Matching is substring-based and case-insensitive.
# IMPORTANT: verify these against /tmp/clawrouter/src/router/rules.ts before
# production — the keyword lists may have been updated upstream.
REASONING_KEYWORDS: set[str] = {
    "prove", "proof", "theorem", "lemma", "derive", "formal verification",
    "type theory", "induction", "contradiction", "axiom", "corollary",
    "if and only if", "necessary and sufficient", "qed",
}
CODE_KEYWORDS: set[str] = {
    "function", "class", "async", "await", "import", "export", "def ",
    "return", "const ", "let ", "var ", "interface", "struct", "trait",
    "implement", "refactor", "debug", "optimize", "algorithm", "complexity",
    "big o", "recursion", "dependency injection", "design pattern",
}
DOMAIN_KEYWORDS: set[str] = {
    "quantum", "thermodynamics", "stochastic", "differential equation",
    "machine learning", "neural network", "transformer", "attention mechanism",
    "distributed system", "consensus", "byzantine", "cryptography", "zk proof",
    "pharmacology", "jurisprudence", "epistemology", "ontology",
}
FACTUAL_KEYWORDS: set[str] = {
    "what is", "who is", "when was", "where is", "how many", "what year",
    "capital of", "population of", "translate", "definition of",
    "convert", "time in", "weather in",
}
MULTI_STEP_KEYWORDS: set[str] = {
    "first", "then", "next", "finally", "step by step", "step-by-step",
    "walk me through", "sequence", "pipeline", "workflow", "plan",
}
```

- [ ] **Step 2: Run tests — error should change from ModuleNotFoundError to ImportError on classify**

```bash
conda activate openclaw && cd ~/litepicker
pytest tests/test_classifier.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'classify' from 'classifier'`

The module loads now, but `classify` doesn't exist yet — progress.

---

### Task 5: Classifier — helper functions

**Files:**
- Modify: `classifier.py` (append helpers after the keyword sets)

Tests still fail after this step — `classify()` not yet defined.

- [ ] **Step 1: Append helper functions to classifier.py**

```python
# ── Internal helpers ─────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * (x - SIGMOID_MIDPOINT)))


def _keyword_score(text: str, keywords: set[str]) -> float:
    """Fraction of keywords present in text, capped at 1.0. Case-insensitive."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords) * 0.1, 1.0), 1.0)


def _avg_word_length(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def _score_dimensions(text: str, is_user_turn: bool) -> dict[str, float]:
    """Score one message across all 14 dimensions. All values in [0.0, 1.0].

    reasoning_depth is 0.0 for non-user turns — system prompt boilerplate
    (e.g. 'think step by step') must not inflate this score.
    All regex patterns use re.IGNORECASE.
    """
    word_count = len(text.split())
    return {
        "vocabulary_complexity":  min(_avg_word_length(text) / 10.0, 1.0),
        "reasoning_depth":        _keyword_score(text, REASONING_KEYWORDS) if is_user_turn else 0.0,
        "code_complexity":        _keyword_score(text, CODE_KEYWORDS),
        "domain_specificity":     _keyword_score(text, DOMAIN_KEYWORDS),
        "instruction_complexity": min(text.count(",") / 10.0, 1.0),
        "context_dependency":     1.0 if re.search(r"\b(it|this|that|they|them|those)\b", text, re.I) else 0.0,
        "output_format":          1.0 if re.search(r"\b(json|xml|table|markdown|format|structure)\b", text, re.I) else 0.0,
        "ambiguity":              1.0 if "?" in text and word_count < 10 else 0.0,
        "multi_step":             _keyword_score(text, MULTI_STEP_KEYWORDS),
        "creative_requirement":   1.0 if re.search(r"\b(write|compose|create|generate|imagine|story|poem)\b", text, re.I) else 0.0,
        "factual_lookup":         _keyword_score(text, FACTUAL_KEYWORDS),
        "message_length":         min(word_count / 500.0, 1.0),
        "tool_use_required":      0.0,   # set externally in classify() when tools are present
        "language_mix":           0.0,   # placeholder for multilingual support
    }
```

- [ ] **Step 2: Confirm tests still fail (classify not yet defined)**

```bash
pytest tests/test_classifier.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'classify' from 'classifier'`

---

### Task 6: classify() — green phase

**Files:**
- Modify: `classifier.py` (append classify() after the helpers)

After this step all 8 tests pass.

- [ ] **Step 1: Append classify() to classifier.py**

```python
# ── Public API ───────────────────────────────────────────────────────────────

def classify(messages: list[dict], tools: list | None = None) -> tuple[str, str, float]:
    """
    Classify a message list into a routing tier.

    Args:
        messages: OpenAI-format message list.
        tools:    Top-level 'tools' array from the request body, if present.
                  Passed by main.py via body.get("tools") — NOT a per-message field.

    Returns:
        (tier, model_alias, confidence)
        tier        — "simple" | "medium" | "reasoning" in v1 ("complex" is suppressed)
        model_alias — LiteLLM model_list alias string
        confidence  — sigmoid-calibrated score in [0.0, 1.0]

    Note on 'complex' tier: the bias correction threshold (confidence < 0.80) is
    intentionally always-true for the 'complex' confidence range (0.50–0.70), making
    'complex' unreachable in v1. Lower the threshold to ~0.65 once bandit outcome
    data confirms the classifier's complex-tier accuracy.
    """
    recent = messages[-3:]
    if not recent:
        tier = settings.default_tier
        return tier, TIER_TO_MODEL[tier], 0.5

    combined: dict[str, float] = {dim: 0.0 for dim in DIMENSION_WEIGHTS}
    reasoning_marker_count = 0
    total_weight = 0.0

    for i, msg in enumerate(recent):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI content-blocks format: [{"type": "text", "text": "..."}]
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if not content:
            continue

        # Most-recent message gets 2× weight; earlier messages get 1×
        msg_weight = 2.0 if i == len(recent) - 1 else 1.0
        total_weight += msg_weight

        scores = _score_dimensions(content, is_user_turn=(role == "user"))
        for dim, score in scores.items():
            combined[dim] += score * msg_weight

        if role == "user":
            content_lower = content.lower()
            # Per-message count, accumulated across turns. Same keyword in two
            # different messages counts twice. No cross-message deduplication.
            reasoning_marker_count += sum(
                1 for kw in REASONING_KEYWORDS if kw in content_lower
            )

    # tools is a top-level request body field. Overwrite (not add) — tool_use_required
    # is always 0.0 after the loop above, so the overwrite is safe.
    # After normalisation this becomes 1.0, contributing its weight to raw_score.
    if tools:
        combined["tool_use_required"] = total_weight

    if total_weight == 0:
        tier = settings.default_tier
        return tier, TIER_TO_MODEL[tier], 0.5

    normalised = {k: v / total_weight for k, v in combined.items()}
    raw_score = sum(normalised[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSION_WEIGHTS)
    confidence = _sigmoid(raw_score)

    # Hard override: 2+ reasoning markers always forces REASONING regardless of confidence
    if reasoning_marker_count >= REASONING_MARKERS_THRESHOLD:
        return "reasoning", TIER_TO_MODEL["reasoning"], 0.97

    # Map confidence to tier
    if confidence < SIMPLE_BOUNDARY:
        tier = "simple"
    elif confidence < MEDIUM_BOUNDARY:
        tier = "medium"
    elif confidence < COMPLEX_BOUNDARY:
        tier = "complex"
    else:
        tier = "reasoning"

    # Bias correction: suppress 'complex' until bandit data validates classifier accuracy.
    # COMPLEX_BOUNDARY (0.70) < 0.80, so this condition is always true when tier == 'complex'.
    # To re-enable the 'complex' tier, lower this threshold to ~0.65 in a future update.
    if tier == "complex" and confidence < 0.80:
        tier = "medium"

    return tier, TIER_TO_MODEL[tier], confidence
```

- [ ] **Step 2: Run all tests**

```bash
conda activate openclaw
cd ~/litepicker
pytest tests/test_classifier.py -v
```

Expected:
```
tests/test_classifier.py::test_simple_factual PASSED
tests/test_classifier.py::test_medium_request PASSED
tests/test_classifier.py::test_reasoning_hard_override PASSED
tests/test_classifier.py::test_empty_messages PASSED
tests/test_classifier.py::test_tool_use_not_simple PASSED
tests/test_classifier.py::test_bias_correction PASSED
tests/test_classifier.py::test_returns_valid_model_alias PASSED
tests/test_classifier.py::test_confidence_bounds PASSED

8 passed
```

**If `test_bias_correction` fails** with "Test input miscalibrated", the message doesn't score in [0.50, 0.70). Debug with:

```bash
python -c "
from classifier import classify, MEDIUM_BOUNDARY, COMPLEX_BOUNDARY
msg = [{'role':'user','content':'prove that implementing a neural network algorithm is efficient'}]
tier, alias, conf = classify(msg)
print(f'confidence={conf:.4f}  in_range={MEDIUM_BOUNDARY <= conf < COMPLEX_BOUNDARY}')
"
```

If confidence is outside [0.50, 0.70), find a message that scores there by adding or removing domain/code keywords, and update `test_bias_correction` accordingly. The test's assertion message tells you what range to aim for.

- [ ] **Step 3: Classifier smoke test from the command line**

```bash
python -c "
from classifier import classify

cases = [
    ('what time is it',                                                                          'simple'),
    ('debug and refactor this async function to use dependency injection, returning a structured JSON response', 'medium'),
    ('prove by induction that this theorem holds using contradiction',                            'reasoning'),
]
print(f'{\"Input\":<72}  Expected    Actual')
print('-' * 105)
for text, expected in cases:
    tier, alias, conf = classify([{'role':'user','content':text}])
    status = 'OK' if tier == expected else 'FAIL'
    print(f'{text[:70]:<72}  {expected:<10}  {tier:<10} {conf:.3f}  {status}')
"
```

Expected: all three rows show `OK`.

- [ ] **Step 4: Commit**

```bash
git add classifier.py
git commit -m "feat: implement classifier with 14-dimension scoring algorithm"
```

---

### Task 7: logger.py and schema.sql

**Files:**
- Create: `logger.py`
- Create: `schema.sql`

No tests for logger (out of scope). Write, verify import, confirm existing tests still pass.

- [ ] **Step 1: Write schema.sql**

```sql
-- Run once before first deploy:
--   psql $POSTGRES_URL -f schema.sql
-- Join with LiteLLM spend logs on request_id for bandit training signal:
--   SELECT lp.tier, lp.confidence, ll.status, ll.total_cost
--   FROM litepicker_classifications lp
--   LEFT JOIN <litellm_spend_table> ll ON ll.request_id = lp.request_id;
-- (Verify the LiteLLM table name with \dt in psql.)
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

- [ ] **Step 2: Write logger.py**

```python
import sys

import asyncpg
from datetime import datetime, timezone

from config import settings

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.postgres_url)
    return _pool


async def log_classification(
    request_id: str,
    tier: str,
    model_alias: str,
    confidence: float,
    message_count: int,
    preview: str,
) -> None:
    """
    Write a classification decision to Postgres.

    Silent on any failure — a DB outage must never block the proxy response.
    Join with LiteLLM's spend logs on request_id to get bandit training signal.
    """
    if not settings.log_classifications:
        return
    try:
        pool = await _get_pool()
        await pool.execute(
            """
            INSERT INTO litepicker_classifications
                (request_id, tier, model_alias, confidence,
                 message_count, preview, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (request_id) DO NOTHING
            """,
            request_id,
            tier,
            model_alias,
            confidence,
            message_count,
            preview[:200],
            datetime.now(timezone.utc),
        )
    except Exception as exc:
        print(f"[litepicker] log_classification failed: {exc}", file=sys.stderr)
```

- [ ] **Step 3: Verify logger.py imports cleanly**

```bash
conda activate openclaw
python -c "from logger import log_classification; print('ok')"
```

Expected: `ok` (pool is not created until the first async call — no DB needed for this check)

- [ ] **Step 4: Run existing tests — nothing should break**

```bash
pytest tests/test_classifier.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add logger.py schema.sql
git commit -m "feat: add async Postgres logger and schema DDL"
```

---

### Task 8: main.py

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
import asyncio
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from classifier import classify
from config import settings
from logger import log_classification

# Module-level task set prevents CPython GC from cancelling fire-and-forget tasks
# before they complete. Each task removes itself via done_callback on completion.
_bg_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shared httpx client — enables TCP connection pooling to LiteLLM.

    asyncpg pool is created lazily in logger.py on first request.
    """
    app.state.http = httpx.AsyncClient(timeout=120.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="litepicker", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")       # top-level body field — NOT on any message object
    request_id = str(uuid.uuid4())

    # 1. Classify — pure Python, synchronous, no network hop, <1ms
    tier, model_alias, confidence = classify(messages, tools)

    # 2. Rewrite model field — this is the entire point of litepicker
    body["model"] = model_alias

    # 3. Extract preview from last user message for classification log
    preview = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            preview = content
            break

    # 4. Fire-and-forget classification log — never blocks the proxy response.
    #    _bg_tasks holds a strong ref until the task completes, preventing GC cancellation.
    task = asyncio.create_task(
        log_classification(
            request_id=request_id,
            tier=tier,
            model_alias=model_alias,
            confidence=confidence,
            message_count=len(messages),
            preview=preview,
        )
    )
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)

    # 5. Build forward headers
    forward_headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Litepicker-Tier": tier,
        "X-Litepicker-Confidence": str(round(confidence, 3)),
    }
    if auth := request.headers.get("authorization"):
        forward_headers["Authorization"] = auth

    client: httpx.AsyncClient = request.app.state.http
    litellm_url = f"{settings.litellm_url}/v1/chat/completions"

    # 6. Forward to LiteLLM
    if body.get("stream", False):
        async def stream_gen():
            async with client.stream("POST", litellm_url, json=body, headers=forward_headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        # Always returns HTTP 200 — upstream status not available before the generator
        # starts. LiteLLM errors surface as malformed SSE chunks handled by nanobot's SDK.
        return StreamingResponse(stream_gen(), status_code=200, media_type="text/event-stream")

    resp = await client.post(litellm_url, json=body, headers=forward_headers)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )


@app.get("/health")
async def health():
    """Confirm litepicker is running and show the LiteLLM URL it proxies to."""
    return {"status": "ok", "litellm_url": settings.litellm_url}
```

- [ ] **Step 2: Verify main.py imports cleanly**

```bash
conda activate openclaw
python -c "from main import app; print('FastAPI app:', app.title)"
```

Expected: `FastAPI app: litepicker`

- [ ] **Step 3: Run existing tests — nothing should break**

```bash
pytest tests/test_classifier.py -v
```

Expected: `8 passed`

- [ ] **Step 4: Start server and hit /health**

```bash
conda activate openclaw
cd ~/litepicker
uvicorn main:app --port 8765 &
sleep 2
curl -s http://localhost:8765/health | python -m json.tool
pkill -f "uvicorn main:app"
```

Expected (assuming no `.env` file overrides `LITELLM_URL`):
```json
{
    "status": "ok",
    "litellm_url": "http://localhost:4000"
}
```

If `LITELLM_URL` is set in your `.env`, the value shown will be whatever you configured. Either way, `"status": "ok"` confirms the app is running.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add FastAPI proxy with lifespan, completions route, and health check"
```

---

### Task 9: Systemd service

**Files:**
- Create: `systemd/litepicker.service`

- [ ] **Step 1: Create the service file**

```bash
mkdir systemd
```

Write `systemd/litepicker.service`:

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

- [ ] **Step 2: Commit**

```bash
git add systemd/litepicker.service
git commit -m "chore: add systemd user service unit"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
conda activate openclaw
cd ~/litepicker
pytest tests/ -v
```

Expected: `8 passed`, no warnings, no import errors.

- [ ] **Step 2: Classifier smoke test**

```bash
python -c "
from classifier import classify

cases = [
    ('what time is it',
     'simple'),
    ('debug and refactor this async function to use dependency injection, returning a structured JSON response',
     'medium'),
    ('prove by induction that this theorem holds using contradiction',
     'reasoning'),
]
print(f'{\"Input\":<72}  Expected    Actual')
print('-' * 108)
for text, expected in cases:
    tier, alias, conf = classify([{'role':'user','content':text}])
    status = 'OK' if tier == expected else 'FAIL'
    print(f'{text[:70]:<72}  {expected:<10}  {tier:<10} {conf:.3f}  {status}')
"
```

Expected: all three rows show `OK`.

- [ ] **Step 3: Verify clean repo state**

```bash
git log --oneline
git status
```

Expected log (newest first):
```
chore: add systemd user service unit
feat: add FastAPI proxy with lifespan, completions route, and health check
feat: add async Postgres logger and schema DDL
feat: implement classifier with 14-dimension scoring algorithm
test: add classifier unit tests (all failing — red phase)
feat: add config settings module
chore: bootstrap project structure and dependencies
Add litepicker design spec and CLAUDE.md
```

`git status` should show clean working tree. `.env` and `__pycache__/` are expected to appear as untracked.

---

## Manual Steps Before Production

These steps are NOT automated — do them once before wiring nanobot:

**1. Create the Postgres table**
```bash
psql $POSTGRES_URL -f schema.sql
```

**2. Verify ClawRouter keyword lists**
```bash
git clone https://github.com/BlockRunAI/ClawRouter.git /tmp/clawrouter
# Diff classifier.py keyword sets against:
#   /tmp/clawrouter/src/router/rules.ts
# Update REASONING_KEYWORDS, CODE_KEYWORDS, etc. if upstream has additions.
```

**3. Confirm LiteLLM spend log table name**
```bash
psql $POSTGRES_URL -c '\dt'
# Find the actual table name and update the join query in CLAUDE.md
```

**4. Install and start the systemd service**
```bash
cp systemd/litepicker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now litepicker
systemctl --user status litepicker
```

**5. Wire nanobot** — update `~/.nanobot/config.json`:
```json
{
  "providers": {
    "openai": { "apiKey": "litepicker-passthrough", "baseUrl": "http://localhost:8765/v1" }
  },
  "agents": { "defaults": { "model": "openai/auto" } }
}
```

**6. LiteLLM safety check** — run before any `pip install` in the openclaw env:
```bash
pip show litellm | grep Version
# Safe: <= 1.82.6   Compromised: 1.82.7 or 1.82.8
find ~/.local -name "litellm_init.pth" 2>/dev/null
# If it prints anything: rotate ALL API keys immediately
```
