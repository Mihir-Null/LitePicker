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
