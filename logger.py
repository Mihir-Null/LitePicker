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
