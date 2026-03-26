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
