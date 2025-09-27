-- === Extensions ===
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- === Minimal Users ===
CREATE TABLE IF NOT EXISTS users (
  user_id    TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- === Raw turns (live chat) ===
CREATE TABLE IF NOT EXISTS conv_turns (
  turn_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  conversation_id TEXT NOT NULL,
  turn_index      INT  NOT NULL,  -- 0,1,2...
  role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content         TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_turn_idx
  ON conv_turns (user_id, conversation_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_turns_user_conv_created
  ON conv_turns (user_id, conversation_id, created_at DESC);

-- === Row-per-window vector store ===
CREATE TABLE IF NOT EXISTS conv_windows (
  window_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  conversation_id  TEXT NOT NULL,
  start_index      INT NOT NULL,
  end_index        INT NOT NULL,
  turn_count       INT NOT NULL,
  text             TEXT NOT NULL,
  fts              tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text,''))) STORED,
  embedding        VECTOR(1536),
  test_group       INT NOT NULL DEFAULT 1,
  first_turn_at    TIMESTAMPTZ,
  last_turn_at     TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT now(),
  text_hash        TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_conv_windows_span
  ON conv_windows (user_id, conversation_id, start_index, end_index);

CREATE INDEX IF NOT EXISTS idx_conv_windows_user_conv
  ON conv_windows (user_id, conversation_id, last_turn_at DESC);

CREATE INDEX IF NOT EXISTS idx_conv_windows_emb_ivf
  ON conv_windows USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_conv_windows_fts_gin
  ON conv_windows USING GIN (fts);

-- Optional analytics
CREATE TABLE IF NOT EXISTS analytics_events (
  user_id     TEXT NOT NULL,
  test_group  INT  NOT NULL,
  event_name  TEXT NOT NULL,
  data        JSONB,
  ds          TIMESTAMPTZ DEFAULT now(),
  st          BIGINT
);
CREATE INDEX IF NOT EXISTS idx_ae_user_ds ON analytics_events (user_id, ds DESC);
CREATE INDEX IF NOT EXISTS idx_ae_event_ds ON analytics_events (event_name, ds DESC);
CREATE INDEX IF NOT EXISTS idx_ae_testgroup_ds ON analytics_events (test_group, ds DESC);
CREATE INDEX IF NOT EXISTS idx_ae_data_gin ON analytics_events USING GIN (data);

ANALYZE users;
ANALYZE conv_turns;
ANALYZE conv_windows;
ANALYZE analytics_events;
