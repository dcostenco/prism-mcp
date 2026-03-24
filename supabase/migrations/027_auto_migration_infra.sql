-- ═══════════════════════════════════════════════════════════════════
-- Migration 027: Auto-Migration Infrastructure
--
-- Creates the scaffolding for automatic schema migrations on server
-- startup. After this migration, future DDL changes are applied via
-- the prism_apply_ddl() RPC — no manual SQL execution needed.
--
-- Components:
--   1. prism_schema_versions  — tracks which migrations are applied
--   2. prism_apply_ddl()      — SECURITY DEFINER function that
--                                executes DDL and records versions
-- ═══════════════════════════════════════════════════════════════════

-- ─── 1. Migration Tracking Table ─────────────────────────────────

CREATE TABLE IF NOT EXISTS prism_schema_versions (
  version   INTEGER PRIMARY KEY,
  name      TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE prism_schema_versions ENABLE ROW LEVEL SECURITY;

-- Only service_role can write; anon/authenticated can read
CREATE POLICY "Service role full access"
  ON prism_schema_versions
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated read access"
  ON prism_schema_versions
  FOR SELECT
  TO authenticated, anon
  USING (true);


-- ─── 2. DDL Execution Function ───────────────────────────────────

CREATE OR REPLACE FUNCTION prism_apply_ddl(
  p_version  INTEGER,
  p_name     TEXT,
  p_sql      TEXT
)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER          -- Runs with owner (postgres) privileges
SET search_path = public  -- Prevent search_path hijacking
AS $$
DECLARE
  v_count INTEGER;
BEGIN
  -- 1. Idempotency check: skip if already applied
  SELECT count(*) INTO v_count
    FROM prism_schema_versions
   WHERE version = p_version;

  IF v_count > 0 THEN
    RETURN json_build_object(
      'status',  'already_applied',
      'version', p_version
    );
  END IF;

  -- 2. Execute the DDL statement(s)
  EXECUTE p_sql;

  -- 3. Record success
  INSERT INTO prism_schema_versions (version, name)
  VALUES (p_version, p_name);

  RETURN json_build_object(
    'status',  'applied',
    'version', p_version
  );

EXCEPTION WHEN OTHERS THEN
  -- Surface the error clearly so the MCP server can log it
  RAISE EXCEPTION 'prism_apply_ddl migration % (%) failed: %',
    p_version, p_name, SQLERRM;
END;
$$;


-- ─── 3. Seed Previously Applied Migrations ──────────────────────
-- Mark all historical migrations as already applied so the
-- auto-migration runner doesn't try to re-apply them.

INSERT INTO prism_schema_versions (version, name) VALUES
  (15, 'initial_schema'),
  (16, 'session_improvements'),
  (17, 'search_and_analytics'),
  (18, 'semantic_search'),
  (19, 'compaction'),
  (20, 'time_travel'),
  (21, 'dashboard_settings'),
  (22, 'legacy_cleanup'),
  (23, 'handoff_history'),
  (24, 'agent_hivemind'),
  (25, 'fix_handoff_constraint'),
  (26, 'active_behavioral_memory'),
  (27, 'auto_migration_infra')
ON CONFLICT (version) DO NOTHING;
