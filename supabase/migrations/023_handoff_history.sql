-- ==============================================================================
-- MIGRATION 023: Create session_handoffs_history Table (Time Travel)
-- ==============================================================================
-- The v2.0 code (supabase.ts lines 224, 238) reads/writes to
-- session_handoffs_history for the memory_history and memory_checkout
-- tools. This table was defined in SQLite (sqlite.ts line 162) but
-- was never created in any Supabase migration.
-- ==============================================================================

-- ─── Table ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_handoffs_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',
    version INTEGER NOT NULL,
    snapshot JSONB NOT NULL,
    branch TEXT NOT NULL DEFAULT 'main',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Indexes ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_handoff_history_project
    ON session_handoffs_history(project, user_id);

CREATE INDEX IF NOT EXISTS idx_handoff_history_version
    ON session_handoffs_history(project, version);

-- ─── RLS (multi-tenant isolation) ────────────────────────────
-- Match the RLS pattern used in migration 020 for other tables.
ALTER TABLE session_handoffs_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own handoff history"
    ON session_handoffs_history
    FOR ALL
    USING (user_id = current_setting('request.jwt.claims', true)::json->>'sub'
           OR user_id = 'default')
    WITH CHECK (user_id = current_setting('request.jwt.claims', true)::json->>'sub'
                OR user_id = 'default');

-- Allow service_role full access (for server-side MCP calls)
CREATE POLICY "Service role full access on handoff history"
    ON session_handoffs_history
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ==============================================================================
-- MIGRATION COMPLETE
-- ==============================================================================

COMMENT ON TABLE session_handoffs_history IS
    'Point-in-time snapshots of session_handoffs for time travel. '
    'Created automatically by saveHandoff. Used by memory_history/memory_checkout tools.';
