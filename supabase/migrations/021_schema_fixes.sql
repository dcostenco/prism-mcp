-- ==============================================================================
-- MIGRATION 021: Schema Fixes for Prism MCP v2.0 Code Compatibility
-- ==============================================================================
-- Fixes mismatches between the migration-defined schema and the v2.0 code:
--   1. Adds missing `conversation_id` column to session_ledger
--   2. Adds missing `todos` column alias to session_ledger
--   3. Updates `search_knowledge` RPC to accept `p_user_id` parameter
-- ==============================================================================

-- ─── Fix 1: Add conversation_id column ──────────────────────────
-- The v2.0 code (supabase.ts line 56) inserts `conversation_id`,
-- but migration 015 only has `agent_name`. Add the column.
ALTER TABLE session_ledger
  ADD COLUMN IF NOT EXISTS conversation_id TEXT NOT NULL DEFAULT '';

-- ─── Fix 2: Add todos column ────────────────────────────────────
-- The v2.0 code inserts `todos` (array), but migration 015 uses `todo_next`.
-- Add `todos` as a proper column so PostgREST inserts work.
ALTER TABLE session_ledger
  ADD COLUMN IF NOT EXISTS todos TEXT[] DEFAULT '{}';

-- ─── Fix 3: Update search_knowledge with p_user_id ──────────────
-- The v2.0 code sends p_user_id but the current function (from 016)
-- only has 5 params. Drop old and create with 6 params.
DROP FUNCTION IF EXISTS search_knowledge(TEXT, TEXT[], TEXT, TEXT, INT);

CREATE OR REPLACE FUNCTION search_knowledge(
    p_project TEXT DEFAULT NULL,
    p_keywords TEXT[] DEFAULT '{}',
    p_category TEXT DEFAULT NULL,
    p_query_text TEXT DEFAULT NULL,
    p_limit INT DEFAULT 10,
    p_user_id TEXT DEFAULT 'default'
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    results JSONB;
    search_keywords TEXT[];
BEGIN
    search_keywords := p_keywords;

    IF p_category IS NOT NULL AND p_category != '' THEN
        search_keywords := search_keywords || ARRAY['cat:' || p_category];
    END IF;

    SELECT COALESCE(jsonb_agg(entry ORDER BY entry->>'relevance_score' DESC), '[]'::jsonb)
    INTO results
    FROM (
        -- Search session_ledger
        SELECT jsonb_build_object(
            'source', 'ledger',
            'project', sl.project,
            'summary', sl.summary,
            'keywords', to_jsonb(sl.keywords),
            'decisions', to_jsonb(sl.decisions),
            'files_changed', to_jsonb(sl.files_changed),
            'date', sl.session_date,
            'created_at', sl.created_at,
            'relevance_score', (
                CASE WHEN array_length(search_keywords, 1) > 0
                    THEN (SELECT COUNT(*) FROM unnest(sl.keywords) k WHERE k = ANY(search_keywords))
                    ELSE 0
                END
                +
                CASE WHEN p_query_text IS NOT NULL AND p_query_text != ''
                     AND to_tsvector('english', sl.summary) @@ plainto_tsquery('english', p_query_text)
                    THEN 5
                    ELSE 0
                END
            )
        ) AS entry
        FROM session_ledger sl
        WHERE
            sl.user_id = p_user_id
            AND (p_project IS NULL OR sl.project = p_project)
            AND sl.archived_at IS NULL
            AND (
                (array_length(search_keywords, 1) > 0 AND sl.keywords && search_keywords)
                OR
                (p_query_text IS NOT NULL AND p_query_text != ''
                 AND to_tsvector('english', sl.summary) @@ plainto_tsquery('english', p_query_text))
            )

        UNION ALL

        -- Search session_handoffs
        SELECT jsonb_build_object(
            'source', 'handoff',
            'project', sh.project,
            'summary', COALESCE(sh.last_summary, ''),
            'keywords', to_jsonb(sh.keywords),
            'decisions', to_jsonb(sh.active_decisions),
            'updated_at', sh.updated_at,
            'relevance_score', (
                CASE WHEN array_length(search_keywords, 1) > 0
                    THEN (SELECT COUNT(*) FROM unnest(sh.keywords) k WHERE k = ANY(search_keywords))
                    ELSE 0
                END
                +
                CASE WHEN p_query_text IS NOT NULL AND p_query_text != ''
                     AND sh.last_summary IS NOT NULL
                     AND to_tsvector('english', sh.last_summary) @@ plainto_tsquery('english', p_query_text)
                    THEN 5
                    ELSE 0
                END
            )
        ) AS entry
        FROM session_handoffs sh
        WHERE
            sh.user_id = p_user_id
            AND (p_project IS NULL OR sh.project = p_project)
            AND (
                (array_length(search_keywords, 1) > 0 AND sh.keywords && search_keywords)
                OR
                (p_query_text IS NOT NULL AND p_query_text != ''
                 AND sh.last_summary IS NOT NULL
                 AND to_tsvector('english', sh.last_summary) @@ plainto_tsquery('english', p_query_text))
            )
    ) sub
    WHERE (sub.entry->>'relevance_score')::int > 0
    LIMIT p_limit;

    RETURN jsonb_build_object(
        'results', results,
        'count', jsonb_array_length(results),
        'search_keywords', to_jsonb(search_keywords),
        'query_text', p_query_text
    );
END;
$$;

COMMENT ON FUNCTION search_knowledge(TEXT, TEXT[], TEXT, TEXT, INT, TEXT) IS
    'Search accumulated session knowledge with multi-tenant isolation. '
    'Uses GIN indexes for fast array-overlap and full-text search queries. '
    'Results scoped to p_user_id.';

-- ==============================================================================
-- MIGRATION COMPLETE
-- ==============================================================================

-- ─── Fix 4: Make title column nullable ──────────────────────────
-- The v2.0 code doesn't always send a title, but migration 015
-- defines it as NOT NULL. Fix by adding a default.
ALTER TABLE session_ledger ALTER COLUMN title SET DEFAULT '';
ALTER TABLE session_ledger ALTER COLUMN title DROP NOT NULL;
