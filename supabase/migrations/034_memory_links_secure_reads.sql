-- Migration 034: Secure Read RPCs for Memory Links
-- Closes a security and correctness gap where Supabase REST API reads on memory_links
-- lacked GDPR tombstone, TTL archive filtering, and tenant isolation, unlike SQLite.

BEGIN;

-- RPC for securely reading outbound links (getLinksFrom)
CREATE OR REPLACE FUNCTION public.prism_get_links_from(
    p_source_id UUID,
    p_user_id TEXT,
    p_min_strength REAL DEFAULT 0.0,
    p_limit INTEGER DEFAULT 25
)
RETURNS TABLE (
    source_id UUID,
    target_id UUID,
    link_type TEXT,
    strength REAL,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    last_traversed_at TIMESTAMPTZ
)
LANGUAGE sql
SECURITY DEFINER
AS $$
    SELECT m.source_id, m.target_id, m.link_type, m.strength, m.metadata, m.created_at, m.last_traversed_at
    FROM public.memory_links m
    JOIN public.session_ledger target ON m.target_id = target.id
    WHERE m.source_id = p_source_id
      AND m.strength >= p_min_strength
      AND target.user_id = p_user_id
      AND target.deleted_at IS NULL
      AND (target.archived_at IS NULL OR m.link_type IN ('spawned_from', 'supersedes'))
    ORDER BY m.strength DESC, m.last_traversed_at DESC
    LIMIT p_limit;
$$;

-- RPC for securely reading inbound links (getLinksTo)
CREATE OR REPLACE FUNCTION public.prism_get_links_to(
    p_target_id UUID,
    p_user_id TEXT,
    p_min_strength REAL DEFAULT 0.0,
    p_limit INTEGER DEFAULT 25
)
RETURNS TABLE (
    source_id UUID,
    target_id UUID,
    link_type TEXT,
    strength REAL,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    last_traversed_at TIMESTAMPTZ
)
LANGUAGE sql
SECURITY DEFINER
AS $$
    SELECT m.source_id, m.target_id, m.link_type, m.strength, m.metadata, m.created_at, m.last_traversed_at
    FROM public.memory_links m
    JOIN public.session_ledger source ON m.source_id = source.id
    WHERE m.target_id = p_target_id
      AND m.strength >= p_min_strength
      AND source.user_id = p_user_id
      AND source.deleted_at IS NULL
      AND (source.archived_at IS NULL OR m.link_type IN ('spawned_from', 'supersedes'))
    ORDER BY m.strength DESC, m.last_traversed_at DESC
    LIMIT p_limit;
$$;

COMMIT;
