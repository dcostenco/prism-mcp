-- Migration 041: GDPR soft-delete filter for get_compaction_candidates
--
-- Migration 020 added p_user_id scoping to get_compaction_candidates but
-- predates the soft-delete contract from migration 035. As a result the
-- RPC counts tombstoned (deleted_at IS NOT NULL) entries toward each
-- project's "needs compaction" total. Two consequences:
--
--   1. A user who exercised their GDPR right to erasure on N entries
--      still sees those entries inflate the active count — a
--      soft-deleted entry should be invisible to every read path.
--   2. Compaction may be triggered on projects whose "live" entry count
--      is below threshold, causing wasted summarization API calls.
--
-- Fix: add `AND sl.deleted_at IS NULL` to the WHERE clause. Drop +
-- recreate is required because the function signature is unchanged
-- (CREATE OR REPLACE on the same signature is sufficient for SQL-LANG
-- functions, but the comment is also being refreshed).

BEGIN;

DROP FUNCTION IF EXISTS get_compaction_candidates(INT, INT, TEXT);

CREATE OR REPLACE FUNCTION get_compaction_candidates(
  p_threshold INT DEFAULT 50,
  p_keep_recent INT DEFAULT 10,
  p_user_id TEXT DEFAULT 'default'
) RETURNS TABLE(project TEXT, total_entries BIGINT, to_compact BIGINT)
LANGUAGE sql
AS $$
  SELECT
    sl.project,
    COUNT(*) AS total_entries,
    COUNT(*) - p_keep_recent AS to_compact
  FROM session_ledger sl
  WHERE sl.archived_at IS NULL
    AND sl.deleted_at IS NULL
    AND sl.is_rollup = FALSE
    AND sl.user_id = p_user_id
  GROUP BY sl.project
  HAVING COUNT(*) > p_threshold;
$$;

COMMENT ON FUNCTION get_compaction_candidates(INT, INT, TEXT) IS
  'Finds projects needing compaction, scoped to a single user_id. '
  'GDPR: tombstoned entries (deleted_at IS NOT NULL) are excluded '
  'from the active-count total — a user who erased entries should '
  'not see those entries trigger compaction.';

COMMIT;
