/**
 * Auto-Linker — Automatic Graph Edge Creation (v6.0 Phase 3, Workstream 1)
 *
 * ═══════════════════════════════════════════════════════════════════
 * PURPOSE:
 *   When a new ledger entry is saved, this module fires two non-blocking
 *   graph-building tasks in parallel:
 *
 *   1. Temporal Chain: Links the new entry to the previous entry in the
 *      same conversation via a `temporal_next` directed edge.
 *
 *   2. Keyword Overlap: Pushes heavy intersection logic to the DB layer
 *      via `findKeywordOverlapEntries()`, then creates bidirectional
 *      `related_to` edges for each match.
 *
 * DESIGN PRINCIPLES:
 *   - **Non-blocking**: All work is fire-and-forget via `setImmediate()`.
 *     graph failures NEVER affect the primary MCP response path.
 *   - **Idempotent**: All `createLink()` calls use INSERT OR IGNORE on
 *     the composite PK (source_id, target_id, link_type).
 *   - **Tenant-safe**: All queries are scoped by user_id.
 *   - **GDPR-safe**: The `findKeywordOverlapEntries()` query respects
 *     deleted_at IS NULL and archived_at IS NULL filters.
 *
 * INTEGRATION POINT:
 *   Called from `sessionSaveLedgerHandler()` after a successful save,
 *   wrapped in:
 *     setImmediate(() => {
 *       autoLinkEntry(entryId, project, keywords, conversationId, userId, storage)
 *         .catch(err => console.error(`[autoLinker] Non-fatal: ${err.message}`));
 *     });
 * ═══════════════════════════════════════════════════════════════════
 */

import { debugLog } from "./logger.js";
import type { StorageBackend, MemoryLink } from "../storage/interface.js";

// ─── Configuration ───────────────────────────────────────────
// Minimum shared keywords required to create a `related_to` link.
// Set to 3 per the approved design (avoids noise from low-quality matches).
const MIN_SHARED_KEYWORDS = 3;

// Maximum number of related_to links to create per entry.
// Prevents runaway link creation for entries with common keywords.
const MAX_KEYWORD_LINKS_PER_ENTRY = 10;

// Default link strength for newly created auto-links.
// Related-to links start lower (0.5) since they're auto-generated.
// Temporal links start at 1.0 since the relationship is structural/certain.
const TEMPORAL_LINK_STRENGTH = 1.0;
const KEYWORD_LINK_STRENGTH = 0.5;

/**
 * Main entry point: auto-link a newly saved ledger entry.
 *
 * Runs temporal chaining and keyword overlap in parallel.
 * Both are independently wrapped in try/catch — if one fails,
 * the other still completes.
 *
 * @param entryId        - ID of the newly saved entry
 * @param project        - Project identifier
 * @param keywords       - Extracted keywords from the entry
 * @param conversationId - Conversation this entry belongs to
 * @param userId         - Tenant ID for isolation
 * @param storage        - StorageBackend instance
 * @param createdAt      - Original creation timestamp for exact temporal lookup
 */
export async function autoLinkEntry(
  entryId: string,
  project: string,
  keywords: string[],
  conversationId: string,
  userId: string,
  storage: StorageBackend,
  createdAt: string
): Promise<{ temporalLinked: boolean; keywordLinksCreated: number }> {
  const result = { temporalLinked: false, keywordLinksCreated: 0 };

  // Run both strategies in parallel — they are independent
  const [temporalResult, keywordResult] = await Promise.allSettled([
    linkTemporal(entryId, project, conversationId, userId, storage, createdAt),
    linkByKeywordOverlap(entryId, project, keywords, userId, storage),
  ]);

  if (temporalResult.status === "fulfilled") {
    result.temporalLinked = temporalResult.value;
  } else {
    debugLog(`[autoLinker] Temporal linking failed (non-fatal): ${temporalResult.reason}`);
  }

  if (keywordResult.status === "fulfilled") {
    result.keywordLinksCreated = keywordResult.value;
  } else {
    debugLog(`[autoLinker] Keyword linking failed (non-fatal): ${keywordResult.reason}`);
  }

  debugLog(
    `[autoLinker] Entry ${entryId.substring(0, 8)}: ` +
    `temporal=${result.temporalLinked}, keywords=${result.keywordLinksCreated} links`
  );

  return result;
}

// ─── Strategy 1: Temporal Chain ──────────────────────────────
//
// Links the new entry to the previous entry in the same conversation.
// This creates a chronological chain that enables "what did I do next?"
// traversal patterns.
//
// SQL: Find the most recent entry in the same conversation_id + project
// that was created BEFORE this entry. The link is directed:
//   previousEntry → temporal_next → newEntry

async function linkTemporal(
  entryId: string,
  project: string,
  conversationId: string,
  userId: string,
  storage: StorageBackend,
  createdAt: string
): Promise<boolean> {
  // Find the immediate predecessor. GDPR: skip tombstoned entries —
  // a soft-deleted entry must not appear as the temporal predecessor
  // of a new one (would link the new entry to data the user erased).
  const previousEntries = await storage.getLedgerEntries({
    user_id: `eq.${userId}`,
    project: `eq.${project}`,
    conversation_id: `eq.${conversationId}`,
    created_at: `lt.${createdAt}`,
    deleted_at: "is.null",
    select: "id,created_at",
    order: "created_at.desc",
    limit: "1",
  });

  if (!previousEntries || previousEntries.length === 0) {
    debugLog(`[autoLinker] No previous entry in conversation ${conversationId.substring(0, 8)} before ${createdAt} — skipping temporal link`);
    return false;
  }

  const previousEntry = previousEntries[0];
  const link: MemoryLink = {
    source_id: (previousEntry as any).id,
    target_id: entryId,
    link_type: "temporal_next",
    strength: TEMPORAL_LINK_STRENGTH,
    metadata: JSON.stringify({ conversation_id: conversationId }),
  };

  await storage.createLink(link, userId);
  debugLog(
    `[autoLinker] Temporal link: ${(previousEntry as any).id.substring(0, 8)} → ${entryId.substring(0, 8)}`
  );
  return true;
}

// ─── Strategy 2: Keyword Overlap ─────────────────────────────
//
// Finds existing entries that share ≥ MIN_SHARED_KEYWORDS keywords
// with the new entry. Creates bidirectional `related_to` links.
//
// The heavy intersection logic is pushed to the DB layer via
// `findKeywordOverlapEntries()` — this uses CTE-based json_each
// explosion with hash joins (O(N) vs O(N²) cross-join).

async function linkByKeywordOverlap(
  entryId: string,
  project: string,
  keywords: string[],
  userId: string,
  storage: StorageBackend,
): Promise<number> {
  if (!keywords || keywords.length < MIN_SHARED_KEYWORDS) {
    debugLog(`[autoLinker] Entry has ${keywords?.length ?? 0} keywords (min: ${MIN_SHARED_KEYWORDS}) — skipping keyword overlap`);
    return 0;
  }

  // Push intersection logic to the DB — see SqliteStorage.findKeywordOverlapEntries()
  const overlappingEntries = await storage.findKeywordOverlapEntries(
    entryId,
    project,
    keywords,
    userId,
    MIN_SHARED_KEYWORDS,
    MAX_KEYWORD_LINKS_PER_ENTRY,
  );

  if (overlappingEntries.length === 0) {
    debugLog(`[autoLinker] No keyword overlaps found for entry ${entryId.substring(0, 8)}`);
    return 0;
  }

  // Create bidirectional related_to links
  let created = 0;
  for (const match of overlappingEntries) {
    try {
      // Forward link: newEntry → match
      await storage.createLink({
        source_id: entryId,
        target_id: match.id,
        link_type: "related_to",
        strength: KEYWORD_LINK_STRENGTH,
        metadata: JSON.stringify({ shared_keywords: match.shared_count }),
      }, userId);

      // Reverse link: match → newEntry (bidirectional)
      await storage.createLink({
        source_id: match.id,
        target_id: entryId,
        link_type: "related_to",
        strength: KEYWORD_LINK_STRENGTH,
        metadata: JSON.stringify({ shared_keywords: match.shared_count }),
      }, userId);

      created++;
    } catch (err) {
      // Individual link creation failure is non-fatal — continue with next match
      debugLog(`[autoLinker] Failed to create link to ${match.id.substring(0, 8)}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  debugLog(
    `[autoLinker] Created ${created} keyword links for entry ${entryId.substring(0, 8)} ` +
    `(${overlappingEntries.length} candidates found)`
  );
  return created;
}
