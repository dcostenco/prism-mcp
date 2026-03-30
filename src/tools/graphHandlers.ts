
import { formatRulesBlock, applySentinelBlock, SENTINEL_START, SENTINEL_END, REDACT_PATTERNS } from "./commonHelpers.js";
/**
 * Session Memory Handlers (v2.0 — StorageBackend Refactor)
 *
 * ═══════════════════════════════════════════════════════════════════
 * v2.0 CHANGES IN THIS FILE (Step 1: Pure Refactor)
 *
 * BEFORE: All handlers called supabasePost/Get/Rpc/Patch/Delete directly.
 * AFTER:  All handlers call StorageBackend methods via `getStorage()`.
 *
 * This refactor changes ZERO behavior. Every method call maps 1:1 to
 * the same Supabase API call (see src/storage/supabase.ts for mapping).
 *
 * WHY: This enables Step 2 (SQLite local mode) — once SqliteStorage
 * implements the same interface, the handlers work with both backends
 * without any code changes.
 * ═══════════════════════════════════════════════════════════════════
 */

import { debugLog } from "../utils/logger.js";
import { getStorage } from "../storage/index.js";
import { toKeywordArray } from "../utils/keywordExtractor.js";
import { getLLMProvider } from "../utils/llm/factory.js";
import { getCurrentGitState, getGitDrift } from "../utils/git.js";
import { getSetting, getAllSettings } from "../storage/configStorage.js";
import { mergeHandoff, dbToHandoffSchema, sanitizeForMerge } from "../utils/crdtMerge.js";

// ─── Phase 1: Explainability & Memory Lineage ────────────────
// These utilities provide structured tracing metadata for search operations.
// When `enable_trace: true` is passed to session_search_memory or knowledge_search,
// a separate MCP content block (content[1]) is returned with a MemoryTrace object
// containing: strategy, scores, latency breakdown (embedding/storage/total), and metadata.
// See src/utils/tracing.ts for full type definitions and design decisions.
import { createMemoryTrace, traceToContentBlock } from "../utils/tracing.js";
import { GOOGLE_API_KEY, PRISM_USER_ID, PRISM_AUTO_CAPTURE, PRISM_CAPTURE_PORTS } from "../config.js";
import { captureLocalEnvironment } from "../utils/autoCapture.js";
import { fireCaptionAsync } from "../utils/imageCaptioner.js";
import {
  isSessionSaveLedgerArgs,
  isSessionSaveHandoffArgs,
  isSessionLoadContextArgs,
  isKnowledgeSearchArgs,
  isKnowledgeForgetArgs,
  isSessionSearchMemoryArgs,
  isBackfillEmbeddingsArgs,
  isMemoryHistoryArgs,
  isMemoryCheckoutArgs,
  isSessionHealthCheckArgs,        // v2.2.0: health check type guard
  isSessionForgetMemoryArgs,       // Phase 2: GDPR-compliant memory deletion type guard
  isKnowledgeSetRetentionArgs,     // v3.1: TTL retention policy type guard
  // v4.0: Active Behavioral Memory type guards
  isSessionSaveExperienceArgs,
  isKnowledgeVoteArgs,
  // v4.2: Sync Rules type guard
  isKnowledgeSyncRulesArgs,
  // v5.1: Deep Storage Mode type guard
  isDeepStoragePurgeArgs,
  // v5.5: SDM Intuitive Recall type guard
  isSessionIntuitiveRecallArgs,
} from "./sessionMemoryDefinitions.js";

// v4.2: File system access for knowledge_sync_rules
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, dirname, resolve, isAbsolute, sep, relative } from "node:path";

// v3.1: In-memory debounce lock for auto-compaction.
// Prevents multiple concurrent Gemini compaction tasks for the same project
// when many agents call session_save_ledger at the same time.
const activeCompactions = new Set<string>();


// ─── v0.4.0: Import server type for resource notifications ───
import type { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { notifyResourceUpdate } from "../server.js";

// ─── Save Ledger Handler ──────────────────────────────────────

/**
 * Appends an immutable session log entry.
 *
 * Think of the ledger as a "commit log" for agent work — once written, entries
 * are never modified. This creates a permanent audit trail of all work done.
 *
 * After saving, generates an embedding vector for the entry via fire-and-forget.
 */
import { computeEffectiveImportance, updateLastAccessed } from "../utils/cognitiveMemory.js";
export async function knowledgeSearchHandler(args: unknown) {
  if (!isKnowledgeSearchArgs(args)) {
    throw new Error("Invalid arguments for knowledge_search");
  }

  // Phase 1: destructure enable_trace (defaults to false for backward compat)
  const { project, query, category, limit = 10, enable_trace = false } = args;

  debugLog(`[knowledge_search] Searching: project=${project || "all"}, query="${query || ""}", category=${category || "any"}, limit=${limit}`);

  // Phase 1: Capture total start time for latency measurement
  const totalStart = performance.now();
  const searchKeywords = query ? toKeywordArray(query) : [];
  const storage = await getStorage();

  // Phase 1: Capture storage-specific start time to isolate DB latency
  // from keyword extraction and other overhead
  const storageStart = performance.now();
  const data = await storage.searchKnowledge({
    project: project || null,
    keywords: searchKeywords,
    category: category || null,
    queryText: query || null,
    limit: Math.min(limit, 50),
    userId: PRISM_USER_ID,
  });
  const storageMs = performance.now() - storageStart;
  const totalMs = performance.now() - totalStart;

  if (!data) {
    // Phase 1: Use contentBlocks array instead of inline object
    // so we can conditionally push the trace block at content[1]
    const contentBlocks: Array<{ type: string; text: string }> = [{
      type: "text",
      text: `🔍 No knowledge found matching your search.\n` +
        (query ? `Query: "${query}"\n` : "") +
        (category ? `Category: ${category}\n` : "") +
        (project ? `Project: ${project}\n` : "") +
        `\nTip: Try session_search_memory for semantic (meaning-based) search ` +
        `if keyword search doesn't find what you need.`,
    }];

    // Phase 1: Append trace block even on empty results — this tells
    // the developer the search DID execute, it just found nothing.
    // topScore and threshold are null for keyword search (no scoring system).
    if (enable_trace) {
      const trace = createMemoryTrace({
        strategy: "keyword",
        query: query || "",
        resultCount: 0,
        topScore: null,     // keyword search doesn't produce similarity scores
        threshold: null,     // keyword search has no threshold concept
        embeddingMs: 0,      // no embedding needed for keyword search
        storageMs,
        totalMs,
        project: project || null,
      });
      contentBlocks.push(traceToContentBlock(trace));
    }

    return { content: contentBlocks, isError: false };
  }

  if (data.results && Array.isArray(data.results)) {
    const resultIds = data.results.map((r: any) => r.id).filter(Boolean);
    if (resultIds.length > 0) {
      updateLastAccessed(resultIds);
    }
    
    // Mutate results to surface effective importance
    for (const r of data.results as any[]) {
      if (typeof r.importance === 'number' && r.importance > 0) {
        r.effective_importance = computeEffectiveImportance(r.importance, r.last_accessed_at, r.created_at);
      }
    }
  }

  // Phase 1: Wrap in contentBlocks array for optional trace attachment
  const contentBlocks: Array<{ type: string; text: string }> = [{
    type: "text",
    text: `🧠 Found ${data.count} knowledge entries:\n\n${JSON.stringify(data.results || data, null, 2)}`,
  }];

  // Phase 1: Attach MemoryTrace with strategy="keyword" and timing data
  if (enable_trace) {
    const trace = createMemoryTrace({
      strategy: "keyword",
      query: query || "",
      resultCount: data.count,
      topScore: null,       // keyword search doesn't produce similarity scores
      threshold: null,       // keyword search has no threshold concept
      embeddingMs: 0,        // no embedding needed for keyword search
      storageMs,
      totalMs,
      project: project || null,
    });
    contentBlocks.push(traceToContentBlock(trace));
  }

  // ── v6.0 Phase 3: 1-Hop Graph Expansion ──────────────────
  // Same pattern as sessionSearchMemoryHandler:
  // Traverse outbound links from direct hits to find associated memories.
  // Graph-expanded results are BONUS — don't consume limit slots.
  try {
    // Extract IDs from the knowledge search results
    const directIds = new Set<string>();
    if (data.results && Array.isArray(data.results)) {
      for (const entry of data.results) {
        if ((entry as any)?.id) directIds.add((entry as any).id);
      }
    }

    if (directIds.size > 0) {
      const enrichedIds = new Set<string>();
      const maxGraphResults = Math.min(limit, 10);

      for (const directId of directIds) {
        if (enrichedIds.size >= maxGraphResults) break;
        const links = await storage.getLinksFrom(directId, PRISM_USER_ID, 0.3, 5);
        for (const link of links) {
          if (!directIds.has(link.target_id) && !enrichedIds.has(link.target_id)) {
            enrichedIds.add(link.target_id);
            if (enrichedIds.size >= maxGraphResults) break;
          }
        }
      }

      if (enrichedIds.size > 0) {
        const enrichedEntries = await storage.getLedgerEntries({
          user_id: `eq.${PRISM_USER_ID}`,
          ids: [...enrichedIds],
          select: "id,summary,project,created_at",
        });

        if (enrichedEntries.length > 0) {
          const graphFormatted = enrichedEntries.map((e: any) =>
            `[🔗] ${e.created_at?.split("T")[0] || "unknown"} — ${e.project || "unknown"}\n` +
            `  Summary: ${e.summary}`
          ).join("\n");

          contentBlocks[0] = {
            type: "text",
            text: contentBlocks[0].text +
              `\n\n🔗 Graph-connected memories (${enrichedEntries.length} via 1-hop expansion):\n\n${graphFormatted}`,
          };

          // Fire-and-forget: reinforce traversed links
          for (const directId of directIds) {
            storage.getLinksFrom(directId, PRISM_USER_ID, 0.3, 5)
              .then(links => {
                for (const link of links) {
                  if (enrichedIds.has(link.target_id)) {
                    storage.reinforceLink(directId, link.target_id, link.link_type).catch(() => {});
                  }
                }
              })
              .catch(() => {});
          }
        }
      }
    }
  } catch (graphErr) {
    debugLog(`[knowledge_search] Graph expansion failed (non-fatal): ${graphErr instanceof Error ? graphErr.message : String(graphErr)}`);
  }

  return { content: contentBlocks, isError: false };
}

export async function knowledgeForgetHandler(args: unknown) {
  if (!isKnowledgeForgetArgs(args)) {
    throw new Error("Invalid arguments for knowledge_forget");
  }

  const {
    project,
    category,
    older_than_days,
    clear_handoff = false,
    confirm_all = false,
    dry_run = false,
  } = args;

  if (!project && !confirm_all) {
    return {
      content: [{
        type: "text",
        text: `⚠️ Safety check: You must specify a 'project' to forget, ` +
          `or set 'confirm_all: true' to wipe all entries.\n` +
          `This prevents accidental deletion of all knowledge.`,
      }],
      isError: true,
    };
  }

  debugLog(`[knowledge_forget] ${dry_run ? "DRY RUN: " : ""}Forgetting: ` +
    `project=${project || "ALL"}, category=${category || "any"}, ` +
    `older_than=${older_than_days || "any"}d, clear_handoff=${clear_handoff}`);

  const storage = await getStorage();

  const ledgerParams: Record<string, string> = {};
  ledgerParams.user_id = `eq.${PRISM_USER_ID}`;
  if (project) {
    ledgerParams.project = `eq.${project}`;
  }
  if (category) {
    ledgerParams.keywords = `cs.{cat:${category}}`;
  }
  if (older_than_days) {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - older_than_days);
    ledgerParams.created_at = `lt.${cutoffDate.toISOString()}`;
  }

  let ledgerCount = 0;
  let handoffCleared = false;

  if (dry_run) {
    const selectParams = { ...ledgerParams, select: "id" };
    const entries = await storage.getLedgerEntries(selectParams);
    ledgerCount = entries.length;
  } else {
    const result = await storage.deleteLedger(ledgerParams);
    ledgerCount = result.length;

    if (clear_handoff && project) {
      await storage.deleteHandoff(project, PRISM_USER_ID);
      handoffCleared = true;
    }
  }

  const action = dry_run ? "would be forgotten" : "forgotten";
  const emoji = dry_run ? "🔍" : "🧹";

  return {
    content: [{
      type: "text",
      text: `${emoji} ${ledgerCount} ledger entries ${action}` +
        (project ? ` for project "${project}"` : "") +
        (category ? ` in category "${category}"` : "") +
        (older_than_days ? ` older than ${older_than_days} days` : "") +
        `.\n` +
        (handoffCleared ? `🗑️ Handoff state also cleared for "${project}".\n` : "") +
        (dry_run ? `\n💡 This was a dry run — nothing was actually deleted. Remove dry_run to execute.` : "") +
        (!dry_run && ledgerCount > 0 ? `\n✅ Knowledge base pruned. Fresh start!` : ""),
    }],
    isError: false,
  };
}

export async function sessionSearchMemoryHandler(args: unknown) {
  if (!isSessionSearchMemoryArgs(args)) {
    throw new Error("Invalid arguments for session_search_memory");
  }

  const {
    query,
    project,
    limit = 5,
    similarity_threshold = 0.7,
    // Phase 1: enable_trace defaults to false for full backward compatibility.
    // When true, a MemoryTrace JSON block is appended as content[1].
    enable_trace = false,
    // v5.2: Context-Weighted Retrieval — biases search toward active work context
    context_boost = false,
  } = args;

  debugLog(
    `[session_search_memory] Semantic search: query="${query}", ` +
    `project=${project || "all"}, limit=${limit}, threshold=${similarity_threshold}` +
    `${context_boost ? ", context_boost=ON" : ""}`
  );

  // Phase 1: Start total latency timer BEFORE any work (embedding + storage)
  const totalStart = performance.now();

  // Step 1: Generate embedding for the search query
  if (!GOOGLE_API_KEY) {
    return {
      content: [{
        type: "text",
        text: `❌ Semantic search requires GOOGLE_API_KEY for embedding generation.\n` +
          `Set this environment variable and restart the server.\n\n` +
          `💡 As a workaround, try knowledge_search (keyword-based) instead.`,
      }],
      isError: true,
    };
  }

  let queryEmbedding: number[];
  // Phase 1: Start embedding latency timer — isolates Gemini API call time.
  // This is the most variable component: 50ms on a good day, 2000ms under load.
  const embeddingStart = performance.now();

  // ── v5.2: Context-Weighted Retrieval ───────────────────────────
  // When context_boost is enabled, prepend active project context to the
  // search query before embedding generation. This naturally biases the
  // embedding vector toward memories from the same project/branch/context.
  // Elegant: no scoring heuristics needed — semantics do the work.
  let effectiveQuery = query;
  if (context_boost && project) {
    try {
      const storage = await getStorage();
      const ctx = await storage.loadContext(project, "quick", PRISM_USER_ID);
      const contextParts: string[] = [];
      if (ctx && typeof ctx === "object") {
        const ctxObj = ctx as Record<string, unknown>;
        if (ctxObj.active_branch) contextParts.push(`branch: ${ctxObj.active_branch}`);
        if (ctxObj.key_context) contextParts.push(`context: ${String(ctxObj.key_context).substring(0, 200)}`);
        const keywords = ctxObj.keywords as string[] | undefined;
        if (keywords?.length) contextParts.push(`keywords: ${keywords.slice(0, 5).join(", ")}`);
      }
      if (contextParts.length > 0) {
        effectiveQuery = `[${contextParts.join("; ")}] ${query}`;
        debugLog(`[session_search_memory] Context boost applied: "${effectiveQuery.substring(0, 100)}..."`);
      }
    } catch {
      // Context load failed — proceed with unmodified query (graceful degradation)
      debugLog("[session_search_memory] Context boost failed (non-fatal) — using original query");
    }
  } else if (context_boost && !project) {
    // User enabled context_boost but didn't specify a project — can't boost without context
    debugLog("[session_search_memory] context_boost ignored — requires a project parameter to load context");
  }

  try {
    queryEmbedding = await getLLMProvider().generateEmbedding(effectiveQuery);
  } catch (err) {
    return {
      content: [{
        type: "text",
        text: `❌ Failed to generate embedding for query: ${err instanceof Error ? err.message : String(err)}\n\n` +
          `💡 Try knowledge_search (keyword-based) as a fallback.`,
      }],
      isError: true,
    };
  }
  // Phase 1: Capture embedding API latency
  const embeddingMs = performance.now() - embeddingStart;

  // Step 2: Search via storage backend
  try {
    const storage = await getStorage();
    // Phase 1: Start storage latency timer — isolates DB query time.
    // For Supabase: this measures the pgvector cosine distance RPC call.
    // For SQLite: this measures the local sqlite-vec similarity search.
    const storageStart = performance.now();
    const results = await storage.searchMemory({
      queryEmbedding: JSON.stringify(queryEmbedding),
      project: project || null,
      limit: Math.min(limit, 20),
      similarityThreshold: similarity_threshold,
      userId: PRISM_USER_ID,
    });
    // Phase 1: Capture storage query latency and compute total
    const storageMs = performance.now() - storageStart;
    const totalMs = performance.now() - totalStart;

    if (results.length === 0) {
      // Phase 1: Use contentBlocks array so we can optionally push trace at [1]
      const contentBlocks: Array<{ type: string; text: string }> = [{
        type: "text",
        text: `🔍 No semantically similar sessions found for: "${query}"\n` +
          (project ? `Project: ${project}\n` : "") +
          `Similarity threshold: ${similarity_threshold}\n\n` +
          `Tips:\n` +
          `• Lower the similarity_threshold (e.g., 0.5) for broader results\n` +
          `• Try knowledge_search for keyword-based matching\n` +
          `• Ensure sessions have been saved with embeddings (requires GOOGLE_API_KEY)`,
      }];

      // Phase 1: Trace is still valuable on empty results — it proves the search
      // executed and reveals whether the bottleneck was embedding or storage.
      if (enable_trace) {
        const trace = createMemoryTrace({
          strategy: "semantic",
          query,
          resultCount: 0,
          topScore: null,          // no results = no top score
          threshold: similarity_threshold,
          embeddingMs,
          storageMs,
          totalMs,
          project: project || null,
        });
        contentBlocks.push(traceToContentBlock(trace));
      }

      return { content: contentBlocks, isError: false };
    }

    // ── v5.2: Dynamic Importance Decay (Ebbinghaus Curve) ──────
    // Compute effective_importance at retrieval time
    const resultIds = results.map((r: any) => r.id).filter(Boolean);

    // Fire-and-forget: update last_accessed_at for all returned results
    if (resultIds.length > 0) {
      updateLastAccessed(resultIds);
    }

    // Format results with similarity scores + effective importance
    const formatted = results.map((r: any, i: number) => {
      const score = typeof r.similarity === "number"
        ? `${(r.similarity * 100).toFixed(1)}%`
        : "N/A";

      // Dynamic importance decay
      const baseImportance = r.importance ?? 0;
      const effectiveImportance = computeEffectiveImportance(baseImportance, r.last_accessed_at, r.created_at);

      const importanceStr = baseImportance > 0
        ? `  Importance: ${effectiveImportance}${effectiveImportance !== baseImportance ? ` (base: ${baseImportance}, decayed)` : ""}\n`
        : "";

      return `[${i + 1}] ${score} similar — ${r.session_date || "unknown date"}\n` +
        `  Project: ${r.project}\n` +
        `  Summary: ${r.summary}\n` +
        importanceStr +
        (r.decisions?.length ? `  Decisions: ${r.decisions.join("; ")}\n` : "") +
        (r.files_changed?.length ? `  Files: ${r.files_changed.join(", ")}\n` : "");
    }).join("\n");

    // Phase 1: content[0] = human-readable results (unchanged from pre-Phase 1)
    const contentBlocks: Array<{ type: string; text: string }> = [{
      type: "text",
      text: `🧠 Found ${results.length} semantically similar sessions:\n\n${formatted}`,
    }];

    // Phase 1: content[1] = machine-readable MemoryTrace (only when enable_trace=true)
    // topScore is read from results[0].similarity — this is the cosine distance
    // already returned by SemanticSearchResult in the storage interface.
    // No storage layer modifications were needed ("Score Bubbling" reviewer level-up).
    if (enable_trace) {
      const topScore = results.length > 0 && typeof results[0].similarity === "number"
        ? results[0].similarity
        : null;
      const trace = createMemoryTrace({
        strategy: "semantic",
        query,
        resultCount: results.length,
        topScore,
        threshold: similarity_threshold,
        embeddingMs,
        storageMs,
        totalMs,
        project: project || null,
      });
      contentBlocks.push(traceToContentBlock(trace));
    }

    // ── v6.0 Phase 3: 1-Hop Graph Expansion ──────────────────
    // After direct hits, traverse outbound links from each result to
    // find associated memories. Graph-expanded results are BONUS — they
    // don't consume limit slots. Hard-capped at `limit` additional results
    // to protect LLM context windows.
    //
    // Fire-and-forget: errors degrade gracefully to just direct hits.
    try {
      const directIds = new Set(results.map((r: any) => r.id).filter(Boolean));
      const enrichedIds = new Set<string>();
      const maxGraphResults = Math.min(limit, 10); // Hard cap

      for (const directId of directIds) {
        if (enrichedIds.size >= maxGraphResults) break;
        const links = await storage.getLinksFrom(directId, PRISM_USER_ID, 0.3, 5);
        for (const link of links) {
          if (!directIds.has(link.target_id) && !enrichedIds.has(link.target_id)) {
            enrichedIds.add(link.target_id);
            if (enrichedIds.size >= maxGraphResults) break;
          }
        }
      }

      if (enrichedIds.size > 0) {
        // Fetch the actual entries for enriched IDs
        const enrichedEntries = await storage.getLedgerEntries({
          user_id: `eq.${PRISM_USER_ID}`,
          ids: [...enrichedIds],
          select: "id,summary,project,created_at",
        });

        if (enrichedEntries.length > 0) {
          const graphFormatted = enrichedEntries.map((e: any) =>
            `[🔗] ${e.created_at?.split("T")[0] || "unknown"} — ${e.project || "unknown"}\n` +
            `  Summary: ${e.summary}`
          ).join("\n");

          contentBlocks[0] = {
            type: "text",
            text: contentBlocks[0].text +
              `\n\n🔗 Graph-connected memories (${enrichedEntries.length} via 1-hop expansion):\n\n${graphFormatted}`,
          };

          // Fire-and-forget: reinforce traversed links
          for (const directId of directIds) {
            storage.getLinksFrom(directId, PRISM_USER_ID, 0.3, 5)
              .then(links => {
                for (const link of links) {
                  if (enrichedIds.has(link.target_id)) {
                    storage.reinforceLink(directId, link.target_id, link.link_type).catch(() => {});
                  }
                }
              })
              .catch(() => {});
          }
        }
      }
    } catch (graphErr) {
      debugLog(`[session_search_memory] Graph expansion failed (non-fatal): ${graphErr instanceof Error ? graphErr.message : String(graphErr)}`);
    }

    return { content: contentBlocks, isError: false };
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (errorMsg.includes("vector") || errorMsg.includes("does not exist")) {
      return {
        content: [{
          type: "text",
          text: `❌ Semantic search is not available: pgvector extension may not be enabled.\n\n` +
            `To fix: Go to Supabase Dashboard → Database → Extensions → enable "vector"\n` +
            `Then run migration 018_semantic_search.sql\n\n` +
            `💡 Use knowledge_search (keyword-based) as an alternative.`,
        }],
        isError: true,
      };
    }
    throw err;
  }
}

export async function knowledgeUpvoteHandler(args: unknown) {
  if (!isKnowledgeVoteArgs(args)) {
    throw new Error("Invalid arguments for knowledge_upvote");
  }

  const storage = await getStorage();
  try {
    await storage.adjustImportance(args.id, 1, PRISM_USER_ID);
    debugLog(`[knowledge_upvote] Upvoted entry ${args.id}`);
    return {
      content: [{ type: "text", text: `👍 Entry ${args.id} upvoted (+1 importance).` }],
      isError: false,
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text", text: `❌ Failed to upvote entry ${args.id}: ${msg}` }],
      isError: true,
    };
  }
}

export async function knowledgeDownvoteHandler(args: unknown) {
  if (!isKnowledgeVoteArgs(args)) {
    throw new Error("Invalid arguments for knowledge_downvote");
  }

  const storage = await getStorage();
  try {
    await storage.adjustImportance(args.id, -1, PRISM_USER_ID);
    debugLog(`[knowledge_downvote] Downvoted entry ${args.id}`);
    return {
      content: [{ type: "text", text: `👎 Entry ${args.id} downvoted (-1 importance).` }],
      isError: false,
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text", text: `❌ Failed to downvote entry ${args.id}: ${msg}` }],
      isError: true,
    };
  }
}

export async function knowledgeSyncRulesHandler(args: unknown) {
  if (!isKnowledgeSyncRulesArgs(args)) {
    throw new Error("Invalid arguments for knowledge_sync_rules");
  }

  const { project, target_file = ".cursorrules", dry_run = false } = args;
  const storage = await getStorage();

  // 1. Resolve repo path
  const repoPath = await getSetting(`repo_path:${project}`, "");
  if (!repoPath || !repoPath.trim()) {
    return {
      content: [{
        type: "text",
        text: `❌ No repo_path configured for project "${project}".\n` +
          `Set it in the Mind Palace dashboard (Settings → Project Repo Paths) before syncing rules.`,
      }],
      isError: true,
    };
  }

  const normalizedRepoPath = repoPath.trim().replace(/\/+$/, "");

  // 2. Fetch graduated insights
  const insights = await storage.getGraduatedInsights(project, PRISM_USER_ID, 7);

  if (insights.length === 0) {
    return {
      content: [{
        type: "text",
        text: `ℹ️ No graduated insights found for project "${project}".\n` +
          `Insights graduate when their importance score reaches 7 or higher.\n` +
          `Use \`knowledge_upvote\` to increase importance of valuable entries.`,
      }],
      isError: false,
    };
  }

  // 3. Format rules block
  const rulesBlock = formatRulesBlock(
    insights.map(i => ({ ...i, importance: i.importance ?? 0 })),
    project
  );

  // 4. Dry-run: return preview without writing
  if (dry_run) {
    return {
      content: [{
        type: "text",
        text: `🔍 **Dry Run Preview** — ${insights.length} graduated insight(s) for "${project}":\n\n` +
          `Target: ${normalizedRepoPath}/${target_file}\n\n` +
          `\`\`\`markdown\n${rulesBlock}\n\`\`\`\n\n` +
          `Run again without \`dry_run\` to write this to disk.`,
      }],
      isError: false,
    };
  }

  // 5. Idempotent file write — with path traversal protection
  // Reject absolute paths (e.g. "/etc/hosts")
  if (isAbsolute(target_file)) {
    return {
      content: [{
        type: "text",
        text: `❌ Security Error: target_file cannot be an absolute path. Got: "${target_file}"`,
      }],
      isError: true,
    };
  }

  // Resolve both paths to their canonical forms, then assert containment
  const resolvedRepo = resolve(normalizedRepoPath);
  const targetPath = resolve(resolvedRepo, target_file);
  const relativePath = relative(resolvedRepo, targetPath);

  // Ensure the resolved target is strictly inside the repo root
  // (handles "../../../etc/hosts" style traversal)
  if (relativePath.startsWith("..") || isAbsolute(relativePath)) {
    return {
      content: [{
        type: "text",
        text: `❌ Security Error: Path traversal blocked.\n` +
          `"${target_file}" resolves outside the repo root "${resolvedRepo}".`,
      }],
      isError: true,
    };
  }

  // Ensure directory exists (handles nested target_file like ".config/rules.md")
  const targetDir = dirname(targetPath);
  if (!existsSync(targetDir)) {
    await mkdir(targetDir, { recursive: true });
  }

  let existingContent = "";
  try {
    existingContent = await readFile(targetPath, "utf-8");
  } catch {
    // File doesn't exist yet — will be created
    debugLog(`[knowledge_sync_rules] File ${targetPath} doesn't exist, creating new`);
  }

  const newContent = applySentinelBlock(existingContent, rulesBlock);
  await writeFile(targetPath, newContent, "utf-8");

  debugLog(`[knowledge_sync_rules] Synced ${insights.length} insights to ${targetPath}`);

  return {
    content: [{
      type: "text",
      text: `✅ Synced ${insights.length} graduated insight(s) to \`${targetPath}\`\n\n` +
        `Top insights synced:\n` +
        insights.slice(0, 5).map(i =>
          `  • [${i.importance}] ${i.summary.substring(0, 80)}${i.summary.length > 80 ? "..." : ""}`
        ).join("\n") +
        (insights.length > 5 ? `\n  ... and ${insights.length - 5} more` : ""),
    }],
    isError: false,
  };
}

export async function sessionIntuitiveRecallHandler(
  args: unknown
): Promise<{ content: Array<{ type: "text"; text: string }>; isError?: boolean }> {
  if (!isSessionIntuitiveRecallArgs(args)) {
    return {
      content: [{ type: "text", text: "Invalid arguments for session_intuitive_recall" }],
      isError: true,
    };
  }

  try {
    const { getSdmEngine } = await import("../sdm/sdmEngine.js");
    const { decodeSdmVector } = await import("../sdm/sdmDecoder.js");

    const queryVector = await getLLMProvider().generateEmbedding(args.query);
    const sdmEngine = getSdmEngine(args.project);
    const targetVector = sdmEngine.read(new Float32Array(queryVector));

    const limit = args.limit ?? 3;
    const threshold = args.threshold ?? 0.55;

    const topMatches = await decodeSdmVector(args.project, targetVector, limit, threshold);

    let recallBlock = `🧠 **SDM Intuitive Recall for "${args.project}"**\n\n`;
    recallBlock += `Query: "${args.query}"\n`;
    recallBlock += `Target vector generated. Scanning ${topMatches.length > 0 ? topMatches.length + " latents surfaced." : "No strong patterns surfaced."}\n\n`;

    if (topMatches.length === 0) {
      recallBlock += `*No stored patterns resonated above the ${(threshold * 100).toFixed(1)}% similarity threshold.*`;
    } else {
      for (const match of topMatches) {
        recallBlock += `- [Similarity: ${(match.similarity * 100).toFixed(1)}%] ${match.summary}\n`;
      }
    }

    return {
      content: [{ type: "text", text: recallBlock }],
      isError: false,
    };
  } catch (err) {
    debugLog(`[session_intuitive_recall] Failed: ${err instanceof Error ? err.message : String(err)}`);
    return {
      content: [{ type: "text", text: `Error triggering Intuitive Recall: ${err instanceof Error ? err.message : String(err)}` }],
      isError: true,
    };
  }
}
