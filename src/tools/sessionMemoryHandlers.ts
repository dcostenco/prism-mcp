import { supabasePost, supabaseRpc, supabaseDelete, supabaseGet } from "../utils/supabaseApi.js";
import { toKeywordArray } from "../utils/keywordExtractor.js";
import {
  isSessionSaveLedgerArgs,
  isSessionSaveHandoffArgs,
  isSessionLoadContextArgs,
  isKnowledgeSearchArgs,
  isKnowledgeForgetArgs,
} from "./sessionMemoryDefinitions.js";

/**
 * Session Memory Handlers
 *
 * These handlers implement the actual logic for each session memory tool.
 * They follow the same pattern as the other MCP tool handlers in handlers.ts:
 *   1. Validate the incoming arguments using a type guard
 *   2. Call the Supabase API (via supabaseApi.ts)
 *   3. Return a formatted MCP response
 *
 * The three tools work together as a session lifecycle:
 *   session_load_context  → called at session START (read previous state)
 *   session_save_ledger   → called at session END (append immutable log)
 *   session_save_handoff  → called at session END (upsert live state for next boot)
 */

// ─── Save Ledger Handler ──────────────────────────────────────

/**
 * Appends an immutable session log entry.
 *
 * Think of the ledger as a "commit log" for agent work — once written, entries
 * are never modified. This creates a permanent audit trail of all work done.
 *
 * The ledger table grows over time. See the README maintenance guide for
 * cleanup instructions if you need to prune old entries.
 */
export async function sessionSaveLedgerHandler(args: unknown) {
  if (!isSessionSaveLedgerArgs(args)) {
    throw new Error("Invalid arguments for session_save_ledger");
  }

  const { project, conversation_id, summary, todos, files_changed, decisions } = args;

  console.error(`[session_save_ledger] Saving ledger entry for project="${project}"`);

  // Auto-extract keywords from summary + decisions for knowledge accumulation
  const combinedText = [summary, ...(decisions || [])].join(" ");
  const keywords = toKeywordArray(combinedText);
  console.error(`[session_save_ledger] Extracted ${keywords.length} keywords: ${keywords.slice(0, 5).join(", ")}...`);

  // Build the record to insert into the session_ledger table
  const record = {
    project,
    conversation_id,
    summary,
    todos: todos || [],
    files_changed: files_changed || [],
    decisions: decisions || [],
    keywords,
  };

  const result = await supabasePost("session_ledger", record);

  // Return a human-readable confirmation with key stats
  return {
    content: [{
      type: "text",
      text: `✅ Session ledger saved for project "${project}"\n` +
        `Summary: ${summary}\n` +
        (todos?.length ? `TODOs: ${todos.length} items\n` : "") +
        (files_changed?.length ? `Files changed: ${files_changed.length}\n` : "") +
        (decisions?.length ? `Decisions: ${decisions.length}\n` : "") +
        `\nRaw response: ${JSON.stringify(result)}`,
    }],
    isError: false,
  };
}

// ─── Save Handoff Handler ─────────────────────────────────────

/**
 * Upserts (insert-or-update) the latest project handoff state.
 *
 * Unlike the ledger, the handoff table keeps only ONE row per project.
 * Each call replaces the previous handoff for the same project.
 *
 * This is the "live" state that gets loaded when a new session starts
 * with session_load_context.
 *
 * The upsert is done using PostgREST's "merge-duplicates" conflict resolution
 * on the "project" column (which has a UNIQUE constraint).
 */
export async function sessionSaveHandoffHandler(args: unknown) {
  if (!isSessionSaveHandoffArgs(args)) {
    throw new Error("Invalid arguments for session_save_handoff");
  }

  const { project, open_todos, active_branch, last_summary, key_context } = args;

  console.error(`[session_save_handoff] Upserting handoff for project="${project}"`);

  // Auto-extract keywords from summary + context for knowledge accumulation
  const combinedText = [last_summary || "", key_context || ""].filter(Boolean).join(" ");
  const keywords = combinedText ? toKeywordArray(combinedText) : undefined;
  if (keywords) {
    console.error(`[session_save_handoff] Extracted ${keywords.length} keywords: ${keywords.slice(0, 5).join(", ")}...`);
  }

  // Only include fields that were actually provided (avoids overwriting with nulls)
  const record: Record<string, unknown> = { project };
  if (open_todos !== undefined) record.open_todos = open_todos;
  if (active_branch !== undefined) record.active_branch = active_branch;
  if (last_summary !== undefined) record.last_summary = last_summary;
  if (key_context !== undefined) record.key_context = key_context;
  if (keywords !== undefined) record.keywords = keywords;

  // Use PostgREST upsert: on_conflict=project tells it which column has the UNIQUE constraint
  // "resolution=merge-duplicates" merges the new data into existing rows instead of erroring
  const result = await supabasePost(
    "session_handoffs",
    record,
    { on_conflict: "project" },
    { "Prefer": "return=representation,resolution=merge-duplicates" }
  );

  return {
    content: [{
      type: "text",
      text: `✅ Handoff saved for project "${project}"\n` +
        (last_summary ? `Last summary: ${last_summary}\n` : "") +
        (open_todos?.length ? `Open TODOs: ${open_todos.length} items\n` : "") +
        (active_branch ? `Active branch: ${active_branch}\n` : "") +
        `\nRaw response: ${JSON.stringify(result)}`,
    }],
    isError: false,
  };
}

// ─── Load Context Handler ─────────────────────────────────────

/**
 * Loads session context for a project at the requested depth level.
 *
 * This calls the get_session_context() PostgreSQL function (RPC) which
 * returns different amounts of data depending on the level:
 *
 *   "quick"    — Just keywords and open TODOs (~50 tokens)
 *                Best for: quick check-ins, simple follow-up tasks
 *
 *   "standard" — Keywords + TODOs + last summary + active decisions (~200 tokens)
 *                Best for: normal work sessions (recommended default)
 *
 *   "deep"     — Everything above + last 5 full session logs (~1000+ tokens)
 *                Best for: recovering context after a long break, project audits
 */
export async function sessionLoadContextHandler(args: unknown) {
  if (!isSessionLoadContextArgs(args)) {
    throw new Error("Invalid arguments for session_load_context");
  }

  // Default to "standard" if no level specified — best balance of context vs token cost
  const { project, level = "standard" } = args;

  // Validate the level before making the API call
  const validLevels = ["quick", "standard", "deep"];
  if (!validLevels.includes(level)) {
    return {
      content: [{
        type: "text",
        text: `Invalid level "${level}". Must be one of: ${validLevels.join(", ")}`,
      }],
      isError: true,
    };
  }

  console.error(`[session_load_context] Loading ${level} context for project="${project}"`);

  // Call the PostgreSQL RPC function. The function handles all the logic for
  // which fields to include based on the level parameter.
  const result = await supabaseRpc("get_session_context", {
    p_project: project,
    p_level: level,
  });

  // The RPC returns a JSONB object. Handle the case where no data exists yet.
  const data = Array.isArray(result) ? result[0] : result;

  if (!data) {
    return {
      content: [{
        type: "text",
        text: `No session context found for project "${project}" at level ${level}.\n` +
          `This project has no previous session history. Starting fresh.`,
      }],
      isError: false,
    };
  }

  return {
    content: [{
      type: "text",
      text: `📋 Session context for "${project}" (${level}):\n\n${JSON.stringify(data, null, 2)}`,
    }],
    isError: false,
  };
}

// ─── Knowledge Search Handler ─────────────────────────────────

/**
 * Searches accumulated knowledge across all past sessions.
 *
 * This is the "brain query" tool — it searches keywords that were
 * automatically extracted from every saved ledger and handoff entry.
 * Results are ranked by relevance (keyword overlap + full-text match).
 */
export async function knowledgeSearchHandler(args: unknown) {
  if (!isKnowledgeSearchArgs(args)) {
    throw new Error("Invalid arguments for knowledge_search");
  }

  const { project, query, category, limit = 10 } = args;

  console.error(`[knowledge_search] Searching: project=${project || "all"}, query="${query || ""}", category=${category || "any"}, limit=${limit}`);

  // Extract keywords from the query text to use in array-overlap search
  const searchKeywords = query ? toKeywordArray(query) : [];

  const result = await supabaseRpc("search_knowledge", {
    p_project: project || null,
    p_keywords: searchKeywords,
    p_category: category || null,
    p_query_text: query || null,
    p_limit: Math.min(limit, 50),
  });

  const data = Array.isArray(result) ? result[0] : result;

  if (!data || !data.results || data.count === 0) {
    return {
      content: [{
        type: "text",
        text: `🔍 No knowledge found matching your search.\n` +
          (query ? `Query: "${query}"\n` : "") +
          (category ? `Category: ${category}\n` : "") +
          (project ? `Project: ${project}\n` : "") +
          `\nTip: Knowledge accumulates as sessions are saved. Try broader search terms.`,
      }],
      isError: false,
    };
  }

  return {
    content: [{
      type: "text",
      text: `🧠 Found ${data.count} knowledge entries:\n\n${JSON.stringify(data, null, 2)}`,
    }],
    isError: false,
  };
}

// ─── Knowledge Forget Handler ─────────────────────────────────

/**
 * Selectively forget (delete) accumulated knowledge entries.
 *
 * Like a brain pruning bad memories — removes outdated, incorrect,
 * or irrelevant session data to keep the knowledge base clean.
 *
 * Supports multiple forget modes:
 *   - By project: clear all entries for a project
 *   - By category: remove entries matching a specific category
 *   - By age: forget entries older than N days
 *   - Full wipe: clear everything (requires confirm_all flag)
 *   - Dry run: preview what would be deleted without deleting
 */
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

  // Safety: require either a project filter or explicit confirm_all
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

  console.error(`[knowledge_forget] ${dry_run ? "DRY RUN: " : ""}Forgetting: ` +
    `project=${project || "ALL"}, category=${category || "any"}, ` +
    `older_than=${older_than_days || "any"}d, clear_handoff=${clear_handoff}`);

  // Build PostgREST filter params for the ledger DELETE
  const ledgerParams: Record<string, string> = {};
  if (project) {
    ledgerParams.project = `eq.${project}`;
  }
  if (category) {
    // Filter entries that have the "cat:<category>" keyword
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
    // Dry run: count what would be deleted using GET with the same filters
    const selectParams = { ...ledgerParams, select: "id" };
    const entries = await supabaseGet("session_ledger", selectParams);
    ledgerCount = Array.isArray(entries) ? entries.length : 0;
  } else {
    // Actually delete ledger entries
    const result = await supabaseDelete("session_ledger", ledgerParams);
    ledgerCount = Array.isArray(result) ? result.length : 0;

    // Optionally clear the handoff for this project
    if (clear_handoff && project) {
      await supabaseDelete("session_handoffs", { project: `eq.${project}` });
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

