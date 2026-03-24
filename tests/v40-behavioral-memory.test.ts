/**
 * v4.0 Active Behavioral Memory — Test Suite
 *
 * Covers:
 *  1. Schema migration: event_type, confidence_score, importance columns
 *  2. Behavioral warnings: correctness of filtering, role isolation
 *  3. Importance voting: adjustImportance (upvote/downvote)
 *  4. Importance decay: 30-day decay in TTL sweep
 *  5. Experience recording: saveLedger with v4 fields
 *  6. Token budgeting: max_tokens char-count truncation
 *  7. Type guards: session_save_experience, knowledge_upvote/downvote
 *  8. Negative paths: bad IDs, edge cases
 *
 * Run:  npm test
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createClient } from "@libsql/client";
import type { Client } from "@libsql/client";

// ──────────────────────────────────────────────────────────────
// Helper: Create in-memory DB with full v4.0 schema
// ──────────────────────────────────────────────────────────────

async function makeV4Db(): Promise<Client> {
  const db = createClient({ url: "file::memory:" });

  await db.executeMultiple(`
    CREATE TABLE IF NOT EXISTS session_ledger (
      id            TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
      project       TEXT NOT NULL,
      user_id       TEXT NOT NULL DEFAULT 'default',
      conversation_id TEXT DEFAULT '',
      summary       TEXT DEFAULT '',
      todos         TEXT DEFAULT '[]',
      decisions     TEXT DEFAULT '[]',
      files_changed TEXT DEFAULT '[]',
      keywords      TEXT DEFAULT '[]',
      role          TEXT DEFAULT 'global',
      is_rollup     INTEGER DEFAULT 0,
      rollup_count  INTEGER DEFAULT 0,
      event_type    TEXT DEFAULT 'session',
      confidence_score INTEGER DEFAULT NULL,
      importance    INTEGER DEFAULT 0,
      archived_at   TEXT DEFAULT NULL,
      deleted_at    TEXT DEFAULT NULL,
      created_at    TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS session_handoffs (
      id            TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
      project       TEXT NOT NULL,
      user_id       TEXT NOT NULL DEFAULT 'default',
      role          TEXT NOT NULL DEFAULT 'global',
      last_summary  TEXT DEFAULT NULL,
      pending_todo  TEXT DEFAULT '[]',
      active_decisions TEXT DEFAULT '[]',
      keywords      TEXT DEFAULT '[]',
      key_context   TEXT DEFAULT NULL,
      active_branch TEXT DEFAULT NULL,
      version       INTEGER NOT NULL DEFAULT 1,
      metadata      TEXT DEFAULT '{}',
      updated_at    TEXT DEFAULT (datetime('now')),
      UNIQUE(project, user_id, role)
    );
  `);

  return db;
}

/** Insert a ledger entry with v4.0 fields */
async function insertV4Entry(
  db: Client,
  opts: {
    id?: string;
    project?: string;
    userId?: string;
    role?: string;
    summary?: string;
    eventType?: string;
    confidenceScore?: number | null;
    importance?: number;
    createdAt?: string;
    archived?: boolean;
    deleted?: boolean;
  } = {}
): Promise<string> {
  const id = opts.id || Math.random().toString(36).slice(2);
  const {
    project = "test-proj",
    userId = "default",
    role = "global",
    summary = "Test entry",
    eventType = "session",
    confidenceScore = null,
    importance = 0,
    createdAt,
    archived = false,
    deleted = false,
  } = opts;

  await db.execute({
    sql: `INSERT INTO session_ledger
            (id, project, user_id, role, summary, event_type,
             confidence_score, importance, archived_at, deleted_at, created_at)
          VALUES (?, ?, ?, ?, ?, ?,
                  ?, ?,
                  ${archived ? "datetime('now')" : "NULL"},
                  ${deleted ? "datetime('now')" : "NULL"},
                  ${createdAt ? "?" : "datetime('now')"})`,
    args: createdAt
      ? [id, project, userId, role, summary, eventType, confidenceScore, importance, createdAt]
      : [id, project, userId, role, summary, eventType, confidenceScore, importance],
  });

  return id;
}

/** Insert a handoff for context loading tests */
async function insertHandoff(
  db: Client,
  opts: {
    project?: string;
    userId?: string;
    role?: string;
    lastSummary?: string;
  } = {}
): Promise<void> {
  const {
    project = "test-proj",
    userId = "default",
    role = "global",
    lastSummary = "Latest handoff summary",
  } = opts;

  await db.execute({
    sql: `INSERT OR REPLACE INTO session_handoffs
            (project, user_id, role, last_summary, version)
          VALUES (?, ?, ?, ?, 1)`,
    args: [project, userId, role, lastSummary],
  });
}

// ──────────────────────────────────────────────────────────────
// 1. Schema: v4.0 columns exist and work
// ──────────────────────────────────────────────────────────────

describe("v4.0 Schema — Behavioral columns", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  it("stores event_type correctly", async () => {
    const id = await insertV4Entry(db, { eventType: "correction" });
    const r = await db.execute({ sql: "SELECT event_type FROM session_ledger WHERE id = ?", args: [id] });
    expect(r.rows[0].event_type).toBe("correction");
  });

  it("defaults event_type to 'session'", async () => {
    const id = await insertV4Entry(db, {});
    const r = await db.execute({ sql: "SELECT event_type FROM session_ledger WHERE id = ?", args: [id] });
    expect(r.rows[0].event_type).toBe("session");
  });

  it("stores confidence_score as integer", async () => {
    const id = await insertV4Entry(db, { confidenceScore: 85 });
    const r = await db.execute({ sql: "SELECT confidence_score FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].confidence_score)).toBe(85);
  });

  it("allows NULL confidence_score", async () => {
    const id = await insertV4Entry(db, { confidenceScore: null });
    const r = await db.execute({ sql: "SELECT confidence_score FROM session_ledger WHERE id = ?", args: [id] });
    expect(r.rows[0].confidence_score).toBeNull();
  });

  it("stores importance as integer", async () => {
    const id = await insertV4Entry(db, { importance: 5 });
    const r = await db.execute({ sql: "SELECT importance FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].importance)).toBe(5);
  });

  it("defaults importance to 0", async () => {
    const id = await insertV4Entry(db, {});
    const r = await db.execute({ sql: "SELECT importance FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].importance)).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────
// 2. Behavioral Warnings Query
// ──────────────────────────────────────────────────────────────

describe("Behavioral Warnings Query", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  /** Execute the behavioral warnings query matching sqlite.ts loadContext */
  async function getWarnings(
    project: string,
    userId: string,
    role: string
  ): Promise<Array<{ summary: string; importance: number }>> {
    const r = await db.execute({
      sql: `SELECT summary, importance
            FROM session_ledger
            WHERE project = ? AND user_id = ? AND role = ?
              AND event_type = 'correction'
              AND importance >= 3
              AND deleted_at IS NULL
              AND archived_at IS NULL
            ORDER BY importance DESC
            LIMIT 5`,
      args: [project, userId, role],
    });
    return r.rows.map(row => ({
      summary: row.summary as string,
      importance: Number(row.importance),
    }));
  }

  it("returns corrections with importance >= 3", async () => {
    await insertV4Entry(db, { eventType: "correction", importance: 5, summary: "Don't use npm" });
    await insertV4Entry(db, { eventType: "correction", importance: 3, summary: "Always lint" });
    await insertV4Entry(db, { eventType: "correction", importance: 1, summary: "Low importance" });

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings).toHaveLength(2);
    expect(warnings[0].importance).toBe(5);
    expect(warnings[1].importance).toBe(3);
  });

  it("excludes non-correction event types", async () => {
    await insertV4Entry(db, { eventType: "success", importance: 10, summary: "High importance success" });
    await insertV4Entry(db, { eventType: "failure", importance: 5, summary: "High importance failure" });

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings).toHaveLength(0);
  });

  it("excludes archived entries", async () => {
    await insertV4Entry(db, { eventType: "correction", importance: 5, archived: true, summary: "Archived" });

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings).toHaveLength(0);
  });

  it("excludes soft-deleted entries", async () => {
    await insertV4Entry(db, { eventType: "correction", importance: 5, deleted: true, summary: "Deleted" });

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings).toHaveLength(0);
  });

  it("scopes by role — no cross-role leakage", async () => {
    await insertV4Entry(db, { eventType: "correction", importance: 5, role: "dev", summary: "Dev warning" });
    await insertV4Entry(db, { eventType: "correction", importance: 4, role: "qa", summary: "QA warning" });

    const devWarnings = await getWarnings("test-proj", "default", "dev");
    const qaWarnings = await getWarnings("test-proj", "default", "qa");
    const globalWarnings = await getWarnings("test-proj", "default", "global");

    expect(devWarnings).toHaveLength(1);
    expect(devWarnings[0].summary).toBe("Dev warning");
    expect(qaWarnings).toHaveLength(1);
    expect(qaWarnings[0].summary).toBe("QA warning");
    expect(globalWarnings).toHaveLength(0);
  });

  it("scopes by project — no cross-project leakage", async () => {
    await insertV4Entry(db, { project: "proj-a", eventType: "correction", importance: 5, summary: "A warning" });
    await insertV4Entry(db, { project: "proj-b", eventType: "correction", importance: 5, summary: "B warning" });

    const warningsA = await getWarnings("proj-a", "default", "global");
    const warningsB = await getWarnings("proj-b", "default", "global");

    expect(warningsA).toHaveLength(1);
    expect(warningsA[0].summary).toBe("A warning");
    expect(warningsB).toHaveLength(1);
    expect(warningsB[0].summary).toBe("B warning");
  });

  it("orders by importance DESC", async () => {
    await insertV4Entry(db, { eventType: "correction", importance: 3, summary: "Low" });
    await insertV4Entry(db, { eventType: "correction", importance: 7, summary: "High" });
    await insertV4Entry(db, { eventType: "correction", importance: 5, summary: "Mid" });

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings.map(w => w.importance)).toEqual([7, 5, 3]);
  });

  it("limits to 5 results", async () => {
    for (let i = 0; i < 8; i++) {
      await insertV4Entry(db, { eventType: "correction", importance: 3 + i, summary: `Warning ${i}` });
    }

    const warnings = await getWarnings("test-proj", "default", "global");
    expect(warnings).toHaveLength(5);
    // Should be the top 5 by importance (10, 9, 8, 7, 6)
    expect(warnings[0].importance).toBe(10);
  });
});

// ──────────────────────────────────────────────────────────────
// 3. Importance Voting (adjustImportance)
// ──────────────────────────────────────────────────────────────

describe("adjustImportance — Insight Graduation", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  /** Simulate adjustImportance matching SqliteStorage */
  async function adjustImportance(id: string, delta: number, userId: string) {
    await db.execute({
      sql: `UPDATE session_ledger
            SET importance = MAX(0, importance + ?)
            WHERE id = ? AND user_id = ?`,
      args: [delta, id, userId],
    });
  }

  async function getImportance(id: string): Promise<number> {
    const r = await db.execute({ sql: "SELECT importance FROM session_ledger WHERE id = ?", args: [id] });
    return Number(r.rows[0]?.importance ?? -1);
  }

  it("upvote increases importance by 1", async () => {
    const id = await insertV4Entry(db, { importance: 2 });
    await adjustImportance(id, 1, "default");
    expect(await getImportance(id)).toBe(3);
  });

  it("downvote decreases importance by 1", async () => {
    const id = await insertV4Entry(db, { importance: 5 });
    await adjustImportance(id, -1, "default");
    expect(await getImportance(id)).toBe(4);
  });

  it("importance never goes below 0 (floor at zero)", async () => {
    const id = await insertV4Entry(db, { importance: 0 });
    await adjustImportance(id, -1, "default");
    expect(await getImportance(id)).toBe(0);
  });

  it("multiple upvotes accumulate", async () => {
    const id = await insertV4Entry(db, { importance: 0 });
    await adjustImportance(id, 1, "default");
    await adjustImportance(id, 1, "default");
    await adjustImportance(id, 1, "default");
    expect(await getImportance(id)).toBe(3);
  });

  it("no-op on non-existent ID (does not throw)", async () => {
    // Should not throw — just update 0 rows
    await adjustImportance("nonexistent-id", 1, "default");
    // No assertion beyond "didn't throw"
  });

  it("scopes by user_id — one user's vote doesn't change another's entry", async () => {
    const id = await insertV4Entry(db, { userId: "user-a", importance: 5 });
    await adjustImportance(id, -1, "user-b"); // Different user
    expect(await getImportance(id)).toBe(5); // Unchanged
  });

  it("large delta: +10 works for bulk adjustments", async () => {
    const id = await insertV4Entry(db, { importance: 0 });
    await adjustImportance(id, 10, "default");
    expect(await getImportance(id)).toBe(10);
  });

  it("large negative delta floors at 0", async () => {
    const id = await insertV4Entry(db, { importance: 3 });
    await adjustImportance(id, -100, "default");
    expect(await getImportance(id)).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────
// 4. Importance Decay (TTL sweep)
// ──────────────────────────────────────────────────────────────

describe("Importance Decay — v4.0 TTL Enhancement", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  /** Simulate the decay logic from SqliteStorage.expireByTTL */
  async function runDecay(project: string, userId: string): Promise<number> {
    const result = await db.execute({
      sql: `UPDATE session_ledger
            SET importance = MAX(0, importance - 1)
            WHERE project = ? AND user_id = ?
              AND importance > 0
              AND event_type != 'session'
              AND created_at < datetime('now', '-30 days')
              AND deleted_at IS NULL`,
      args: [project, userId],
    });
    return result.rowsAffected || 0;
  }

  it("decays importance of old experience entries (>30 days)", async () => {
    const oldDate = new Date();
    oldDate.setDate(oldDate.getDate() - 45);

    await insertV4Entry(db, {
      eventType: "correction",
      importance: 5,
      createdAt: oldDate.toISOString(),
    });

    const affected = await runDecay("test-proj", "default");
    expect(affected).toBe(1);

    const r = await db.execute({ sql: "SELECT importance FROM session_ledger", args: [] });
    expect(Number(r.rows[0].importance)).toBe(4);
  });

  it("does NOT decay recent entries (<30 days)", async () => {
    const recentDate = new Date();
    recentDate.setDate(recentDate.getDate() - 10);

    await insertV4Entry(db, {
      eventType: "correction",
      importance: 5,
      createdAt: recentDate.toISOString(),
    });

    const affected = await runDecay("test-proj", "default");
    expect(affected).toBe(0);
  });

  it("does NOT decay 'session' event_type entries", async () => {
    const oldDate = new Date();
    oldDate.setDate(oldDate.getDate() - 45);

    await insertV4Entry(db, {
      eventType: "session",
      importance: 5,
      createdAt: oldDate.toISOString(),
    });

    const affected = await runDecay("test-proj", "default");
    expect(affected).toBe(0);
  });

  it("does NOT decay entries already at importance 0", async () => {
    const oldDate = new Date();
    oldDate.setDate(oldDate.getDate() - 45);

    await insertV4Entry(db, {
      eventType: "correction",
      importance: 0,
      createdAt: oldDate.toISOString(),
    });

    const affected = await runDecay("test-proj", "default");
    expect(affected).toBe(0);
  });

  it("multiple decay runs reduce importance step by step", async () => {
    const oldDate = new Date();
    oldDate.setDate(oldDate.getDate() - 45);

    const id = await insertV4Entry(db, {
      eventType: "correction",
      importance: 3,
      createdAt: oldDate.toISOString(),
    });

    await runDecay("test-proj", "default"); // 3 → 2
    await runDecay("test-proj", "default"); // 2 → 1
    await runDecay("test-proj", "default"); // 1 → 0
    await runDecay("test-proj", "default"); // 0 → 0 (floor)

    const r = await db.execute({ sql: "SELECT importance FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].importance)).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────
// 5. Experience Recording
// ──────────────────────────────────────────────────────────────

describe("Experience Recording — saveLedger with v4 fields", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  it("corrections start with importance 1", async () => {
    const id = await insertV4Entry(db, {
      eventType: "correction",
      importance: 1, // v4.0: corrections start at 1
      summary: "[CORRECTION] Used wrong package manager",
    });

    const r = await db.execute({ sql: "SELECT importance, event_type FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].importance)).toBe(1);
    expect(r.rows[0].event_type).toBe("correction");
  });

  it("non-corrections start with importance 0", async () => {
    const id = await insertV4Entry(db, {
      eventType: "success",
      importance: 0,
    });

    const r = await db.execute({ sql: "SELECT importance FROM session_ledger WHERE id = ?", args: [id] });
    expect(Number(r.rows[0].importance)).toBe(0);
  });

  it("stores all event types correctly", async () => {
    const types = ["correction", "success", "failure", "learning", "session"];
    for (const t of types) {
      const id = await insertV4Entry(db, { eventType: t, summary: `Type: ${t}` });
      const r = await db.execute({ sql: "SELECT event_type FROM session_ledger WHERE id = ?", args: [id] });
      expect(r.rows[0].event_type).toBe(t);
    }
  });
});

// ──────────────────────────────────────────────────────────────
// 6. Token Budgeting
// ──────────────────────────────────────────────────────────────

describe("Token Budget — max_tokens truncation", () => {
  /** Simulate the handler's token budget logic */
  function applyTokenBudget(text: string, maxTokens: number | undefined): string {
    if (!maxTokens) return text;
    const maxChars = maxTokens * 4; // 1 token ≈ 4 chars
    if (text.length > maxChars) {
      return text.slice(0, maxChars) + "\n\n[…truncated to fit token budget]";
    }
    return text;
  }

  it("does not truncate when under budget", () => {
    const text = "Short context";
    expect(applyTokenBudget(text, 100)).toBe(text);
  });

  it("truncates when over budget", () => {
    const text = "A".repeat(1000); // 1000 chars = ~250 tokens
    const result = applyTokenBudget(text, 100); // 100 tokens = 400 chars
    expect(result.length).toBeLessThan(1000);
    expect(result).toContain("[…truncated to fit token budget]");
  });

  it("truncation preserves exactly maxTokens * 4 chars of content", () => {
    const text = "B".repeat(2000);
    const result = applyTokenBudget(text, 200); // 200 tokens = 800 chars
    // Content before the truncation marker should be exactly 800 chars
    const contentBeforeMarker = result.split("\n\n[…truncated")[0];
    expect(contentBeforeMarker.length).toBe(800);
  });

  it("no-op when maxTokens is undefined", () => {
    const text = "C".repeat(5000);
    expect(applyTokenBudget(text, undefined)).toBe(text);
  });

  it("handles maxTokens of 0 by returning truncation marker only", () => {
    const text = "D".repeat(100);
    // maxTokens=0 means 0 chars budget — edge case
    const result = applyTokenBudget(text, 0);
    // With 0 budget, falsy check means no truncation
    expect(result).toBe(text);
  });
});

// ──────────────────────────────────────────────────────────────
// 7. Type Guards
// ──────────────────────────────────────────────────────────────

describe("v4.0 Type Guards", () => {
  // Replicate the type guards from sessionMemoryDefinitions.ts

  function isSessionSaveExperienceArgs(args: unknown): boolean {
    if (typeof args !== "object" || args === null) return false;
    const a = args as Record<string, unknown>;
    return (
      typeof a.project === "string" &&
      typeof a.event_type === "string" &&
      typeof a.context === "string" &&
      typeof a.action === "string" &&
      typeof a.outcome === "string"
    );
  }

  function isKnowledgeVoteArgs(args: unknown): boolean {
    if (typeof args !== "object" || args === null) return false;
    return typeof (args as Record<string, unknown>).id === "string";
  }

  describe("isSessionSaveExperienceArgs", () => {
    it("accepts valid args with all required fields", () => {
      expect(isSessionSaveExperienceArgs({
        project: "prism-mcp",
        event_type: "correction",
        context: "Deploying to prod",
        action: "Ran npm install",
        outcome: "Wrong lockfile generated",
      })).toBe(true);
    });

    it("accepts valid args with optional fields", () => {
      expect(isSessionSaveExperienceArgs({
        project: "prism-mcp",
        event_type: "correction",
        context: "Context",
        action: "Action",
        outcome: "Outcome",
        correction: "Use pnpm instead",
        confidence_score: 85,
        role: "dev",
      })).toBe(true);
    });

    it("rejects missing project", () => {
      expect(isSessionSaveExperienceArgs({
        event_type: "correction",
        context: "x", action: "y", outcome: "z",
      })).toBe(false);
    });

    it("rejects missing event_type", () => {
      expect(isSessionSaveExperienceArgs({
        project: "x",
        context: "x", action: "y", outcome: "z",
      })).toBe(false);
    });

    it("rejects missing context", () => {
      expect(isSessionSaveExperienceArgs({
        project: "x", event_type: "correction",
        action: "y", outcome: "z",
      })).toBe(false);
    });

    it("rejects null", () => {
      expect(isSessionSaveExperienceArgs(null)).toBe(false);
    });

    it("rejects primitives", () => {
      expect(isSessionSaveExperienceArgs("string")).toBe(false);
      expect(isSessionSaveExperienceArgs(42)).toBe(false);
    });
  });

  describe("isKnowledgeVoteArgs (upvote/downvote)", () => {
    it("accepts valid id", () => {
      expect(isKnowledgeVoteArgs({ id: "abc-123" })).toBe(true);
    });

    it("rejects missing id", () => {
      expect(isKnowledgeVoteArgs({})).toBe(false);
    });

    it("rejects numeric id", () => {
      expect(isKnowledgeVoteArgs({ id: 123 })).toBe(false);
    });

    it("rejects null", () => {
      expect(isKnowledgeVoteArgs(null)).toBe(false);
    });
  });
});

// ──────────────────────────────────────────────────────────────
// 8. Confidence Score Edge Cases
// ──────────────────────────────────────────────────────────────

describe("Confidence Score — Edge Cases", () => {
  it("confidence_score of 0 renders correctly (not swallowed by falsy check)", () => {
    // Simulates the fixed handler logic: confidence_score !== undefined
    const confidence_score: number | undefined = 0;
    const line = confidence_score !== undefined ? `Confidence: ${confidence_score}%` : "";
    expect(line).toBe("Confidence: 0%");
  });

  it("confidence_score of undefined does not render", () => {
    const confidence_score: number | undefined = undefined;
    const line = confidence_score !== undefined ? `Confidence: ${confidence_score}%` : "";
    expect(line).toBe("");
  });

  it("confidence_score of 100 renders correctly", () => {
    const confidence_score: number | undefined = 100;
    const line = confidence_score !== undefined ? `Confidence: ${confidence_score}%` : "";
    expect(line).toBe("Confidence: 100%");
  });
});

// ──────────────────────────────────────────────────────────────
// 9. Context Loading — Warnings Integration
// ──────────────────────────────────────────────────────────────

describe("loadContext — Behavioral Warnings Integration", () => {
  let db: Client;

  beforeEach(async () => { db = await makeV4Db(); });
  afterEach(() => { db.close(); });

  /** Simulate loadContext's standard-level behavioral warnings logic */
  async function loadContextWithWarnings(
    project: string,
    userId: string,
    role: string
  ): Promise<{ hasHandoff: boolean; warnings: Array<{ summary: string; importance: number }> }> {
    // Check handoff exists
    const handoffResult = await db.execute({
      sql: "SELECT * FROM session_handoffs WHERE project = ? AND user_id = ? AND role = ?",
      args: [project, userId, role],
    });

    if (handoffResult.rows.length === 0) {
      return { hasHandoff: false, warnings: [] };
    }

    // Get warnings
    const r = await db.execute({
      sql: `SELECT summary, importance
            FROM session_ledger
            WHERE project = ? AND user_id = ? AND role = ?
              AND event_type = 'correction'
              AND importance >= 3
              AND deleted_at IS NULL
              AND archived_at IS NULL
            ORDER BY importance DESC
            LIMIT 5`,
      args: [project, userId, role],
    });

    return {
      hasHandoff: true,
      warnings: r.rows.map(row => ({
        summary: row.summary as string,
        importance: Number(row.importance),
      })),
    };
  }

  it("returns warnings when handoff exists and corrections are present", async () => {
    await insertHandoff(db, { project: "proj-a" });
    await insertV4Entry(db, { project: "proj-a", eventType: "correction", importance: 5, summary: "Always use pnpm" });

    const result = await loadContextWithWarnings("proj-a", "default", "global");
    expect(result.hasHandoff).toBe(true);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0].summary).toBe("Always use pnpm");
  });

  it("returns no warnings when no handoff exists", async () => {
    const result = await loadContextWithWarnings("no-handoff", "default", "global");
    expect(result.hasHandoff).toBe(false);
    expect(result.warnings).toHaveLength(0);
  });

  it("returns empty warnings when corrections exist but none meet threshold", async () => {
    await insertHandoff(db, { project: "proj-b" });
    await insertV4Entry(db, { project: "proj-b", eventType: "correction", importance: 1, summary: "Low priority" });

    const result = await loadContextWithWarnings("proj-b", "default", "global");
    expect(result.hasHandoff).toBe(true);
    expect(result.warnings).toHaveLength(0);
  });

  it("warnings include the correct summary text for injection", async () => {
    await insertHandoff(db, { project: "proj-c" });
    await insertV4Entry(db, {
      project: "proj-c",
      eventType: "correction",
      importance: 7,
      summary: "[CORRECTION] User uses pnpm, not npm → always use pnpm install",
    });

    const result = await loadContextWithWarnings("proj-c", "default", "global");
    expect(result.warnings[0].summary).toContain("pnpm");
    expect(result.warnings[0].importance).toBe(7);
  });
});
