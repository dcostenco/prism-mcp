import {
  PRISM_STORAGE as ENV_PRISM_STORAGE,
  SUPABASE_CONFIGURED,
  SYNALUX_CONFIGURED,
  PRISM_FORCE_LOCAL,
} from "../config.js";
import { debugLog } from "../utils/logger.js";
import { SupabaseStorage } from "./supabase.js";
import type { StorageBackend } from "./interface.js";
import { getSetting } from "./configStorage.js";

/** Validates a URL is a valid http(s) Supabase endpoint. */
function isValidSupabaseUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

let storageInstance: StorageBackend | null = null;
export let activeStorageBackend: string = "local";

/**
 * Returns the singleton storage backend.
 *
 * On first call: creates and initializes the appropriate backend.
 * On subsequent calls: returns the cached instance.
 */
export async function getStorage(): Promise<StorageBackend> {
  if (storageInstance) return storageInstance;

  // Use environment variable if explicitly set, otherwise fall back to db config
  const envStorage = process.env.PRISM_STORAGE as "supabase" | "synalux" | "local" | "auto" | undefined;
  let requestedBackend = (envStorage || await getSetting("PRISM_STORAGE", ENV_PRISM_STORAGE)) as "supabase" | "synalux" | "local" | "auto";

  // ─── v13: PRISM_FORCE_LOCAL hard override ─────────────────────
  // Used by free tier and HIPAA deployments that must never reach the
  // network for memory operations. Wins over every other resolution.
  if (PRISM_FORCE_LOCAL) {
    requestedBackend = "local";
    debugLog("[Prism Storage] PRISM_FORCE_LOCAL=true — forcing local SQLite");
  }

  // ─── v13: Auto-resolve "auto" → synalux > supabase > local ────
  // Synalux portal is the paid-tier default — it mediates project
  // validation, tier gating, and audit. Direct Supabase remains as a
  // legacy fallback for installs that have not yet migrated to a portal
  // API key. Local SQLite is the floor.
  if (requestedBackend === "auto") {
    if (SYNALUX_CONFIGURED) {
      requestedBackend = "synalux";
      debugLog("[Prism Storage] Auto-resolved: synalux (portal credentials)");
    } else {
      const envUrl = process.env.SUPABASE_URL?.trim();
      const envKey = process.env.SUPABASE_KEY?.trim();
      if (envUrl && envKey && isValidSupabaseUrl(envUrl)) {
        requestedBackend = "supabase";
        debugLog("[Prism Storage] Auto-resolved: supabase (env vars, legacy direct-write path)");
      } else {
        const dashUrl = (await getSetting("SUPABASE_URL"))?.trim();
        const dashKey = (await getSetting("SUPABASE_KEY"))?.trim();
        if (dashUrl && dashKey && isValidSupabaseUrl(dashUrl)) {
          requestedBackend = "supabase";
          debugLog("[Prism Storage] Auto-resolved: supabase (dashboard config, legacy)");
        } else {
          requestedBackend = "local";
          debugLog("[Prism Storage] Auto-resolved: local (no portal or Supabase credentials)");
        }
      }
    }
  }

  // Guardrail: if Supabase is requested but env-var credentials are missing,
  // check the dashboard config DB (prism-config.db) as a fallback before
  // giving up. Dashboard settings take precedence over absent env vars.
  let supabaseReady = SUPABASE_CONFIGURED;
  if (!supabaseReady && requestedBackend === "supabase") {
    const dashUrl = await getSetting("SUPABASE_URL");
    const dashKey = await getSetting("SUPABASE_KEY");
    if (dashUrl && dashKey) {
      try {
        const parsed = new URL(dashUrl);
        if (parsed.protocol === "http:" || parsed.protocol === "https:") {
          supabaseReady = true;
          // Inject into process.env so downstream consumers (SupabaseStorage,
          // SyncBus) pick them up without needing their own dashboard lookups.
          process.env.SUPABASE_URL = dashUrl;
          process.env.SUPABASE_KEY = dashKey;
          debugLog("[Prism Storage] Using Supabase credentials from dashboard config");
        }
      } catch {
        // Invalid URL — fall through to local fallback
      }
    }
  }

  if (requestedBackend === "supabase" && !supabaseReady) {
    activeStorageBackend = "local";
    console.error(
      "[Prism Storage] Supabase backend requested but SUPABASE_URL/SUPABASE_KEY are invalid or unresolved. Falling back to local storage."
    );
  } else if (requestedBackend === "synalux" && !SYNALUX_CONFIGURED) {
    activeStorageBackend = "local";
    console.error(
      "[Prism Storage] Synalux backend requested but PRISM_SYNALUX_BASE_URL/PRISM_SYNALUX_API_KEY are missing or invalid. Falling back to local storage."
    );
  } else {
    activeStorageBackend = requestedBackend;
  }

  debugLog(`[Prism Storage] Initializing backend: ${activeStorageBackend}`);

  if (activeStorageBackend === "local") {
    const { SqliteStorage } = await import("./sqlite.js");
    storageInstance = new SqliteStorage();
  } else if (activeStorageBackend === "supabase") {
    storageInstance = new SupabaseStorage();
  } else if (activeStorageBackend === "synalux") {
    const { SynaluxStorage } = await import("./synalux.js");
    storageInstance = new SynaluxStorage();
  } else {
    throw new Error(
      `Unknown PRISM_STORAGE value: "${activeStorageBackend}". ` +
      `Must be "local", "supabase", or "synalux".`
    );
  }

  await storageInstance.initialize(activeStorageBackend === "local");

  // ─── v9.2.4: Cross-Backend Handoff Reconciliation ──────────────
  // When running on local SQLite but Supabase credentials exist,
  // pull any newer handoffs from Supabase into SQLite. This fixes
  // the split-brain where Claude Desktop writes go to Supabase but
  // Antigravity reads from SQLite and sees stale data.
  //
  // IMPORTANT: The supabaseReady check above only resolves dashboard
  // credentials when requestedBackend==="supabase". For reconciliation
  // we need credentials even when backend is "local", so we do a
  // second probe here.
  if (activeStorageBackend === "local") {
    let canReconcile = supabaseReady;
    if (!canReconcile) {
      // Probe dashboard config for Supabase credentials
      const dashUrl = await getSetting("SUPABASE_URL");
      const dashKey = await getSetting("SUPABASE_KEY");
      if (dashUrl && dashKey) {
        try {
          const parsed = new URL(dashUrl);
          if (parsed.protocol === "http:" || parsed.protocol === "https:") {
            canReconcile = true;
            process.env.SUPABASE_URL = dashUrl;
            process.env.SUPABASE_KEY = dashKey;
            debugLog("[Prism Storage] Reconciliation: using Supabase credentials from dashboard config");
          }
        } catch {
          // Invalid URL — skip reconciliation
        }
      }
    }

    if (canReconcile) {
      try {
        const { reconcileHandoffs } = await import("./reconcile.js");
        const { SqliteStorage } = await import("./sqlite.js");
        const sqliteInstance = storageInstance as InstanceType<typeof SqliteStorage>;
        const getTimestamps = () => sqliteInstance.getHandoffTimestamps();
        await reconcileHandoffs(storageInstance!, getTimestamps);
      } catch (err) {
        // Non-fatal: reconciliation is best-effort
        debugLog(`[Prism Storage] Reconciliation skipped: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  return storageInstance;
}

/**
 * Closes the active storage backend and resets the singleton.
 * Used for testing and graceful shutdown.
 */
export async function closeStorage(): Promise<void> {
  if (storageInstance) {
    await storageInstance.close();
    storageInstance = null;
  }
}

// Re-export the interface types for convenience
export type { StorageBackend } from "./interface.js";
export type {
  LedgerEntry,
  HandoffEntry,
  SaveHandoffResult,
  ContextResult,
  KnowledgeSearchResult,
  SemanticSearchResult,
  PipelineState,
  PipelineStatus,
} from "./interface.js";
