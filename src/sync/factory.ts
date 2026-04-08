import { PRISM_STORAGE } from "../config.js";
import { debugLog } from "../utils/logger.js";
import type { SyncBus } from "./index.js";
import { getSetting } from "../storage/configStorage.js";

let _bus: SyncBus | null = null;

export async function getSyncBus(): Promise<SyncBus> {
  if (_bus) return _bus;

  if (PRISM_STORAGE === "local") {
    const { SqliteSyncBus } = await import("./sqliteSync.js");
    _bus = new SqliteSyncBus();
  } else {
    const { SupabaseSyncBus } = await import("./supabaseSync.js");
    // Check env vars first, then fall back to dashboard config (prism-config.db)
    const url = process.env.SUPABASE_URL || await getSetting("SUPABASE_URL");
    const key = process.env.SUPABASE_KEY || process.env.SUPABASE_ANON_KEY || await getSetting("SUPABASE_KEY");
    if (!url || !key) {
      debugLog(
        "[SyncBus] Supabase credentials not found in env or dashboard — falling back to local sync bus"
      );
      const { SqliteSyncBus } = await import("./sqliteSync.js");
      _bus = new SqliteSyncBus();
    } else {
      _bus = new SupabaseSyncBus(url, key);
    }
  }

  debugLog(`[SyncBus] Initialized: ${_bus.constructor.name} (client=${_bus.clientId.substring(0, 8)})`);
  return _bus;
}
