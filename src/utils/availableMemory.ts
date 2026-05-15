/**
 * Cross-platform "available memory" reader.
 * ─────────────────────────────────────────────────────────────
 * On Linux, `os.freemem()` returns MemFree which already excludes
 * cached pages — accurate. On macOS, `os.freemem()` returns only
 * "wired+free" pages and ignores inactive/speculative/purgeable
 * pages that the OS will reclaim on demand. That makes a fresh
 * MacBook with 48 GB RAM look like it has 2 GB free shortly after
 * any large process loads, which is wrong.
 *
 * We synthesise a Linux-style `MemAvailable` by parsing `vm_stat`
 * on darwin: free + inactive + speculative + (some) purgeable.
 * Falls back to `os.freemem()` on any error.
 *
 * Cached for 1 second to avoid spawning vm_stat per inference call.
 */

import * as os from "os";
import { execSync } from "child_process";

let cache: { ts: number; bytes: number } | null = null;
const TTL_MS = 1000;

const PAGE_SIZE = 16_384; // darwin default; we read it from vm_stat header when present

/** Returns free + inactive + speculative + purgeable in bytes on darwin. */
function darwinAvailable(): number | null {
    try {
        const out = execSync("vm_stat", { encoding: "utf8", timeout: 1000 });
        // Parse page size from "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
        const pageMatch = out.match(/page size of (\d+) bytes/);
        const pageSize = pageMatch ? parseInt(pageMatch[1], 10) : PAGE_SIZE;
        let pages = 0;
        const keys = [
            /Pages free:\s+(\d+)/,
            /Pages inactive:\s+(\d+)/,
            /Pages speculative:\s+(\d+)/,
            /Pages purgeable:\s+(\d+)/,
        ];
        for (const re of keys) {
            const m = out.match(re);
            if (m) pages += parseInt(m[1], 10);
        }
        if (pages <= 0) return null;
        return pages * pageSize;
    } catch {
        return null;
    }
}

/**
 * Returns "available memory" in bytes — analogous to Linux MemAvailable.
 * On darwin synthesises via vm_stat. On other platforms falls back to
 * os.freemem(). Cached for 1 s.
 */
export function getAvailableMemoryBytes(): number {
    const now = Date.now();
    if (cache && now - cache.ts < TTL_MS) return cache.bytes;

    let bytes: number;
    if (process.platform === "darwin") {
        bytes = darwinAvailable() ?? os.freemem();
    } else {
        bytes = os.freemem();
    }
    cache = { ts: now, bytes };
    return bytes;
}

/** Test-only. */
export function _resetAvailableMemoryCacheForTest(): void {
    cache = null;
}
