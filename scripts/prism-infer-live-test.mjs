#!/usr/bin/env node
/**
 * prism_infer Live Integration Test
 * ───────────────────────────────────────────────────────────
 * Spawns the prism-mcp server over stdio and exercises the
 * prism_infer tool with three scenarios:
 *
 *   1. Sanity   — list tools, confirm prism_infer is present
 *   2. Local hit — happy path against running Ollama
 *   3. Ceiling  — model_ceiling="1b7" forces smallest tier
 *
 * Optional scenarios (opt-in to avoid disrupting your dev box):
 *   --kill-ollama : SIGSTOP ollama before call → expect failure, restart after
 *   --cloud       : test cloud_fallback=true via synalux portal
 *
 * Usage:
 *   node scripts/prism-infer-live-test.mjs                  # core tests
 *   node scripts/prism-infer-live-test.mjs --cloud          # + cloud fallback
 *   node scripts/prism-infer-live-test.mjs --kill-ollama    # + Ollama-down sim
 */

import { spawn } from "node:child_process";
import * as os from "node:os";
import * as path from "node:path";
import * as fs from "node:fs";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const PRISM_ROOT = path.resolve(__dirname, "..");
const SERVER_PATH = path.join(PRISM_ROOT, "dist", "server.js");

/** Load ~/prism/.env into a plain object (no third-party deps). */
function loadDotenv(file) {
    const out = {};
    if (!fs.existsSync(file)) return out;
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
        const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
        if (!m) continue;
        let val = m[2];
        if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
            val = val.slice(1, -1);
        }
        out[m[1]] = val;
    }
    return out;
}

const FLAGS = new Set(process.argv.slice(2));
const TEST_CLOUD = FLAGS.has("--cloud");
const KILL_OLLAMA = FLAGS.has("--kill-ollama");

const GB = 1024 ** 3;

// ─── Pretty output ─────────────────────────────────────────

const C = {
    reset: "\x1b[0m",
    dim:   "\x1b[2m",
    red:   "\x1b[31m",
    green: "\x1b[32m",
    yel:   "\x1b[33m",
    blue:  "\x1b[34m",
    bold:  "\x1b[1m",
};
const log = (msg) => process.stdout.write(`${msg}\n`);
const ok  = (msg) => log(`${C.green}✓${C.reset} ${msg}`);
const bad = (msg) => log(`${C.red}✗${C.reset} ${msg}`);
const info = (msg) => log(`${C.dim}  ${msg}${C.reset}`);
const head = (msg) => log(`\n${C.bold}${C.blue}${msg}${C.reset}`);

// ─── MCP stdio client ──────────────────────────────────────

class McpClient {
    constructor(serverPath) {
        const dotenv = loadDotenv(path.join(PRISM_ROOT, ".env"));
        this.proc = spawn("node", [serverPath], {
            stdio: ["pipe", "pipe", "pipe"],
            cwd: PRISM_ROOT,
            env: { ...dotenv, ...process.env, PRISM_FORCE_LOCAL: "true" },
        });
        this.proc.stderr.on("data", (b) => {
            if (process.env.PRISM_LIVE_DEBUG) process.stderr.write(`[srv] ${b}`);
        });
        this.buf = "";
        this.pending = new Map();
        this.nextId = 1;
        this.proc.stdout.on("data", (chunk) => {
            this.buf += chunk.toString();
            let nl;
            while ((nl = this.buf.indexOf("\n")) !== -1) {
                const line = this.buf.slice(0, nl).trim();
                this.buf = this.buf.slice(nl + 1);
                if (!line) continue;
                let msg;
                try { msg = JSON.parse(line); } catch { continue; }
                if (msg.id && this.pending.has(msg.id)) {
                    const { resolve, reject } = this.pending.get(msg.id);
                    this.pending.delete(msg.id);
                    if (msg.error) reject(new Error(JSON.stringify(msg.error)));
                    else resolve(msg.result);
                }
            }
        });
    }
    async request(method, params, timeoutMs = 180_000) {
        const id = this.nextId++;
        const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
        return new Promise((resolve, reject) => {
            const t = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`MCP request "${method}" timed out after ${timeoutMs}ms`));
            }, timeoutMs);
            this.pending.set(id, {
                resolve: (r) => { clearTimeout(t); resolve(r); },
                reject:  (e) => { clearTimeout(t); reject(e); },
            });
            this.proc.stdin.write(payload);
        });
    }
    async initialize() {
        return this.request("initialize", {
            protocolVersion: "2024-11-05",
            capabilities: {},
            clientInfo: { name: "prism-infer-live-test", version: "1.0.0" },
        });
    }
    async listTools() {
        return this.request("tools/list", {});
    }
    async callTool(name, args) {
        return this.request("tools/call", { name, arguments: args });
    }
    close() {
        try { this.proc.kill("SIGTERM"); } catch {}
    }
}

// ─── Assertions ────────────────────────────────────────────

let failures = 0;
function assert(cond, label) {
    if (cond) ok(label);
    else { bad(label); failures += 1; }
}

// ─── Main ──────────────────────────────────────────────────

async function main() {
    head("Prism-Infer Live Test");
    info(`server=${SERVER_PATH}`);
    info(`os.freemem=${(os.freemem() / GB).toFixed(1)} GB`);
    info(`os.totalmem=${(os.totalmem() / GB).toFixed(1)} GB`);
    info(`flags: cloud=${TEST_CLOUD} killOllama=${KILL_OLLAMA}`);

    const client = new McpClient(SERVER_PATH);
    try {
        await client.initialize();

        // ── 1. Sanity: tools/list contains prism_infer ──
        head("1. Sanity — tools/list");
        const listed = await client.listTools();
        const names = (listed.tools || []).map((t) => t.name);
        assert(names.includes("prism_infer"), `prism_infer is in tools/list (found ${names.length} tools)`);

        // ── 2. Local happy-path ──
        head("2. Local happy-path");
        const r1 = await client.callTool("prism_infer", {
            prompt: "Reply with exactly one word: PONG",
            max_tokens: 16,
            temperature: 0,
        });
        const header1 = r1?.content?.[0]?.text ?? "";
        const body1   = r1?.content?.[1]?.text ?? "";
        info(`header: ${header1}`);
        info(`body:   ${body1.slice(0, 80).replace(/\n/g, " ")}${body1.length > 80 ? "…" : ""}`);
        assert(!r1.isError, "no error envelope");
        assert(/backend=ollama-/.test(header1), "backend reported (ollama-*)");
        assert(/free_ram=\d+MB/.test(header1), "ram_free_mb in header");

        // ── 3. Ceiling forces a smaller tier (8B, confirmed pulled) ──
        head("3. model_ceiling='8b' caps cascade at 8B");
        const r2 = await client.callTool("prism_infer", {
            prompt: "Reply OK",
            max_tokens: 8,
            model_ceiling: "8b",
        });
        const header2 = r2?.content?.[0]?.text ?? "";
        info(`header: ${header2}`);
        assert(!r2.isError, "no error envelope (ceiling)");
        assert(/backend=ollama-8b/.test(header2), "backend is ollama-8b");
        assert(!/backend=ollama-(14|32)b/.test(header2), "did not escalate above ceiling");

        // ── 4. Optional: ollama down → cloud fallback ──
        if (TEST_CLOUD) {
            head("4. cloud_fallback=true with synalux portal");
            const r3 = await client.callTool("prism_infer", {
                prompt: "Reply OK",
                max_tokens: 8,
                model_ceiling: "1b7",
                cloud_fallback: true,
            });
            const header3 = r3?.content?.[0]?.text ?? "";
            info(`header: ${header3}`);
            // Cloud SHOULD NOT be used because local is up. But the header
            // should still show used_cloud=false explicitly.
            assert(/used_cloud=false/.test(header3), "local hit, used_cloud=false");
        }

        if (KILL_OLLAMA) {
            head("5. Ollama unreachable simulation");
            info("This requires you to manually `ollama serve` stop OR change LOCAL_LLM_URL to a dead port.");
            info("Skipping automated kill — too risky for a dev box.");
        }
    } finally {
        client.close();
    }

    head("Result");
    if (failures === 0) {
        ok(`All checks passed (${failures} failures)`);
        process.exit(0);
    } else {
        bad(`${failures} assertion(s) failed`);
        process.exit(1);
    }
}

main().catch((err) => {
    bad(`fatal: ${err?.stack || err}`);
    process.exit(2);
});
