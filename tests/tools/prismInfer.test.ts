import { describe, it, expect, vi, beforeEach } from "vitest";
import { runInfer, type InferDeps, type PrismInferArgs } from "../../src/tools/prismInferHandler.js";

const GB = 1024 ** 3;

const INSTALLED_ALL = new Set([
    "prism-coder:32b",
    "prism-coder:14b",
    "prism-coder:8b",
    "prism-coder:1b7",
]);

function makeDeps(overrides: Partial<InferDeps>): InferDeps {
    return {
        freemem: () => 30 * GB,
        listTags: async () => INSTALLED_ALL,
        listLoaded: async () => new Set<string>(),
        callLocal: async () => ({ ok: false as const, reason: "default_mock_fail" }),
        callCloud: async () => ({ ok: false as const, reason: "default_mock_fail" }),
        ollamaUrl: "http://localhost:11434",
        ...overrides,
    };
}

function args(extra: Partial<PrismInferArgs> = {}): PrismInferArgs {
    return { prompt: "ping", ...extra };
}

describe("runInfer — local-first cascade", () => {
    it("hits 32B first on a high-RAM box", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "pong-32b" };
            },
        });
        const r = await runInfer(args(), deps);
        expect(r.backend).toBe("ollama-32b");
        expect(r.model_picked).toBe("prism-coder:32b");
        expect(r.output).toBe("pong-32b");
        expect(r.used_cloud).toBe(false);
        expect(calls).toEqual(["prism-coder:32b"]);
    });

    it("falls down to 14B when 32B fails", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            callLocal: async (_url, model) => {
                calls.push(model);
                if (model === "prism-coder:32b") return { ok: false as const, reason: "timeout" };
                return { ok: true as const, text: `pong-${model}` };
            },
        });
        const r = await runInfer(args(), deps);
        expect(calls).toEqual(["prism-coder:32b", "prism-coder:14b"]);
        expect(r.backend).toBe("ollama-14b");
    });

    it("honors model_ceiling — 14b on a 64GB box never tries 32B", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            freemem: () => 64 * GB,
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "pong" };
            },
        });
        const r = await runInfer(args({ model_ceiling: "14b" }), deps);
        expect(calls).toEqual(["prism-coder:14b"]);
        expect(r.model_picked).toBe("prism-coder:14b");
    });

    it("skips tiers not installed in Ollama", async () => {
        const calls: string[] = [];
        const partial = new Set(["prism-coder:8b", "prism-coder:1b7"]);
        const deps = makeDeps({
            freemem: () => 30 * GB,
            listTags: async () => partial,
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "pong" };
            },
        });
        const r = await runInfer(args(), deps);
        expect(calls).toEqual(["prism-coder:8b"]);
        expect(r.attempts).toContainEqual({ tier: "prism-coder:32b", reason: "not_pulled" });
        expect(r.attempts).toContainEqual({ tier: "prism-coder:14b", reason: "not_pulled" });
    });

    it("RAM gate: 8 GB free skips 32B and 14B", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            freemem: () => 8 * GB,
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "pong" };
            },
        });
        const r = await runInfer(args(), deps);
        expect(calls).toEqual(["prism-coder:8b"]);
        expect(r.model_picked).toBe("prism-coder:8b");
    });

    it("RAM gate: 2 GB free → no local pick, errors with cloud_fallback=false", async () => {
        const deps = makeDeps({
            freemem: () => 2 * GB,
            callLocal: vi.fn(),
        });
        await expect(runInfer(args(), deps)).rejects.toThrow(/no backend produced output/);
        expect(deps.callLocal).not.toHaveBeenCalled();
    });

    it("Ollama unreachable → goes straight to cloud when allowed", async () => {
        const cloudFn = vi.fn(async () => ({ ok: true as const, output: "from-cloud", backend: "ollama-14b" }));
        const deps = makeDeps({
            listTags: async () => null,
            callLocal: vi.fn(),
            callCloud: cloudFn,
        });
        const r = await runInfer(args({ cloud_fallback: true }), deps);
        expect(r.used_cloud).toBe(true);
        expect(r.output).toBe("from-cloud");
        expect(cloudFn).toHaveBeenCalledOnce();
        expect(deps.callLocal).not.toHaveBeenCalled();
    });

    it("all local fail + cloud_fallback=true → cloud answer returned", async () => {
        const cloudFn = vi.fn(async () => ({ ok: true as const, output: "from-claude", backend: "claude-opus-last-resort" }));
        const deps = makeDeps({
            callLocal: async () => ({ ok: false as const, reason: "timeout" }),
            callCloud: cloudFn,
        });
        const r = await runInfer(args({ cloud_fallback: true }), deps);
        expect(r.used_cloud).toBe(true);
        expect(r.backend).toBe("claude-opus-last-resort");
        expect(r.attempts.length).toBeGreaterThanOrEqual(4); // tried all 4 local tiers
    });

    it("all local fail + cloud_fallback=false → throws (token-saving default)", async () => {
        const deps = makeDeps({
            callLocal: async () => ({ ok: false as const, reason: "network" }),
            callCloud: vi.fn(),
        });
        await expect(runInfer(args(), deps)).rejects.toThrow();
        expect(deps.callCloud).not.toHaveBeenCalled();
    });

    it("cloud_fallback=true but cloud also fails → throws with full attempt log", async () => {
        const deps = makeDeps({
            callLocal: async () => ({ ok: false as const, reason: "timeout" }),
            callCloud: async () => ({ ok: false as const, reason: "synalux_http_503" }),
        });
        await expect(runInfer(args({ cloud_fallback: true }), deps)).rejects.toThrow(/synalux_http_503/);
    });
});

describe("runInfer — telemetry", () => {
    it("reports ram_free_mb in megabytes", async () => {
        const deps = makeDeps({
            freemem: () => 16 * GB,
            callLocal: async () => ({ ok: true as const, text: "ok" }),
        });
        const r = await runInfer(args(), deps);
        expect(r.ram_free_mb).toBe(16 * 1024);
    });

    it("latency_ms is non-negative", async () => {
        const deps = makeDeps({
            callLocal: async () => ({ ok: true as const, text: "ok" }),
        });
        const r = await runInfer(args(), deps);
        expect(r.latency_ms).toBeGreaterThanOrEqual(0);
    });
});

describe("runInfer — warm-model bypass", () => {
    it("uses already-loaded 32B even when freemem says insufficient", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            freemem: () => 2 * GB, // would normally block everything
            listLoaded: async () => new Set(["prism-coder:32b"]),
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "warm-32b" };
            },
        });
        const r = await runInfer(args(), deps);
        expect(calls).toEqual(["prism-coder:32b"]);
        expect(r.backend).toBe("ollama-32b");
        expect(r.output).toBe("warm-32b");
    });

    it("warm bypass respects model_ceiling", async () => {
        const calls: string[] = [];
        const deps = makeDeps({
            freemem: () => 2 * GB,
            listLoaded: async () => new Set(["prism-coder:32b", "prism-coder:14b"]),
            callLocal: async (_url, model) => {
                calls.push(model);
                return { ok: true as const, text: "ok" };
            },
        });
        // ceiling forbids 32B; should pick 14B even though both are warm
        const r = await runInfer(args({ model_ceiling: "14b" }), deps);
        expect(calls).toEqual(["prism-coder:14b"]);
        expect(r.model_picked).toBe("prism-coder:14b");
    });
});
