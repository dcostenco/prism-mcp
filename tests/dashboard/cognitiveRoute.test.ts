/**
 * Dashboard API: /api/graph/cognitive-route — Phase 6 Integration Tests (v6.5)
 *
 * ═══════════════════════════════════════════════════════════════════
 * SCOPE:
 *   Tests the HTTP API surface of the cognitive route endpoint through
 *   the handleGraphRoutes router. Exercises:
 *
 *   1. Response shape — ok, isError, text fields
 *   2. Missing params — 400 with clear error message
 *   3. Query param parsing — threshold numbers, explain boolean
 *   4. Error propagation — handler errors surface as 400/500
 *   5. Non-graph route passthrough — returns false
 *
 * APPROACH:
 *   Mock HTTP req/res objects against handleGraphRoutes, matching
 *   the existing dashboard/api.test.ts pattern for /api/graph/metrics.
 *   The handler itself is dynamically imported and its internals are
 *   mocked at the module level to isolate the router's HTTP logic.
 * ═══════════════════════════════════════════════════════════════════
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// ─── Mock the handler so we control its behavior ────────────────
const mockCognitiveRouteHandler = vi.fn(async () => ({
  content: [{ type: "text" as const, text: "🧠 Cognitive Route — project \"test\"\n\nRoute: ACTION_AUTO_ROUTE\nConcept: State:Test\nConfidence: 92.00%\n" }],
  isError: false,
}));

vi.mock("../../src/tools/graphHandlers.js", () => ({
  sessionCognitiveRouteHandler: mockCognitiveRouteHandler,
}));

// Mock metrics so the metrics endpoint still works
vi.mock("../../src/observability/graphMetrics.js", () => ({
  getGraphMetricsSnapshot: vi.fn(() => ({
    synthesis: {},
    testMe: {},
    scheduler: {},
    pruning: {},
    warnings: {},
    slo: {},
    cognitive: {},
  })),
  recordSynthesisRun: vi.fn(),
  resetGraphMetricsForTests: vi.fn(),
}));

// ─── Import the router ──────────────────────────────────────────
const { handleGraphRoutes } = await import("../../src/dashboard/graphRouter.js");

// ─── Helper: create mock HTTP req/res ───────────────────────────
function createMockHttp(urlString: string, method = "GET") {
  const url = new URL(urlString);
  const mockReq = { method } as any;

  let statusCode = 0;
  let headers: Record<string, string> = {};
  let body = "";

  const mockRes = {
    writeHead(code: number, hdrs: Record<string, string>) {
      statusCode = code;
      headers = hdrs;
    },
    end(data: string) {
      body = data;
    },
  } as any;

  const getStorageSafe = async () => ({});

  return {
    url, mockReq, mockRes, getStorageSafe,
    getStatus: () => statusCode,
    getHeaders: () => headers,
    getBody: () => body,
    getParsed: () => JSON.parse(body),
  };
}

// ═══════════════════════════════════════════════════════════════════
// 1. MISSING REQUIRED PARAMS — 400 response
// ═══════════════════════════════════════════════════════════════════

describe("GET /api/graph/cognitive-route — param validation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 400 when project is missing", async () => {
    const h = createMockHttp("http://localhost:3000/api/graph/cognitive-route?state=S&role=R&action=A");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
    expect(h.getParsed().error).toContain("required");
  });

  it("returns 400 when state is missing", async () => {
    const h = createMockHttp("http://localhost:3000/api/graph/cognitive-route?project=p&role=R&action=A");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
  });

  it("returns 400 when role is missing", async () => {
    const h = createMockHttp("http://localhost:3000/api/graph/cognitive-route?project=p&state=S&action=A");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
  });

  it("returns 400 when action is missing", async () => {
    const h = createMockHttp("http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
  });

  it("returns 400 when all params are missing", async () => {
    const h = createMockHttp("http://localhost:3000/api/graph/cognitive-route");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
    expect(h.getHeaders()["Content-Type"]).toBe("application/json");
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. HAPPY PATH — 200 response shape
// ═══════════════════════════════════════════════════════════════════

describe("GET /api/graph/cognitive-route — success response", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCognitiveRouteHandler.mockResolvedValue({
      content: [{ type: "text", text: "🧠 Cognitive Route — project \"test\"\nRoute: ACTION_AUTO_ROUTE" }],
      isError: false,
    });
  });

  it("returns 200 with expected JSON shape", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=State:Test&role=Role:dev&action=Action:inspect"
    );
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(200);
    expect(h.getHeaders()["Content-Type"]).toBe("application/json");

    const parsed = h.getParsed();
    expect(parsed.ok).toBe(true);
    expect(parsed.isError).toBe(false);
    expect(typeof parsed.text).toBe("string");
    expect(parsed.text).toContain("Cognitive Route");
  });

  it("passes params to handler as expected args shape", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?" +
      "project=my-proj&state=State:Active&role=Role:qa&action=Action:review&explain=true"
    );
    await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(mockCognitiveRouteHandler).toHaveBeenCalledWith(
      expect.objectContaining({
        project: "my-proj",
        state: "State:Active",
        role: "Role:qa",
        action: "Action:review",
        explain: true,
      })
    );
  });

  it("parses explain=false correctly", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A&explain=false"
    );
    await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(mockCognitiveRouteHandler).toHaveBeenCalledWith(
      expect.objectContaining({ explain: false })
    );
  });

  it("defaults explain to true when param is omitted", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A"
    );
    await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(mockCognitiveRouteHandler).toHaveBeenCalledWith(
      expect.objectContaining({ explain: true })
    );
  });

  it("parses numeric threshold overrides from query string", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?" +
      "project=p&state=S&role=R&action=A&fallback_threshold=0.4&clarify_threshold=0.7"
    );
    await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(mockCognitiveRouteHandler).toHaveBeenCalledWith(
      expect.objectContaining({
        fallback_threshold: 0.4,
        clarify_threshold: 0.7,
      })
    );
  });

  it("ignores non-numeric threshold values in query string", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?" +
      "project=p&state=S&role=R&action=A&fallback_threshold=abc"
    );
    await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    // Should NOT include fallback_threshold in args
    const call = mockCognitiveRouteHandler.mock.calls[0][0] as any;
    expect(call.fallback_threshold).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. ERROR RESPONSES — handler errors and exceptions
// ═══════════════════════════════════════════════════════════════════

describe("GET /api/graph/cognitive-route — error handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 400 when handler returns isError=true", async () => {
    mockCognitiveRouteHandler.mockResolvedValueOnce({
      content: [{ type: "text", text: "⚠️ disabled" }],
      isError: true,
    });

    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A"
    );
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(400);
    const parsed = h.getParsed();
    expect(parsed.ok).toBe(false);
    expect(parsed.isError).toBe(true);
    expect(parsed.text).toContain("disabled");
  });

  it("returns 500 when handler throws an exception", async () => {
    mockCognitiveRouteHandler.mockRejectedValueOnce(new Error("Internal SDM failure"));

    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A"
    );
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(500);
    const parsed = h.getParsed();
    expect(parsed.error).toContain("Internal SDM failure");
  });

  it("returns 500 with generic message for non-Error throws", async () => {
    mockCognitiveRouteHandler.mockRejectedValueOnce("string error");

    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A"
    );
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);

    expect(handled).toBe(true);
    expect(h.getStatus()).toBe(500);
    const parsed = h.getParsed();
    expect(parsed.error).toBe("Cognitive route failed");
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. ROUTE ISOLATION — non-cognitive routes are not handled
// ═══════════════════════════════════════════════════════════════════

describe("GET /api/graph/cognitive-route — route isolation", () => {
  it("returns false for unrelated routes", async () => {
    const h = createMockHttp("http://localhost:3000/api/settings");
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);
    expect(handled).toBe(false);
  });

  it("returns false for POST method on cognitive-route", async () => {
    const h = createMockHttp(
      "http://localhost:3000/api/graph/cognitive-route?project=p&state=S&role=R&action=A",
      "POST"
    );
    // POST should NOT be handled — the route only accepts GET
    const handled = await handleGraphRoutes(h.url, h.mockReq, h.mockRes, h.getStorageSafe);
    expect(handled).toBe(false);
  });
});
