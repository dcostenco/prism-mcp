/**
 * Task Router Tests (v7.1.0)
 *
 * Unit tests for the heuristic-based routing engine.
 * Tests cover: type guards, individual signals, composite routing,
 * cold-start/edge cases, and output payload structure.
 *
 * These tests are completely isolated — no database, no API calls.
 */

import { describe, it, expect, vi } from "vitest";

// Mock config to avoid pulling in the full dependency chain
vi.mock("../../src/config.js", () => ({
  PRISM_TASK_ROUTER_CONFIDENCE_THRESHOLD: 0.6,
  PRISM_TASK_ROUTER_MAX_CLAW_COMPLEXITY: 4,
}));

import { computeRoute, type TaskRouteResult } from "../../src/tools/taskRouterHandler.js";
import { isSessionTaskRouteArgs, type SessionTaskRouteArgs } from "../../src/tools/sessionMemoryDefinitions.js";

// ─── Type Guard Tests ────────────────────────────────────────

describe("isSessionTaskRouteArgs", () => {
  it("accepts valid minimal args", () => {
    expect(isSessionTaskRouteArgs({ task_description: "add a test" })).toBe(true);
  });

  it("accepts fully populated args", () => {
    expect(
      isSessionTaskRouteArgs({
        task_description: "scaffold a new component",
        files_involved: ["src/foo.ts", "src/bar.ts"],
        estimated_scope: "new_feature",
        project: "prism-mcp",
      })
    ).toBe(true);
  });

  it("rejects missing task_description", () => {
    expect(isSessionTaskRouteArgs({})).toBe(false);
    expect(isSessionTaskRouteArgs({ files_involved: ["a.ts"] })).toBe(false);
  });

  it("rejects non-string task_description", () => {
    expect(isSessionTaskRouteArgs({ task_description: 123 })).toBe(false);
    expect(isSessionTaskRouteArgs({ task_description: null })).toBe(false);
  });

  it("rejects invalid estimated_scope", () => {
    expect(
      isSessionTaskRouteArgs({ task_description: "do x", estimated_scope: "invalid" })
    ).toBe(false);
  });

  it("rejects non-array files_involved", () => {
    expect(
      isSessionTaskRouteArgs({ task_description: "do x", files_involved: "not-an-array" })
    ).toBe(false);
  });

  it("rejects files_involved with non-string items", () => {
    expect(
      isSessionTaskRouteArgs({ task_description: "do x", files_involved: [1, 2] })
    ).toBe(false);
  });

  it("rejects null and non-objects", () => {
    expect(isSessionTaskRouteArgs(null)).toBe(false);
    expect(isSessionTaskRouteArgs("a string")).toBe(false);
    expect(isSessionTaskRouteArgs(42)).toBe(false);
  });
});

// ─── Routing Result Shape Tests ──────────────────────────────

describe("computeRoute output shape", () => {
  it("returns all required fields", () => {
    const result = computeRoute({ task_description: "create a new file" });
    expect(result).toHaveProperty("target");
    expect(result).toHaveProperty("confidence");
    expect(result).toHaveProperty("complexity_score");
    expect(result).toHaveProperty("rationale");
    expect(result).toHaveProperty("recommended_tool");
  });

  it("target is either 'claw' or 'host'", () => {
    const result = computeRoute({ task_description: "fix typo in README" });
    expect(["claw", "host"]).toContain(result.target);
  });

  it("confidence is between 0 and 1", () => {
    const result = computeRoute({ task_description: "add a comment" });
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it("complexity_score is between 1 and 10", () => {
    const result = computeRoute({ task_description: "architect a microservice" });
    expect(result.complexity_score).toBeGreaterThanOrEqual(1);
    expect(result.complexity_score).toBeLessThanOrEqual(10);
  });

  it("recommended_tool is 'claw_run_task' when target is claw", () => {
    const result = computeRoute({
      task_description: "fix typo in the readme, simple, straightforward",
      estimated_scope: "minor_edit",
    });
    if (result.target === "claw") {
      expect(result.recommended_tool).toBe("claw_run_task");
    }
  });

  it("recommended_tool is null when target is host", () => {
    const result = computeRoute({
      task_description: "redesign the architecture of the entire system with a multi-step migration strategy",
      estimated_scope: "refactor",
    });
    if (result.target === "host") {
      expect(result.recommended_tool).toBeNull();
    }
  });
});

// ─── Routing Logic Tests ─────────────────────────────────────

describe("computeRoute routing logic", () => {
  it("routes simple file creation to claw", () => {
    const result = computeRoute({
      task_description: "create file for the new template stub",
      files_involved: ["src/template.ts"],
      estimated_scope: "minor_edit",
    });
    expect(result.target).toBe("claw");
    expect(result.complexity_score).toBeLessThanOrEqual(4);
  });

  it("routes typo fixes to claw", () => {
    const result = computeRoute({
      task_description: "fix typo in the config file, simple change",
      files_involved: ["src/config.ts"],
      estimated_scope: "minor_edit",
    });
    expect(result.target).toBe("claw");
  });

  it("routes architecture redesign to host", () => {
    const result = computeRoute({
      task_description: "redesign the architecture and implement a migration strategy for the database schema across multiple services",
      files_involved: ["src/db.ts", "src/models/", "src/api/", "src/migrations/", "src/services/", "tests/"],
      estimated_scope: "refactor",
    });
    expect(result.target).toBe("host");
    expect(result.complexity_score).toBeGreaterThan(4);
  });

  it("routes multi-step tasks to host", () => {
    const result = computeRoute({
      task_description: "First, refactor the handler. Second, update the tests. Third, update the documentation. Finally, update the changelog.",
    });
    expect(result.target).toBe("host");
  });

  it("routes complex debugging to host", () => {
    const result = computeRoute({
      task_description: "debug complex race condition in the concurrent request handler, investigate root cause and diagnose the issue",
      estimated_scope: "bug_fix",
    });
    expect(result.target).toBe("host");
  });

  it("routes simple test addition to claw", () => {
    const result = computeRoute({
      task_description: "add test for the new utility function, simple unit test",
      files_involved: ["tests/utils.test.ts"],
      estimated_scope: "minor_edit",
    });
    expect(result.target).toBe("claw");
  });

  it("routes security audit to host", () => {
    const result = computeRoute({
      task_description: "perform a security audit on the authentication module and analyze the vulnerability surface",
      estimated_scope: "refactor",
    });
    expect(result.target).toBe("host");
  });
});

// ─── Cold Start & Edge Cases ─────────────────────────────────

describe("computeRoute edge cases", () => {
  it("returns host with low confidence for empty-ish input", () => {
    const result = computeRoute({ task_description: "hi" });
    expect(result.target).toBe("host");
    expect(result.confidence).toBeLessThanOrEqual(0.6);
    expect(result.rationale).toContain("Insufficient");
  });

  it("returns host with low confidence for whitespace-only input", () => {
    const result = computeRoute({ task_description: "     " });
    expect(result.target).toBe("host");
    expect(result.rationale).toContain("Insufficient");
  });

  it("handles empty files_involved gracefully", () => {
    const result = computeRoute({
      task_description: "add a simple boilerplate template",
      files_involved: [],
    });
    // Should still route — file count signal is neutral (0)
    expect(["claw", "host"]).toContain(result.target);
  });

  it("handles no optional fields", () => {
    const result = computeRoute({ task_description: "do something with the code" });
    expect(["claw", "host"]).toContain(result.target);
    expect(result.rationale).toBeTruthy();
  });

  it("handles very long task descriptions", () => {
    const longDesc = "analyze ".repeat(500);
    const result = computeRoute({ task_description: longDesc });
    // Very long → host-favoring length signal
    expect(result.target).toBe("host");
  });
});

// ─── Scope Signal Tests ──────────────────────────────────────

describe("computeRoute scope influence", () => {
  const baseTask = "work on the codebase";

  it("minor_edit scope pushes toward claw", () => {
    const result = computeRoute({ task_description: baseTask, estimated_scope: "minor_edit" });
    const resultNone = computeRoute({ task_description: baseTask });
    // minor_edit should produce lower complexity than no scope
    expect(result.complexity_score).toBeLessThanOrEqual(resultNone.complexity_score);
  });

  it("refactor scope pushes toward host", () => {
    const result = computeRoute({ task_description: baseTask, estimated_scope: "refactor" });
    const resultNone = computeRoute({ task_description: baseTask });
    // refactor should produce higher complexity than no scope
    expect(result.complexity_score).toBeGreaterThanOrEqual(resultNone.complexity_score);
  });

  it("bug_fix scope stays moderate", () => {
    const result = computeRoute({ task_description: baseTask, estimated_scope: "bug_fix" });
    // bug_fix is moderate — not trivial, not maximum
    expect(result.complexity_score).toBeGreaterThanOrEqual(2);
    expect(result.complexity_score).toBeLessThanOrEqual(8);
  });
});
