/**
 * Graph Metrics Tests — Unit tests for the observability module
 *
 * ═══════════════════════════════════════════════════════════════════
 * Tests:
 *   1. Counter increment correctness
 *   2. Duration ring buffer bounds
 *   3. Warning flag computation (edge cases)
 *   4. Reset function clears all state
 *   5. Snapshot shape contract
 * ═══════════════════════════════════════════════════════════════════
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  recordSynthesisRun,
  recordTestMeRequest,
  recordSchedulerSynthesis,
  recordPruningRun,
  recordSweepDuration,
  recordCognitiveRoute,
  getGraphMetricsSnapshot,
  resetGraphMetricsForTests,
} from "../../src/observability/graphMetrics.js";

beforeEach(() => {
  resetGraphMetricsForTests();
});

// ═══════════════════════════════════════════════════════════════════
// 1. Counter Increments
// ═══════════════════════════════════════════════════════════════════

describe("Counter increments", () => {
  it("increments synthesis counters on success", () => {
    recordSynthesisRun({
      project: "test",
      status: "ok",
      duration_ms: 100,
      entries_scanned: 50,
      candidates: 120,
      below_threshold: 80,
      new_links: 15,
      skipped_links: 5,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.synthesis.runs_total).toBe(1);
    expect(snap.synthesis.runs_failed).toBe(0);
    expect(snap.synthesis.links_created_total).toBe(15);
    expect(snap.synthesis.candidates_evaluated_total).toBe(120);
    expect(snap.synthesis.below_threshold_total).toBe(80);
    expect(snap.synthesis.skipped_links_total).toBe(5);
    expect(snap.synthesis.last_status).toBe("ok");
    expect(snap.synthesis.last_links_created).toBe(15);
    expect(snap.synthesis.last_entries_scanned).toBe(50);
    expect(snap.synthesis.last_run_at).not.toBeNull();
  });

  it("increments failure counter on error", () => {
    recordSynthesisRun({
      project: "test",
      status: "error",
      duration_ms: 50,
      error: "test error",
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.synthesis.runs_total).toBe(1);
    expect(snap.synthesis.runs_failed).toBe(1);
    expect(snap.synthesis.last_status).toBe("error");
    // Error run should NOT increment link/candidate counters
    expect(snap.synthesis.links_created_total).toBe(0);
  });

  it("increments test-me counters by status", () => {
    recordTestMeRequest({ project: "p", node_id: "a", status: "success", duration_ms: 10 });
    recordTestMeRequest({ project: "p", node_id: "b", status: "no_api_key", duration_ms: 5 });
    recordTestMeRequest({ project: "p", node_id: "c", status: "generation_failed", duration_ms: 8 });
    recordTestMeRequest({ project: "p", node_id: "d", status: "bad_request", duration_ms: 2 });
    recordTestMeRequest({ project: "p", node_id: "e", status: "error", duration_ms: 3 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.testMe.requests_total).toBe(5);
    expect(snap.testMe.success_total).toBe(1);
    expect(snap.testMe.no_api_key_total).toBe(1);
    expect(snap.testMe.generation_failed_total).toBe(1);
    expect(snap.testMe.bad_request_total).toBe(1);
    // "error" only increments requests_total
  });

  it("accumulates across multiple runs", () => {
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100, new_links: 5, candidates: 30, below_threshold: 10, skipped_links: 2 });
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 200, new_links: 3, candidates: 20, below_threshold: 8, skipped_links: 1 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.synthesis.runs_total).toBe(2);
    expect(snap.synthesis.links_created_total).toBe(8);
    expect(snap.synthesis.candidates_evaluated_total).toBe(50);
    expect(snap.synthesis.below_threshold_total).toBe(18);
    expect(snap.synthesis.skipped_links_total).toBe(3);
    expect(snap.synthesis.last_links_created).toBe(3); // Last run's value
  });

  it("records scheduler synthesis data", () => {
    recordSchedulerSynthesis({
      projects_processed: 3,
      projects_succeeded: 2,
      projects_failed: 1,
      retries: 1,
      links_created: 12,
      duration_ms: 450,
      skipped_backpressure: 1,
      skipped_cooldown: 2,
      skipped_budget: 3,
      skipped_backoff: 4,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.scheduler.projects_processed_last).toBe(3);
    expect(snap.scheduler.projects_succeeded_last).toBe(2);
    expect(snap.scheduler.projects_failed_last).toBe(1);
    expect(snap.scheduler.retries_last).toBe(1);
    expect(snap.scheduler.links_created_last).toBe(12);
    expect(snap.scheduler.duration_ms_last).toBe(450);
    expect(snap.scheduler.skipped_backpressure_last).toBe(1);
    expect(snap.scheduler.skipped_cooldown_last).toBe(2);
    expect(snap.scheduler.skipped_budget_last).toBe(3);
    expect(snap.scheduler.skipped_backoff_last).toBe(4);
    expect(snap.scheduler.last_sweep_at).not.toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. Duration Ring Buffer
// ═══════════════════════════════════════════════════════════════════

describe("Duration buffer", () => {
  it("returns null p50 when no data", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.synthesis.duration_p50_ms).toBeNull();
    expect(snap.testMe.duration_p50_ms).toBeNull();
  });

  it("computes correct p50 for odd count", () => {
    // 3 values: [50, 100, 200] → p50 = 100
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 200 });
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 50 });
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.synthesis.duration_p50_ms).toBe(100);
  });

  it("computes correct p50 for even count", () => {
    // 4 values: [10, 20, 30, 40] → p50 = avg(20, 30) = 25
    recordTestMeRequest({ project: "p", node_id: "a", status: "success", duration_ms: 40 });
    recordTestMeRequest({ project: "p", node_id: "b", status: "success", duration_ms: 10 });
    recordTestMeRequest({ project: "p", node_id: "c", status: "success", duration_ms: 30 });
    recordTestMeRequest({ project: "p", node_id: "d", status: "success", duration_ms: 20 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.testMe.duration_p50_ms).toBe(25);
  });

  it("bounds at 100 entries (ring buffer wraps)", () => {
    // Push 110 entries — buffer should only keep last 100
    for (let i = 0; i < 110; i++) {
      recordSynthesisRun({ project: "p", status: "ok", duration_ms: i * 10 });
    }

    const snap = getGraphMetricsSnapshot();
    // runs_total should be 110 (counter is unbounded)
    expect(snap.synthesis.runs_total).toBe(110);
    // p50 should still be sane (not null)
    expect(snap.synthesis.duration_p50_ms).not.toBeNull();
    expect(typeof snap.synthesis.duration_p50_ms).toBe("number");
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. Warning Flags
// ═══════════════════════════════════════════════════════════════════

describe("Warning flags", () => {
  it("all warnings false on fresh state", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_quality_warning).toBe(false);
    expect(snap.warnings.testme_provider_warning).toBe(false);
    expect(snap.warnings.synthesis_failure_warning).toBe(false);
  });

  it("synthesis_quality_warning triggers when >85% below threshold with >=50 candidates", () => {
    // 60 candidates, 52 below threshold = 86.7%
    recordSynthesisRun({
      project: "p",
      status: "ok",
      duration_ms: 100,
      candidates: 60,
      below_threshold: 52,
    });

    let snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_quality_warning).toBe(true);
  });

  it("synthesis_quality_warning does NOT trigger under 50 candidates", () => {
    // 40 candidates, 38 below = 95%, but under threshold
    recordSynthesisRun({
      project: "p",
      status: "ok",
      duration_ms: 100,
      candidates: 40,
      below_threshold: 38,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_quality_warning).toBe(false);
  });

  it("synthesis_quality_warning does NOT trigger at exactly 85%", () => {
    // 100 candidates, 85 below = exactly 0.85 (not > 0.85)
    recordSynthesisRun({
      project: "p",
      status: "ok",
      duration_ms: 100,
      candidates: 100,
      below_threshold: 85,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_quality_warning).toBe(false);
  });

  it("testme_provider_warning triggers when no_api_key > 0 and success === 0", () => {
    recordTestMeRequest({ project: "p", node_id: "a", status: "no_api_key", duration_ms: 5 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.testme_provider_warning).toBe(true);
  });

  it("testme_provider_warning clears when success > 0", () => {
    recordTestMeRequest({ project: "p", node_id: "a", status: "no_api_key", duration_ms: 5 });
    recordTestMeRequest({ project: "p", node_id: "b", status: "success", duration_ms: 10 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.testme_provider_warning).toBe(false);
  });

  it("synthesis_failure_warning triggers when >20% failures with >=5 runs", () => {
    // 3 ok + 2 error = 40% failure rate
    for (let i = 0; i < 3; i++) {
      recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });
    }
    for (let i = 0; i < 2; i++) {
      recordSynthesisRun({ project: "p", status: "error", duration_ms: 50 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_failure_warning).toBe(true);
  });

  it("synthesis_failure_warning does NOT trigger under 5 runs", () => {
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });
    recordSynthesisRun({ project: "p", status: "error", duration_ms: 50 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.synthesis_failure_warning).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. Reset
// ═══════════════════════════════════════════════════════════════════

describe("resetGraphMetricsForTests", () => {
  it("clears all state back to initial", () => {
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100, new_links: 5 });
    recordTestMeRequest({ project: "p", node_id: "a", status: "success", duration_ms: 10 });
    recordSchedulerSynthesis({
      projects_processed: 2,
      projects_succeeded: 1,
      projects_failed: 1,
      retries: 1,
      links_created: 8,
      duration_ms: 300,
      skipped_backpressure: 0,
      skipped_cooldown: 0,
      skipped_budget: 0,
      skipped_backoff: 0,
    });

    resetGraphMetricsForTests();
    const snap = getGraphMetricsSnapshot();

    expect(snap.synthesis.runs_total).toBe(0);
    expect(snap.synthesis.links_created_total).toBe(0);
    expect(snap.synthesis.last_run_at).toBeNull();
    expect(snap.synthesis.duration_p50_ms).toBeNull();

    expect(snap.testMe.requests_total).toBe(0);
    expect(snap.testMe.last_run_at).toBeNull();

    expect(snap.scheduler.projects_processed_last).toBe(0);
    expect(snap.scheduler.last_sweep_at).toBeNull();

    expect(snap.warnings.synthesis_quality_warning).toBe(false);
    expect(snap.warnings.testme_provider_warning).toBe(false);
    expect(snap.warnings.synthesis_failure_warning).toBe(false);
    expect(snap.warnings.cognitive_fallback_rate_warning).toBe(false);
    expect(snap.warnings.cognitive_ambiguity_rate_warning).toBe(false);

    // Cognitive must reset
    expect(snap.cognitive.evaluations_total).toBe(0);
    expect(snap.cognitive.route_auto_total).toBe(0);
    expect(snap.cognitive.last_run_at).toBeNull();
    expect(snap.cognitive.last_route).toBeNull();
    expect(snap.cognitive.ambiguity_rate).toBeNull();
    expect(snap.cognitive.duration_p50_ms).toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. Snapshot Shape Contract
// ═══════════════════════════════════════════════════════════════════

describe("Snapshot shape contract", () => {
  it("always returns all top-level keys", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap).toHaveProperty("synthesis");
    expect(snap).toHaveProperty("testMe");
    expect(snap).toHaveProperty("scheduler");
    expect(snap).toHaveProperty("pruning");
    expect(snap).toHaveProperty("cognitive");
    expect(snap).toHaveProperty("slo");
    expect(snap).toHaveProperty("warnings");
  });

  it("synthesis has all required fields", () => {
    const snap = getGraphMetricsSnapshot();
    const s = snap.synthesis;
    expect(typeof s.runs_total).toBe("number");
    expect(typeof s.runs_failed).toBe("number");
    expect(typeof s.links_created_total).toBe("number");
    expect(typeof s.candidates_evaluated_total).toBe("number");
    expect(typeof s.below_threshold_total).toBe("number");
    expect(typeof s.skipped_links_total).toBe("number");
    expect(typeof s.last_links_created).toBe("number");
    expect(typeof s.last_entries_scanned).toBe("number");
    // Nullable fields
    expect(["string", "object"]).toContain(typeof s.last_run_at); // string or null
    expect(["string", "object"]).toContain(typeof s.last_status);
    expect(["number", "object"]).toContain(typeof s.duration_p50_ms);
  });

  it("testMe has all required fields", () => {
    const snap = getGraphMetricsSnapshot();
    const t = snap.testMe;
    expect(typeof t.requests_total).toBe("number");
    expect(typeof t.success_total).toBe("number");
    expect(typeof t.no_api_key_total).toBe("number");
    expect(typeof t.generation_failed_total).toBe("number");
    expect(typeof t.bad_request_total).toBe("number");
  });

  it("scheduler has all required fields", () => {
    const snap = getGraphMetricsSnapshot();
    const sc = snap.scheduler;
    expect(typeof sc.projects_processed_last).toBe("number");
    expect(typeof sc.projects_succeeded_last).toBe("number");
    expect(typeof sc.projects_failed_last).toBe("number");
    expect(typeof sc.retries_last).toBe("number");
    expect(typeof sc.links_created_last).toBe("number");
    expect(typeof sc.duration_ms_last).toBe("number");
    expect(typeof sc.skipped_backpressure_last).toBe("number");
    expect(typeof sc.skipped_cooldown_last).toBe("number");
    expect(typeof sc.skipped_budget_last).toBe("number");
    expect(typeof sc.skipped_backoff_last).toBe("number");
  });


  it("pruning has all required fields", () => {
    recordPruningRun({
      projects_considered: 3,
      projects_pruned: 2,
      links_scanned: 50,
      links_soft_pruned: 12,
      min_strength: 0.15,
      duration_ms: 120,
      skipped_backpressure: 1,
      skipped_cooldown: 1,
      skipped_budget: 0,
    });

    const snap = getGraphMetricsSnapshot();
    const p = snap.pruning;
    expect(typeof p.projects_considered_last).toBe("number");
    expect(typeof p.projects_pruned_last).toBe("number");
    expect(typeof p.links_scanned_last).toBe("number");
    expect(typeof p.links_soft_pruned_last).toBe("number");
    expect(typeof p.min_strength_last).toBe("number");
    expect(typeof p.duration_ms_last).toBe("number");
    expect(typeof p.skipped_backpressure_last).toBe("number");
    expect(typeof p.skipped_cooldown_last).toBe("number");
    expect(typeof p.skipped_budget_last).toBe("number");
    expect(["string", "object"]).toContain(typeof p.last_run_at);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. SLO Derivations (WS4)
// ═══════════════════════════════════════════════════════════════════

describe("SLO derivations", () => {
  it("synthesis_success_rate is null when no runs", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.synthesis_success_rate).toBeNull();
  });

  it("synthesis_success_rate computes correctly", () => {
    // 4 ok + 1 error = 80%
    for (let i = 0; i < 4; i++) {
      recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });
    }
    recordSynthesisRun({ project: "p", status: "error", duration_ms: 50 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.synthesis_success_rate).toBe(0.8);
  });

  it("synthesis_success_rate is 1.0 with all successes", () => {
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });
    recordSynthesisRun({ project: "p", status: "ok", duration_ms: 100 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.synthesis_success_rate).toBe(1.0);
  });

  it("net_new_links_last_sweep is positive when synthesis > prune", () => {
    recordSynthesisRun({
      project: "p",
      status: "ok",
      duration_ms: 100,
      new_links: 10,
    });
    recordPruningRun({
      projects_considered: 1,
      projects_pruned: 1,
      links_scanned: 20,
      links_soft_pruned: 3,
      min_strength: 0.15,
      duration_ms: 50,
      skipped_backpressure: 0,
      skipped_cooldown: 0,
      skipped_budget: 0,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.net_new_links_last_sweep).toBe(7); // 10 - 3
  });

  it("net_new_links_last_sweep is negative when prune > synthesis", () => {
    recordSynthesisRun({
      project: "p",
      status: "ok",
      duration_ms: 100,
      new_links: 2,
    });
    recordPruningRun({
      projects_considered: 1,
      projects_pruned: 1,
      links_scanned: 20,
      links_soft_pruned: 8,
      min_strength: 0.15,
      duration_ms: 50,
      skipped_backpressure: 0,
      skipped_cooldown: 0,
      skipped_budget: 0,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.net_new_links_last_sweep).toBe(-6); // 2 - 8
  });

  it("prune_ratio_last_sweep is 0 when no scans", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.prune_ratio_last_sweep).toBe(0);
  });

  it("prune_ratio_last_sweep computes correctly", () => {
    recordPruningRun({
      projects_considered: 1,
      projects_pruned: 1,
      links_scanned: 100,
      links_soft_pruned: 25,
      min_strength: 0.15,
      duration_ms: 50,
      skipped_backpressure: 0,
      skipped_cooldown: 0,
      skipped_budget: 0,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.prune_ratio_last_sweep).toBe(0.25);
  });

  it("scheduler_sweep_duration_ms_last defaults to 0", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.scheduler_sweep_duration_ms_last).toBe(0);
  });

  it("scheduler_sweep_duration_ms_last records via recordSweepDuration", () => {
    recordSweepDuration(1234);

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.scheduler_sweep_duration_ms_last).toBe(1234);
  });

  it("slo snapshot has all required fields", () => {
    const snap = getGraphMetricsSnapshot();
    const slo = snap.slo;
    expect(["number", "object"]).toContain(typeof slo.synthesis_success_rate); // number or null
    expect(typeof slo.net_new_links_last_sweep).toBe("number");
    expect(typeof slo.prune_ratio_last_sweep).toBe("number");
    expect(typeof slo.scheduler_sweep_duration_ms_last).toBe("number");
  });

  it("reset clears sweep duration state", () => {
    recordSweepDuration(5000);
    resetGraphMetricsForTests();

    const snap = getGraphMetricsSnapshot();
    expect(snap.slo.scheduler_sweep_duration_ms_last).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. Cognitive Routing Metrics (v6.5)
// ═══════════════════════════════════════════════════════════════════

describe("Cognitive routing metrics", () => {
  it("increments counters on AUTO route", () => {
    recordCognitiveRoute({
      project: "test",
      route: "ACTION_AUTO_ROUTE",
      concept: "State:ActiveSession",
      confidence: 0.97,
      distance: 3,
      ambiguous: false,
      steps: 2,
      duration_ms: 5,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.evaluations_total).toBe(1);
    expect(snap.cognitive.route_auto_total).toBe(1);
    expect(snap.cognitive.route_clarify_total).toBe(0);
    expect(snap.cognitive.route_fallback_total).toBe(0);
    expect(snap.cognitive.ambiguous_total).toBe(0);
    expect(snap.cognitive.last_route).toBe("ACTION_AUTO_ROUTE");
    expect(snap.cognitive.last_concept).toBe("State:ActiveSession");
    expect(snap.cognitive.last_confidence).toBe(0.97);
    expect(snap.cognitive.last_run_at).not.toBeNull();
  });

  it("increments fallback and ambiguity counters", () => {
    recordCognitiveRoute({
      project: "test",
      route: "ACTION_FALLBACK",
      concept: null,
      confidence: 0.3,
      distance: 12,
      ambiguous: true,
      steps: 5,
      duration_ms: 8,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.route_fallback_total).toBe(1);
    expect(snap.cognitive.ambiguous_total).toBe(1);
  });

  it("increments CLARIFY route", () => {
    recordCognitiveRoute({
      project: "test",
      route: "ACTION_CLARIFY",
      concept: "State:Reviewing",
      confidence: 0.88,
      distance: 5,
      ambiguous: true,
      steps: 3,
      duration_ms: 4,
    });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.route_clarify_total).toBe(1);
  });

  it("accumulates convergence steps across runs", () => {
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "A", confidence: 0.99, distance: 1, ambiguous: false, steps: 3, duration_ms: 2 });
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "B", confidence: 0.98, distance: 2, ambiguous: false, steps: 5, duration_ms: 3 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.convergence_steps_total).toBe(8);
    expect(snap.cognitive.median_convergence_steps).toBe(4); // 8/2 = 4.0
  });

  it("computes derived rates correctly", () => {
    // 3 auto, 1 clarify, 1 fallback, 2 ambiguous
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "X", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 1 });
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "Y", confidence: 0.98, distance: 2, ambiguous: false, steps: 1, duration_ms: 1 });
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "Z", confidence: 0.97, distance: 3, ambiguous: false, steps: 1, duration_ms: 1 });
    recordCognitiveRoute({ project: "p", route: "ACTION_CLARIFY", concept: "Q", confidence: 0.88, distance: 5, ambiguous: true, steps: 2, duration_ms: 2 });
    recordCognitiveRoute({ project: "p", route: "ACTION_FALLBACK", concept: null, confidence: 0.3, distance: 10, ambiguous: true, steps: 4, duration_ms: 5 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.ambiguity_rate).toBe(0.4); // 2/5
    expect(snap.cognitive.fallback_rate).toBe(0.2);  // 1/5
  });

  it("returns null rates when no evaluations", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.ambiguity_rate).toBeNull();
    expect(snap.cognitive.fallback_rate).toBeNull();
    expect(snap.cognitive.median_convergence_steps).toBeNull();
  });

  it("computes duration p50", () => {
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "A", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 10 });
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "B", confidence: 0.98, distance: 2, ambiguous: false, steps: 1, duration_ms: 30 });
    recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "C", confidence: 0.97, distance: 3, ambiguous: false, steps: 1, duration_ms: 20 });

    const snap = getGraphMetricsSnapshot();
    expect(snap.cognitive.duration_p50_ms).toBe(20);
  });

  it("snapshot has all cognitive fields", () => {
    const snap = getGraphMetricsSnapshot();
    const c = snap.cognitive;
    expect(typeof c.evaluations_total).toBe("number");
    expect(typeof c.route_auto_total).toBe("number");
    expect(typeof c.route_clarify_total).toBe("number");
    expect(typeof c.route_fallback_total).toBe("number");
    expect(typeof c.ambiguous_total).toBe("number");
    expect(typeof c.convergence_steps_total).toBe("number");
    expect(["number", "object"]).toContain(typeof c.median_convergence_steps);
    expect(["number", "object"]).toContain(typeof c.ambiguity_rate);
    expect(["number", "object"]).toContain(typeof c.fallback_rate);
    expect(["number", "object"]).toContain(typeof c.last_run_at);
    expect(["string", "object"]).toContain(typeof c.last_route);
    expect(["string", "object"]).toContain(typeof c.last_concept);
    expect(["number", "object"]).toContain(typeof c.last_confidence);
    expect(["number", "object"]).toContain(typeof c.duration_p50_ms);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8. Cognitive Warning Flags (v6.5)
// ═══════════════════════════════════════════════════════════════════

describe("Cognitive warning flags", () => {
  it("cognitive warnings false on fresh state", () => {
    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_fallback_rate_warning).toBe(false);
    expect(snap.warnings.cognitive_ambiguity_rate_warning).toBe(false);
  });

  it("cognitive warnings false under 10 evaluations", () => {
    // 5 fallback routes, but only 5 total — under the 10-evaluation minimum
    for (let i = 0; i < 5; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_FALLBACK", concept: null, confidence: 0.2, distance: 10, ambiguous: true, steps: 3, duration_ms: 5 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_fallback_rate_warning).toBe(false);
    expect(snap.warnings.cognitive_ambiguity_rate_warning).toBe(false);
  });

  it("cognitive_fallback_rate_warning triggers at >30% with >=10 evaluations", () => {
    // 6 auto + 4 fallback = 40% fallback rate
    for (let i = 0; i < 6; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "X", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 1 });
    }
    for (let i = 0; i < 4; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_FALLBACK", concept: null, confidence: 0.2, distance: 10, ambiguous: false, steps: 5, duration_ms: 5 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_fallback_rate_warning).toBe(true);
  });

  it("cognitive_fallback_rate_warning does NOT trigger at exactly 30%", () => {
    // 7 auto + 3 fallback = 30% — boundary case (> 0.3 not >= 0.3)
    for (let i = 0; i < 7; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "X", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 1 });
    }
    for (let i = 0; i < 3; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_FALLBACK", concept: null, confidence: 0.2, distance: 10, ambiguous: false, steps: 5, duration_ms: 5 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_fallback_rate_warning).toBe(false);
  });

  it("cognitive_ambiguity_rate_warning triggers at >40% with >=10 evaluations", () => {
    // 5 clear + 5 ambiguous = 50%
    for (let i = 0; i < 5; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "X", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 1 });
    }
    for (let i = 0; i < 5; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_CLARIFY", concept: "Y", confidence: 0.88, distance: 5, ambiguous: true, steps: 2, duration_ms: 3 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_ambiguity_rate_warning).toBe(true);
  });

  it("cognitive_ambiguity_rate_warning does NOT trigger at exactly 40%", () => {
    // 6 clear + 4 ambiguous = 40% — boundary
    for (let i = 0; i < 6; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_AUTO_ROUTE", concept: "X", confidence: 0.99, distance: 1, ambiguous: false, steps: 1, duration_ms: 1 });
    }
    for (let i = 0; i < 4; i++) {
      recordCognitiveRoute({ project: "p", route: "ACTION_CLARIFY", concept: "Y", confidence: 0.88, distance: 5, ambiguous: true, steps: 2, duration_ms: 3 });
    }

    const snap = getGraphMetricsSnapshot();
    expect(snap.warnings.cognitive_ambiguity_rate_warning).toBe(false);
  });
});
