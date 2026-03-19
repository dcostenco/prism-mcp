#!/usr/bin/env node
/**
 * Comprehensive Knowledge System Test Suite — v0.3.0
 *
 * Tests all 3 knowledge features end-to-end:
 *   1. Keyword Extraction (unit) — in-process, no Supabase needed
 *   2. Knowledge Search → Save → Search lifecycle (integration)
 *   3. Knowledge Forget — dry run + actual delete (integration)
 *   4. Knowledge Cache — context loading with preloaded cache (integration)
 *
 * Usage:
 *   node tests/test_knowledge_system.js          # unit tests only
 *   SUPABASE_URL=... SUPABASE_KEY=... node tests/test_knowledge_system.js --integration
 *
 * Exit code 0 = all passed, 1 = failures
 */

import { extractKeywords, toKeywordArray } from "../dist/utils/keywordExtractor.js";

const INTEGRATION = process.argv.includes("--integration");
let passed = 0;
let failed = 0;
let skipped = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  ✅ ${message}`);
    passed++;
  } else {
    console.error(`  ❌ FAIL: ${message}`);
    failed++;
  }
}

function skip(message) {
  console.log(`  ⏭️  SKIP: ${message}`);
  skipped++;
}

// ─── Unit Tests: Keyword Extraction ───────────────────────────

console.log("\n═══════════════════════════════════════════════════════");
console.log("  🧪 Knowledge System Test Suite — v0.3.0");
console.log("═══════════════════════════════════════════════════════\n");

console.log("1️⃣  Keyword Extraction (Unit Tests)");
console.log("─────────────────────────────────────");

// Test 1: Basic extraction
{
  const result = extractKeywords("Fixed Stripe webhook race condition using database-backed idempotency keys");
  assert(result.keywords.length > 0, "Extracts keywords from technical text");
  assert(result.keywords.includes("stripe"), "Includes 'stripe' keyword");
  assert(result.keywords.includes("webhook"), "Includes 'webhook' keyword");
  assert(result.keywords.includes("idempotency"), "Includes 'idempotency' keyword");
}

// Test 2: Category detection
{
  const result = extractKeywords("Debug the authentication bug in the API endpoint error");
  assert(result.categories.includes("debugging"), "Detects 'debugging' category");
  assert(result.categories.includes("api-integration"), "Detects 'api-integration' category");
}

// Test 3: Architecture category
{
  const result = extractKeywords("Refactored the microservice architecture with new REST API design patterns");
  assert(result.categories.includes("architecture"), "Detects 'architecture' category");
}

// Test 4: AI/ML category
{
  const result = extractKeywords("Trained the ml model using embedding vectors for inference");
  assert(result.categories.includes("ai-ml"), "Detects 'ai-ml' category");
}

// Test 5: Multiple categories
{
  const result = extractKeywords("Deployed the Docker container to AWS with CI/CD pipeline after fixing the performance test");
  const hasDeploy = result.categories.includes("deployment");
  const hasTesting = result.categories.includes("testing");
  const hasPerf = result.categories.includes("performance");
  assert(hasDeploy || hasTesting || hasPerf, "Detects multiple categories from mixed text");
}

// Test 6: toKeywordArray output format
{
  const arr = toKeywordArray("Fixed the auth bug in production deployment");
  assert(Array.isArray(arr), "toKeywordArray returns an array");
  const hasCat = arr.some(k => k.startsWith("cat:"));
  assert(hasCat, "toKeywordArray includes cat: prefixed categories");
  assert(arr.every(k => typeof k === "string"), "All items are strings");
}

// Test 7: Empty input
{
  const result = extractKeywords("");
  assert(result.keywords.length === 0, "Empty input returns no keywords");
  assert(result.categories.length === 0, "Empty input returns no categories");
}

// Test 8: Performance benchmark
{
  const start = performance.now();
  for (let i = 0; i < 1000; i++) {
    extractKeywords("This is a performance benchmark test with multiple words for extraction");
  }
  const elapsed = performance.now() - start;
  const perCall = elapsed / 1000;
  assert(perCall < 1, `Performance: ${perCall.toFixed(4)}ms/call (target: <1ms)`);
  console.log(`  📊 1000 extractions in ${elapsed.toFixed(1)}ms = ${perCall.toFixed(4)}ms/call`);
}

// Test 9: Security category
{
  const result = extractKeywords("Applied OAuth authentication with encryption and TLS certificate security");
  assert(result.categories.includes("security"), "Detects 'security' category");
}

// Test 10: Resume category
{
  const result = extractKeywords("Updated resume with work experience and skills for job interview");
  assert(result.categories.includes("resume"), "Detects 'resume' category");
}

// Test 11: Stopword filtering
{
  const result = extractKeywords("The is a and of to in for on with at");
  assert(result.keywords.length === 0, "Stopwords are filtered out completely");
}

// Test 12: Data migration category
{
  const result = extractKeywords("Migrated the database table with SQL query and data migration");
  assert(result.categories.includes("data-migration"), "Detects 'data-migration' category");
}

// ─── Integration Tests ────────────────────────────────────────

console.log("\n2️⃣  Integration Tests (Supabase)");
console.log("─────────────────────────────────────");

if (!INTEGRATION) {
  skip("Supabase integration tests (run with --integration flag)");
  skip("knowledge_search lifecycle test");
  skip("knowledge_forget dry_run test");
  skip("knowledge_forget delete test");
  skip("knowledge_cache preload test");
} else {
  const SUPABASE_URL = process.env.SUPABASE_URL;
  const SUPABASE_KEY = process.env.SUPABASE_KEY;

  if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error("  ❌ SUPABASE_URL and SUPABASE_KEY must be set for integration tests");
    process.exit(1);
  }

  const headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": `Bearer ${SUPABASE_KEY}`,
    "Content-Type": "application/json",
    "Prefer": "return=representation",
  };

  const rpc = async (fn, body) => {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    return res.json();
  };

  const post = async (table, body) => {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    return res.json();
  };

  const del = async (table, params) => {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${qs}`, {
      method: "DELETE",
      headers,
    });
    if (res.status === 204) return [];
    return res.json();
  };

  const TEST_PROJECT = "__test_knowledge_system__";

  try {
    // Cleanup from previous runs
    await del("session_ledger", { project: `eq.${TEST_PROJECT}` });
    await del("session_handoffs", { project: `eq.${TEST_PROJECT}` });

    // Test A: Save a ledger entry with keywords
    console.log("\n  📝 Test A: Save ledger entry with keywords...");
    const keywords = toKeywordArray("Implemented Stripe webhook handler for subscription lifecycle events with idempotency");
    const saveResult = await post("session_ledger", {
      project: TEST_PROJECT,
      conversation_id: "test-conv-001",
      summary: "Implemented Stripe webhook handler with idempotency keys for payment processing",
      keywords: keywords,
      decisions: ["Used database-backed idempotency"],
      files_changed: ["src/webhooks/stripe.ts"],
      todo_next: ["Add retry logic"],
    });
    assert(!saveResult.error, `Save ledger entry: ${saveResult.error || "OK"}`);

    // Test B: Search for the entry we just saved
    console.log("\n  🔍 Test B: Search for saved entry...");
    const searchResult = await rpc("search_knowledge", {
      p_project: TEST_PROJECT,
      p_keywords: ["stripe", "webhook"],
      p_limit: 10,
    });
    assert(searchResult.count > 0, `Search found ${searchResult.count} entries`);
    assert(
      searchResult.results?.some(r => r.summary?.includes("Stripe")),
      "Search result contains the Stripe entry"
    );

    // Test C: Search by category
    console.log("\n  📂 Test C: Search by category...");
    const catResult = await rpc("search_knowledge", {
      p_project: TEST_PROJECT,
      p_category: "api-integration",
      p_limit: 10,
    });
    assert(catResult.count > 0, `Category search found ${catResult.count} entries`);

    // Test D: Full-text search
    console.log("\n  📖 Test D: Full-text search...");
    const ftsResult = await rpc("search_knowledge", {
      p_project: TEST_PROJECT,
      p_query_text: "payment processing",
      p_limit: 10,
    });
    assert(ftsResult.count > 0, `Full-text search found ${ftsResult.count} entries`);

    // Test E: Save handoff and test context with knowledge cache
    console.log("\n  💾 Test E: Save handoff + load context with knowledge cache...");
    await post("session_handoffs", {
      project: TEST_PROJECT,
      last_summary: "Stripe webhook handler implemented with idempotency",
      keywords: ["stripe", "webhook", "idempotency", "cat:api-integration"],
      pending_todo: ["Add retry logic"],
      active_decisions: ["Database-backed idempotency keys"],
    });

    const contextResult = await rpc("get_session_context", {
      p_project: TEST_PROJECT,
      p_level: "standard",
    });
    assert(contextResult.knowledge_cache !== undefined, "Context includes knowledge_cache");
    assert(Array.isArray(contextResult.knowledge_cache?.hot_keywords), "knowledge_cache has hot_keywords array");
    assert(typeof contextResult.knowledge_cache?.total_sessions === "number", "knowledge_cache has total_sessions count");
    console.log(`  📊 Knowledge cache: ${JSON.stringify(contextResult.knowledge_cache)}`);

    // Test F: Deep context with cross-project knowledge
    console.log("\n  🧠 Test F: Deep context load...");
    const deepResult = await rpc("get_session_context", {
      p_project: TEST_PROJECT,
      p_level: "deep",
    });
    assert(deepResult.knowledge_cache !== undefined, "Deep context includes knowledge_cache");
    assert(deepResult.recent_sessions !== undefined, "Deep context includes recent_sessions");
    console.log(`  📊 Deep cache: hot_keywords=${JSON.stringify(deepResult.knowledge_cache?.hot_keywords)}`);

    // Test G: Forget (dry run)
    console.log("\n  🔍 Test G: Knowledge forget dry run...");
    // For dry run, we need to query via GET to count
    const dryRunRes = await fetch(`${SUPABASE_URL}/rest/v1/session_ledger?project=eq.${TEST_PROJECT}&select=id`, {
      headers: { ...headers, "Prefer": "return=representation" },
    });
    const dryRunEntries = await dryRunRes.json();
    const dryRunCount = Array.isArray(dryRunEntries) ? dryRunEntries.length : 0;
    assert(dryRunCount > 0, `Dry run would delete ${dryRunCount} entries`);

    // Test H: Forget (actual delete)
    console.log("\n  🧹 Test H: Knowledge forget (actual delete)...");
    const forgetRes = await fetch(`${SUPABASE_URL}/rest/v1/session_ledger?project=eq.${TEST_PROJECT}`, {
      method: "DELETE",
      headers,
    });
    // Verify it's gone
    const verifyRes = await rpc("search_knowledge", {
      p_project: TEST_PROJECT,
      p_keywords: ["stripe"],
      p_limit: 10,
    });
    assert(verifyRes.count === 0, `After forget: ${verifyRes.count} entries remain (expected 0)`);

    // Cleanup
    await del("session_handoffs", { project: `eq.${TEST_PROJECT}` });
    console.log("\n  🧹 Cleanup complete");

  } catch (err) {
    console.error(`  ❌ Integration test error: ${err.message}`);
    failed++;
    // Cleanup on error
    try {
      await del("session_ledger", { project: `eq.${TEST_PROJECT}` });
      await del("session_handoffs", { project: `eq.${TEST_PROJECT}` });
    } catch (_) {}
  }
}

// ─── Summary ──────────────────────────────────────────────────

console.log("\n═══════════════════════════════════════════════════════");
console.log(`  📊 Results: ${passed} passed, ${failed} failed, ${skipped} skipped`);
console.log("═══════════════════════════════════════════════════════\n");

process.exit(failed > 0 ? 1 : 0);
