import { z } from "zod";

// ─── v7.2.0: Severity Levels ────────────────────────────────
//  warn  → log and continue
//  gate  → block progression until resolved
//  abort → rollback (fail the pipeline)
export const SeverityLevel = z.enum(["warn", "gate", "abort"]).default("warn");
export type SeverityLevel = z.infer<typeof SeverityLevel>;

// Base for all assertions
const BaseAssertion = z.object({
  target: z.string().describe("The SQL query, URL, or file path"),
  expected: z.any().describe("The expected outcome to match against"),
});

// 1. Declarative Assertions (Split for better type inference)
export const SqliteAssertionSchema = BaseAssertion.extend({ type: z.literal("sqlite_query") });
export const HttpStatusAssertionSchema = BaseAssertion.extend({ type: z.literal("http_status") });
export const FileExistsAssertionSchema = BaseAssertion.extend({ type: z.literal("file_exists") });
export const FileContainsAssertionSchema = BaseAssertion.extend({ type: z.literal("file_contains") });

// 2. Sandboxed JS Assertion
export const SandboxedJsAssertionSchema = z.object({
  type: z.literal("quickjs_eval"),
  code: z.string().describe("JS code to run in QuickJS. Must return a boolean."),
  inputs: z.record(z.string(), z.any()).optional().describe("Data to inject as globals into the sandbox"),
});

// 3. Main Schema Wrapper (v7.2.0 enhanced)
export const TestAssertionSchema = z.object({
  id: z.string(),
  layer: z.enum(["data", "agent", "pipeline"]),
  description: z.string(),
  severity: SeverityLevel,
  // v7.2.0: per-assertion timeout in ms
  timeout_ms: z.number().int().min(50).max(120_000).optional().describe("Per-assertion timeout in milliseconds"),
  // v7.2.0: retry on transient failures (e.g. http_status)
  retry_count: z.number().int().min(0).max(5).optional().describe("Number of retries on transient failures"),
  // v7.2.0: assertion dependency chain
  depends_on: z.string().optional().describe("ID of assertion that must pass first"),
  // Discriminated union gives pinpoint accuracy on parsing errors
  assertion: z.discriminatedUnion("type", [
    SqliteAssertionSchema,
    HttpStatusAssertionSchema,
    FileExistsAssertionSchema,
    FileContainsAssertionSchema,
    SandboxedJsAssertionSchema
  ])
});

export const TestSuiteSchema = z.object({
  tests: z.array(TestAssertionSchema)
});

// Types for TypeScript
export type TestAssertion = z.infer<typeof TestAssertionSchema>;
export type TestSuite = z.infer<typeof TestSuiteSchema>;
export type DeclarativeAssertion = z.infer<typeof SqliteAssertionSchema> | z.infer<typeof HttpStatusAssertionSchema> | z.infer<typeof FileExistsAssertionSchema> | z.infer<typeof FileContainsAssertionSchema>;
export type SandboxedJsAssertion = z.infer<typeof SandboxedJsAssertionSchema>;

// ─── v7.2.0: Verification Result Structures ─────────────────

/** Per-assertion result */
export interface AssertionResult {
  id: string;
  layer: "data" | "agent" | "pipeline";
  description: string;
  severity: SeverityLevel;
  passed: boolean;
  error?: string;
  duration_ms: number;
  retries_used: number;
  skipped: boolean;
  skip_reason?: string;
}

/** Per-layer breakdown */
export interface LayerResult {
  passed: number;
  failed: number;
  skipped: number;
  total: number;
  assertions: AssertionResult[];
}

/** Full verification result from a suite run */
export interface VerificationResult {
  passed: boolean;
  total: number;
  passed_count: number;
  failed_count: number;
  skipped_count: number;
  by_layer: Record<string, LayerResult>;
  duration_ms: number;
  severity_gate: SeverityGateResult;
  assertion_results: AssertionResult[];
}

/** Severity gate evaluation outcome */
export interface SeverityGateResult {
  action: "continue" | "block" | "abort";
  failed_assertions: AssertionResult[];
  summary: string;
}

/** Runtime configuration for verification */
export interface VerificationConfig {
  enabled: boolean;
  layers: string[];
  default_severity: SeverityLevel;
}
