# v7.2.0 — Verification Harness (Front-Loaded Testing)

**Workspace:** `/Users/admin/prism` (main branch)
**Status:** `[In Progress]` — another agent is executing this plan.

---

## Goal

Formalize the `test_assertions.json` contract and build a planning-phase verification harness that forces the agent to emit programmatically verifiable test assertions *before* execution begins, then automatically validates them post-execution.

**Canonical Source:** [ROADMAP.md L153–166](file:///Users/admin/prism/ROADMAP.md#L153-L166)

---

## Existing Primitives (Already Shipped in v5.3)

These files are **already present** and provide the foundation. v7.2.0 enhances — not replaces — them.

| File | Purpose | Key Exports |
|------|---------|-------------|
| [runner.ts](file:///Users/admin/prism/src/verification/runner.ts) | QuickJS sandbox for assertion execution | `VerificationRunner.runSuite()` |
| [schema.ts](file:///Users/admin/prism/src/verification/schema.ts) | Zod schema for `test_assertions.json` | `TestAssertionSchema`, `TestSuiteSchema` |
| [hivemindWatchdog.ts L233-299](file:///Users/admin/prism/src/hivemindWatchdog.ts#L233-L299) | Existing `validation_result` loop | `runWatchdogSweep()` |
| [interface.ts](file:///Users/admin/prism/src/storage/interface.ts) | Health states `verifying`, `failed_validation` | `AgentHealthStatus` |
| [SKILL.md](file:///Users/admin/prism/skills/verification-planner/SKILL.md) | v5.3 verification planner skill | Skill instructions for agents |

---

## Proposed Changes

### Component 1: Enhanced Schema (Layer + Severity Policy)

#### [MODIFY] [schema.ts](file:///Users/admin/prism/src/verification/schema.ts)

The existing schema supports `layer: "data" | "agent" | "pipeline"` and `severity: "warn"`.
v7.2.0 adds:

```typescript
// New severity levels (v7.2.0)
severity: z.enum(["warn", "gate", "abort"]).default("warn")
//  warn  → log and continue
//  gate  → block progression until resolved
//  abort → rollback (fail the pipeline)

// New optional fields
timeout_ms: z.number().optional()     // per-assertion timeout
retry_count: z.number().optional()    // retry on transient failures
depends_on: z.string().optional()     // assertion dependency chain
```

**IMPORTANT:** All new fields are optional with defaults to maintain backward compatibility with existing `test_assertions.json` files.

---

#### [NEW] `src/verification/severityPolicy.ts`

Severity gate enforcement logic — separated from the runner for testability:

```typescript
export interface SeverityGateResult {
  action: "continue" | "block" | "abort";
  failed_assertions: AssertionResult[];
  summary: string;
}

export function evaluateSeverityGates(
  results: AssertionResult[],
  config: VerificationConfig
): SeverityGateResult;
```

**Rules:**
- `warn` failures → logged, always continue
- `gate` failures → block. Return `block` action with failed assertions list
- `abort` failures → immediate abort. Return `abort` action
- When `PRISM_VERIFICATION_DEFAULT_SEVERITY` overrides individual assertion severity

---

### Component 2: Enhanced Runner

#### [MODIFY] [runner.ts](file:///Users/admin/prism/src/verification/runner.ts)

Enhance `VerificationRunner.runSuite()` to:

1. **Accept layer filter:** `runSuite(suite, { layers?: string[] })` — run only specified layers
2. **Accept severity filter:** Skip assertions below configured minimum severity
3. **Per-assertion timeout:** If `timeout_ms` is set, wrap QuickJS eval in `Promise.race`
4. **Retry logic:** If `retry_count > 0`, retry transient failures (network-based assertions like `http_status`)
5. **Return structured result:** `VerificationResult` with per-layer breakdown

```typescript
export interface VerificationResult {
  passed: boolean;
  total: number;
  passed_count: number;
  failed_count: number;
  skipped_count: number;
  by_layer: Record<string, LayerResult>;
  duration_ms: number;
  severity_gate: SeverityGateResult;
}
```

---

### Component 3: `validation_result` Experience Event

#### [MODIFY] [hivemindWatchdog.ts](file:///Users/admin/prism/src/hivemindWatchdog.ts)

Enhance the existing verification loop (L233-299) to:

1. **Emit `validation_result` experience event** after each verification run:
   ```typescript
   await saveExperience({
     project,
     event_type: "validation_result",  // new event type
     context: `Verification run for ${taskContext}`,
     action: "automated_verification",
     outcome: result.passed ? "all_passed" : `${result.failed_count}/${result.total} failed`,
     confidence_score: Math.round((result.passed_count / result.total) * 100),
   });
   ```

2. **Respect severity gates:** When `SeverityGateResult.action === "block"`, set agent health to `failed_validation` and log the blocking assertions. When `"abort"`, additionally set a flag for the Dark Factory (v7.3) to stop iteration.

#### [MODIFY] `src/tools/experienceHandler.ts`

Add `"validation_result"` to the allowed `event_type` enum in the experience event handler.

---

### Component 4: Configuration

#### [MODIFY] [config.ts](file:///Users/admin/prism/src/config.ts)

Add at the end of the file (~L350+):

```typescript
// ─── v7.2: Verification Harness ──────────────────────────────
export const PRISM_VERIFICATION_HARNESS_ENABLED =
  process.env.PRISM_VERIFICATION_HARNESS_ENABLED === "true";

export const PRISM_VERIFICATION_LAYERS = (
  process.env.PRISM_VERIFICATION_LAYERS || "data,agent,pipeline"
).split(",").map(l => l.trim()).filter(Boolean);

export const PRISM_VERIFICATION_DEFAULT_SEVERITY =
  (process.env.PRISM_VERIFICATION_DEFAULT_SEVERITY || "warn") as "warn" | "gate" | "abort";
```

---

### Component 5: Verification Skill Update

#### [MODIFY] [SKILL.md](file:///Users/admin/prism/skills/verification-planner/SKILL.md)

Update the skill document to:
1. Reflect v7.2.0 severity levels (`warn`, `gate`, `abort`) instead of hardcoded `"warn"`
2. Add documentation for `timeout_ms`, `retry_count`, `depends_on` fields
3. Add examples showing multi-layer verification with severity gates

---

### Component 6: Claw-as-Validator Integration

#### [NEW] `src/verification/clawValidator.ts`

The adversarial validation loop — uses `claw_run_task` to execute generated test specs:

```typescript
export async function runClawValidation(
  testSuite: TestSuite,
  workDir: string,
  config: VerificationConfig
): Promise<VerificationResult>;
```

**Flow:**
1. Serialize the `test_assertions.json` to the work directory
2. Call `claw_run_task` with a validation prompt that instructs the local model to:
   - Read `test_assertions.json`
   - Execute each assertion against the actual output
   - Return structured pass/fail results
3. Parse the Claw response into `VerificationResult`
4. Emit `validation_result` experience event

**Fallback:** When Claw is unavailable, fall back to the built-in `VerificationRunner.runSuite()`.

---

## File Summary

| Action | File | Description |
|--------|------|-------------|
| MODIFY | `src/verification/schema.ts` | Add severity levels, timeout, retry, depends_on |
| MODIFY | `src/verification/runner.ts` | Layer/severity filtering, timeout, retry, structured result |
| NEW | `src/verification/severityPolicy.ts` | Severity gate enforcement logic |
| NEW | `src/verification/clawValidator.ts` | Claw-as-validator adversarial loop |
| MODIFY | `src/hivemindWatchdog.ts` | `validation_result` event, severity gate enforcement |
| MODIFY | `src/tools/experienceHandler.ts` | Allow `validation_result` event type |
| MODIFY | `src/config.ts` | `PRISM_VERIFICATION_*` env vars |
| MODIFY | `skills/verification-planner/SKILL.md` | Update for v7.2.0 features |

---

## Verification Plan

### Automated Tests
```bash
# Existing tests must continue to pass
npm test

# New tests
npx vitest run tests/verification/severityPolicy.test.ts
npx vitest run tests/verification/runner.test.ts  # enhanced
npx vitest run tests/verification/clawValidator.test.ts

# Build check
npm run build
```

### Integration Checks
- Feature-gated: with `PRISM_VERIFICATION_HARNESS_ENABLED=false`, existing behavior is unchanged
- With `PRISM_VERIFICATION_HARNESS_ENABLED=true`:
  - Watchdog detects `test_assertions.json` and runs enhanced verification
  - Severity gates correctly block/abort on `gate`/`abort` failures
  - `validation_result` experience events are recorded in the ledger
- Backward compatible: existing `test_assertions.json` files with `severity: "warn"` continue to work

---

## Coordination Note

> [!IMPORTANT]
> This plan is executed on **main** (`/Users/admin/prism`).
> v7.3.0 Dark Factory is developed in parallel at `/Users/admin/prism-7.3` on branch `feature/v7.3.0-dark-factory`.
> The v7.3.0 work codes against the **v5.3 verification interface** (`VerificationRunner.runSuite()`) as a baseline.
> After both are complete, the feature branch merges into main and picks up the v7.2.0 enhancements automatically.
