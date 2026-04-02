---
name: Verification Planner
description: Forces the generation of test_assertions.json during the implementation plan phase. v7.2.0 enhanced with severity gates, dependencies, timeouts, and Claw-as-Validator support.
---

# Verification Planner Skill (v7.2.0)

You are operating within the Prism ecosystem with the **v7.2.0 Verification Harness** enabled.

Any code changes you make will be automatically verified by the **Verification Runner**, which enforces severity gates (warn/gate/abort) and runs assertions through per-layer filtering, dependency chains, and configurable timeouts.

## When to use this skill
You MUST use this skill whenever you are in the **Planning Phase** of a task and are generating an `implementation_plan.md`.

## Instructions

Alongside your `implementation_plan.md`, you must also create a `test_assertions.json` file in the root of the project. This file must conform to our JSON schema and contains the tests that will be executed by the Watchdog once you finish the execution phase.

### Test Schema

Tests are categorized into 3 layers:
- `data`: Validate schema, row counts, and data integrity.
- `pipeline`: Validate expected pipeline execution outcomes (files exist, APIs return 200).
- `agent`: Validate correctness of formats or specific tool traces.

### Severity Levels (v7.2.0)

Each assertion has a `severity` that determines how the Verification Harness handles failures:

| Severity | Behavior |
|----------|----------|
| `warn`   | Logged and continues. Does not block progression. |
| `gate`   | **Blocks progression** until the assertion passes. The agent enters `failed_validation` and must fix the issue. |
| `abort`  | **Aborts the pipeline**. Indicates a critical failure that requires rollback or manual intervention. |

**Default severity** is `warn`. Use `gate` for assertions that must pass before code can ship. Use `abort` only for safety-critical checks.

The global severity floor can be overridden via `PRISM_VERIFICATION_DEFAULT_SEVERITY` env var (e.g., setting it to `gate` treats all assertions as at least gate-level).

### v7.2.0 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `timeout_ms` | number | Per-assertion timeout in milliseconds. Useful for HTTP checks. |
| `retry_count` | number | Number of retries on transient failures (e.g., `http_status`). Uses exponential backoff. |
| `depends_on` | string | ID of an assertion that must pass first. Skips this assertion if the dependency failed. |

### Full Example (v7.2.0)

```json
{
  "tests": [
    {
      "id": "db-schema-exists",
      "layer": "data",
      "description": "Ensure the users table exists",
      "severity": "abort",
      "assertion": {
        "type": "sqlite_query",
        "target": "SELECT name FROM sqlite_master WHERE type='table' AND name='users'",
        "expected": [{ "name": "users" }]
      }
    },
    {
      "id": "db-user-count",
      "layer": "data",
      "description": "Ensure exactly 50 test users were inserted",
      "severity": "gate",
      "depends_on": "db-schema-exists",
      "assertion": {
        "type": "sqlite_query",
        "target": "SELECT COUNT(*) as c FROM users",
        "expected": [{ "c": 50 }]
      }
    },
    {
      "id": "api-health",
      "layer": "pipeline",
      "description": "API health endpoint returns 200",
      "severity": "gate",
      "timeout_ms": 5000,
      "retry_count": 2,
      "assertion": {
        "type": "http_status",
        "target": "http://localhost:3000/health",
        "expected": 200
      }
    },
    {
      "id": "output-file-check",
      "layer": "pipeline",
      "description": "Check that the output file exists",
      "severity": "warn",
      "assertion": {
        "type": "file_exists",
        "target": "output/results.json",
        "expected": true
      }
    },
    {
      "id": "config-has-key",
      "layer": "pipeline",
      "description": "Config contains the required API key placeholder",
      "severity": "gate",
      "assertion": {
        "type": "file_contains",
        "target": ".env.example",
        "expected": "API_KEY="
      }
    },
    {
      "id": "complex-logic",
      "layer": "agent",
      "description": "Custom validation of business logic",
      "severity": "warn",
      "assertion": {
        "type": "quickjs_eval",
        "code": "return inputs.amount > 100;",
        "inputs": {
          "amount": 150
        }
      }
    }
  ]
}
```

### Supported Assertion Types

1. **`sqlite_query`**:
   - `target`: The raw SQL query to run.
   - `expected`: The exact expected JSON array outcome.
2. **`http_status`**:
   - `target`: The URL to fetch (GET).
   - `expected`: Numeric HTTP status code (e.g., 200).
3. **`file_exists`**:
   - `target`: Path to the file.
   - `expected`: Boolean indicating expected existence.
4. **`file_contains`**:
   - `target`: Path to the file.
   - `expected`: String that must be present in the file content.
5. **`quickjs_eval`**:
   - For anything that cannot be expressed declaratively.
   - `code`: Valid Javascript. Must return a boolean `true` (pass) or `false` (fail).
   - `inputs`: Optional dictionary of any JSON data to inject into the execution scope context.

### Severity Guidelines

- **Always start with `warn`** for non-critical assertions during exploration.
- **Promote to `gate`** when an assertion guards a feature's correctness (schema migrations, API contracts).
- **Use `abort`** only for safety-critical or data-destructive scenarios.
- **Use `depends_on`** to create assertion chains (e.g., check table exists → check row count).
- **Use `timeout_ms`** on all `http_status` assertions (recommend 5000ms).
- **Use `retry_count`** for flaky network checks (recommend 2).

### Claw-as-Validator (Adversarial Mode)

When `PRISM_VERIFICATION_HARNESS_ENABLED=true`, the system can optionally delegate your test suite to the local **Claw agent** for adversarial review before execution. Claw will:
1. Check for missing layer coverage
2. Identify potential false positives/negatives
3. Suggest additional assertions
4. Flag incorrect severity levels

This creates a **host ↔ Claw dialectic** that catches issues before the automated runner executes.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_VERIFICATION_HARNESS_ENABLED` | `false` | Master switch for v7.2.0 enhanced verification |
| `PRISM_VERIFICATION_LAYERS` | `data,agent,pipeline` | Comma-separated list of active layers |
| `PRISM_VERIFICATION_DEFAULT_SEVERITY` | `warn` | Global severity floor override |
