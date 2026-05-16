# Code Review Cascade Benchmark

Measures how well the local Prism cascade (1.7B → 14B → 32B → cloud) catches **real production bugs** vs. how many **false positives** each tier introduces.

This is different from the [routing benchmark](../prism-routing-100/) — that measures *tool selection*. This measures **correctness review** on real code that shipped (or almost shipped) bugs.

## Why this exists

Routing tasks have a clear right/wrong answer (the model picked the correct tool or not). Code review is fuzzier — a model can be "right but useless" (caught style nits but missed the real bug) or "wrong but loud" (lots of confident false positives).

The cascade goal: minimize cost while maintaining high recall on **real** bugs, with low precision-loss to false positives.

## Schema (cases.jsonl)

Each line is one test case:

```jsonc
{
  "id": "short-slug",
  "description": "What this case tests + provenance",
  "language": "typescript|python|swift|...",
  "framework": "next.js-app-router|...",   // optional
  "code": "...",                            // the snippet
  "real_bugs": [
    {
      "severity": "high|medium|low",
      "tags": ["security", "crypto", ...],
      "description": "...",
      "fix_hint": "..."
    }
  ],
  "non_bugs": [                             // common false-positive traps
    "missing import of X (assume top-of-file)",
    ...
  ],
  "models_observed": {                      // optional — log per-model results
    "prism-coder:14b-v28": {
      "real_bugs_caught": ["..."],
      "false_positives": 6,
      "summary": "..."
    }
  }
}
```

## Scoring

For each case + each model:

- **Recall** = `real_bugs_caught / total_real_bugs`
- **Precision** = `real_bugs_caught / (real_bugs_caught + false_positives)`
- **F1** = harmonic mean

Cascade metrics:

- **Best-of-N**: union of all local models' real_bugs_caught (proxy for ensemble ceiling)
- **Marginal value of cloud**: real bugs caught ONLY by cloud (i.e., cascade must escalate)
- **Cost**: estimated `$ per case` (cloud calls priced, local = $0)

## Current cases (1)

| id | language | severity ladder | local catch rate |
|---|---|---|---|
| `stripe-hmac-body-encoding` | TypeScript | 1 high + 1 med + 1 low | 0/3 (both 14B + 32B miss the HMAC body bug) — requires cloud |

## Adding a case

When a real bug ships (or almost ships), capture it here:

1. Strip identifying info (env names, table names) but keep enough that a model can reason about it
2. Write the `real_bugs` list as **the diff that fixed it in production** — not theoretical issues
3. Write the `non_bugs` list as **whatever a 14B-class model would falsely flag** (run it to find out)
4. Run all configured models and record `models_observed` so we can track regressions
