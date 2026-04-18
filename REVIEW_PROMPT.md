# Prism MCP Server — README/CHANGELOG/Roadmap Review (v10.0.0)

**Repomix:** `repomix-prism-coder-prism.txt` (~15.6K tokens)
**Also feed:** The `git diff` of `README.md`, `CHANGELOG.md`, and `package.json`.

---

## Context

This is a **documentation review** for the v10.0.0 release of Prism MCP Server.
The release adds HIPAA-hardened local LLM integration (`prism-coder:7b`) with
10 security findings closed across 3 rounds of adversarial review.

The reviewer should verify that the documentation accurately reflects the code.

---

## Review Checklist

### 1. README Hero Line (L15)
**New text:** *"...runs 100% on-device via `prism-coder:7b`, a fine-tuned local LLM
hardened against SSRF, prompt injection, and HIPAA data exfiltration."*

**Verify:**
- Does the code actually support full on-device operation for compaction and routing?
- Is the claim "hardened against SSRF" substantiated by `redirect: "error"` in the code?
- Is "HIPAA data exfiltration" prevention real (`PRISM_STRICT_LOCAL_MODE`)?

### 2. Capability Matrix
**New rows:**
| Feature | Local | Cloud |
|---------|-------|-------|
| Ledger compaction | ✅ `prism-coder:7b` | ✅ Text provider |
| Task routing (LLM tiebreaker) | ✅ `prism-coder:7b` | N/A |

**Verify:**
- Does `summarizeEntries()` actually call `callLocalLlm()` when `PRISM_LOCAL_LLM_ENABLED=true`?
- Does `askLocalLlmForRoute()` exist and work as described?
- Is "Auto-compaction ❌" correctly removed (it's now ✅ via local)?

### 3. HIPAA Security Hardening Table
**8 defense layers documented.**

**Verify each claim against the actual code:**
1. `PRISM_STRICT_LOCAL_MODE` — blocks cloud fallback? (compactionHandler.ts)
2. `redirect: "error"` — present in fetch? (localLlm.ts)
3. URL credential redaction — `redactUrl()` in both config.ts and localLlm.ts?
4. Entry-boundary truncation — splits on `\n\n`, not raw chars? (compactionHandler.ts)
5. Full XML escaping — 5 entities, applied to id/session_date? (compactionHandler.ts)
6. `<task>` boundary — description XML-escaped before injection? (taskRouterHandler.ts)
7. setTimeout cap — `Math.min(raw, 300_000)`? (config.ts)
8. Graceful HIPAA errors — try/catch in compactLedgerHandler? (compactionHandler.ts)

### 4. CHANGELOG v9.15.0 Entry
**Verify:**
- Does the "Security Audit Summary" table (10 findings, 3 rounds) match the work done?
- Are all 5 new env vars documented (`PRISM_LOCAL_LLM_ENABLED`, `_MODEL`, `_URL`,
  `_TIMEOUT_MS`, `PRISM_STRICT_LOCAL_MODE`)?
- Does the changelog accurately describe what `callLocalLlm()` does?

### 5. Roadmap
**Verify:**
- Is "Current: v10.0.0" consistent with `package.json` version?
- Are the Future Tracks (Semantic Routing, Background Mutex, Zero-Search) reasonable
  next steps based on the current architecture?

### 6. Package Version
**Bumped:** `9.13.4` → `10.0.0`

**Verify:**
- Major version bump signals breaking change — is there any breaking change (env var rename, API change)?
- Is the version consistent across package.json, README, and CHANGELOG?

---

## Output Format

```
### Documentation Review Verdict
- **Accuracy:** (Do docs match code?)
- **Completeness:** (Any missing features or env vars?)
- **Marketing Quality:** (Does this read as a wow-factor release?)
- **Issues:** (List any, or "CLEAN")
```
