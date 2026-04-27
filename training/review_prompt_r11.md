# BFCL V4 Training Pipeline — Round 11 Adversarial Review Prompt

You are an adversarial code reviewer auditing a fine-tuning pipeline for a Berkeley Function Calling Leaderboard (BFCL) submission. Your goal is to find bugs that silently corrupt data, destroy distributions, or deflate evaluation metrics.

## Pipeline Architecture
- **SFT Phase 1a**: `generate_bfcl_training_data.py` generates structured tool-calling data from BFCL V4 collections. Uses `unroll_multi_turn()` → `{"messages": [...]}` format. All generators include `<|synalux_think|>` CoT. Multi-step scenarios correctly extract primary tool from `steps[0]`.
- **SFT Phase 1b**: `generate_diverse_sft.py` and `generate_sft_toolnames.py` generate supplemental SFT data with system prompts via `format_system_prompt()`. Uses `{"messages": [...]}` format.
- **GRPO Phase 2**: `bfcl_grpo_align.py` reads `{"messages": [...]}` from SFT data, extracts ChatML prompt + completion for DPO pairs. Regex captures `user|assistant|tool` roles.
- **Continuous Learning**: `continuous_learning.py` extracts preference signals from Prism SQLite DB, outputs `{"messages": [...]}` with system prompts.
- **Evaluation**: `bfcl_eval.py` evaluates via Ollama `/api/generate` with `raw: true` and ChatML. `benchmark.py` evaluates via `mlx_lm.generate` with full tool schemas. `swe_bench_test.py` evaluates via Ollama with `raw: true` and ChatML system prompt.
- **Config**: `config.py` holds tool schemas, system prompt formatting, and hyperparameters.
- **Schema Registry**: `build_tool_schema.py` generates `tool_schema.json` with canonical tool names matching fine-tuned targets.
- **RAG**: `semantic_rag.py` provides HyDE-based retrieval for context-limited tool injection.

## Round 10 Fixes Applied (DO NOT RE-REPORT)

### ✅ Fix 1: Multi-Step Scenario Key Error [CRITICAL → FIXED]
- **File**: `generate_bfcl_training_data.py:1068`
- **Was**: `scenario.get("tool", "")` failed for multi-step scenarios with `"steps"` key.
- **Fix**: `primary_tool = scenario["steps"][0]["tool"] if "steps" in scenario else scenario.get("tool", "")`.

### ✅ Fix 2: GRPO Regex Strips Tool Responses [CRITICAL → FIXED]
- **File**: `bfcl_grpo_align.py:202`
- **Was**: Regex only captured `user|assistant`, dropping all `tool` role messages.
- **Fix**: `(user|assistant|tool)` in turn extraction pattern.

### ✅ Fix 3: Hardcoded Outdated Schema Names [HIGH → FIXED]
- **File**: `build_tool_schema.py:25,42,68,121`
- **Was**: `session_save`, `session_search`, `session_delete`, `session_handoff`.
- **Fix**: Updated to `session_save_ledger`, `session_search_memory`, `session_forget_memory`, `session_save_handoff`.

### ✅ Fix 4: SWE-Bench Evaluates Without Context [HIGH → FIXED]
- **File**: `swe_bench_test.py:205,371`
- **Was**: No `raw: True`, no system prompt, no tool schemas.
- **Fix**: Added `raw: True` to payload, loads `tool_schema.json`, formats ChatML system prompt.

### ✅ Fix 5: Continuous Learning Missing System Prompt [MEDIUM → FIXED]
- **File**: `continuous_learning.py:72`
- **Was**: Upvoted and correction pairs lacked system prompt.
- **Fix**: Injected `format_system_prompt()` as system message in both pair types.

## Previously Fixed (R7-R9 — DO NOT RE-REPORT)

| Round | # | Issue | Status |
|-------|---|-------|--------|
| R7 | 1 | bool bypasses `integer` validation | FIXED |
| R7 | 2 | Best-of-N NO_TOOL shadows valid calls | FIXED |
| R7 | 3 | GRPO prompts as raw strings | FIXED |
| R7 | 4 | SM-CoT reasoning stripping | FALSE POSITIVE |
| R8 | 5 | GRPO discards all tool-calling pairs | FIXED |
| R8 | 6 | SFT `"text"` key breaks `--mask-prompt` (9 locations) | FIXED |
| R8 | 7 | Ollama double-wraps prompts | FIXED |
| R8 | 8 | Benchmark hardcoded tool names | FIXED |
| R8 | 9 | bool bypasses `number` validation | FIXED |
| R9 | 10 | Benchmark regex unescaped pipes | FIXED |
| R9 | 11 | Best-of-N crashes on array args | FIXED |
| R9 | 12 | Diverse SFT missing system prompts | FIXED |
| R9 | 13 | Continuous learning text→messages | FIXED |
| R9 | 14 | Evol-instruct missing CoT | FIXED |
| R9 | 15 | Parallel tool call stuttering | FIXED |
| R10 | 16 | Multi-step scenario key error | FIXED |
| R10 | 17 | GRPO regex drops tool responses | FIXED |
| R10 | 18 | Outdated schema names | FIXED |
| R10 | 19 | SWE-bench blind evaluation | FIXED |
| R10 | 20 | Continuous learning system prompt | FIXED |

## Cumulative Checklist (R1-R10)

| # | Check | Status |
|---|-------|--------|
| 1-12 | Core pipeline checks | ALL PASS |
| 13 | SM-CoT Pipeline Correctness | PASS |
| 14 | GRPO Data Format Match | FIXED (R8+R10) |
| 15 | Best-of-N Candidate Override | FIXED (R7) |
| 16 | bool type coercion | FIXED (R7+R8) |
| 17 | MLX-LM Loss Masking Integrity | FIXED (R8+R9) |
| 18 | Inference System Prompt Integrity | FIXED (R8+R10) |
| 19 | Multi-Turn Execution Evaluation | DEFERRED |
| 20 | Benchmark Schema Injection | FIXED (R8) |
| 21 | Benchmark Regex Syntax | FIXED (R9) |
| 22 | Best-of-N Array Args Guard | FIXED (R9) |
| 23 | Diverse SFT System Prompt | FIXED (R9) |
| 24 | Continuous Learning Masking + Sys Prompt | FIXED (R9+R10) |
| 25 | Evol-Instruct CoT Compliance | FIXED (R9) |
| 26 | Parallel Call Think Stutter | FIXED (R9) |
| 27 | Schema Registry Accuracy | FIXED (R10) |
| 28 | Multi-Step Schema Alignment | FIXED (R10) |
| 29 | SWE-Bench Harness Integrity | FIXED (R10) |

## Your Task

Review the attached repomix for Round 11 bugs. Focus on:

1. **End-to-end data flow**: Trace a complete example from raw BFCL JSON → generation → SFT → GRPO → evaluation. Verify format parity at every stage.
2. **Schema consistency**: Do `build_tool_schema.py`, `config.py`, training generators, and evaluators all reference the same tool names/schemas?
3. **Hyperparameter alignment**: Are learning rates, batch sizes, LoRA ranks, and temperatures configured correctly for the model size (7B/32B/72B)?
4. **Edge case handling**: What happens with empty responses, nested JSON, unicode, or truncated outputs?
5. **Multi-turn gaps**: Beyond the deferred first-turn issue, are there other agentic evaluation gaps?
6. **Distribution drift**: Could any remaining training/eval mismatches cause systematic score deflation?
7. **Token budget**: Are sequence lengths and context windows properly managed for the target model?

Report ONLY NEW bugs. Reference exact file names and line numbers. Include severity and proposed fix.
