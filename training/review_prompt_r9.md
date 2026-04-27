# BFCL V4 Training Pipeline — Round 9 Adversarial Review Prompt

You are an adversarial code reviewer auditing a fine-tuning pipeline for a Berkeley Function Calling Leaderboard (BFCL) submission. Your goal is to find bugs that silently corrupt data, destroy distributions, or deflate evaluation metrics.

## Pipeline Architecture
- **SFT Phase 1a**: `generate_bfcl_training_data.py` generates structured tool-calling training data from BFCL V4 collections. Uses `unroll_multi_turn()` to produce `{"messages": [...]}` format.
- **SFT Phase 1b**: `generate_diverse_sft.py` and `generate_sft_toolnames.py` generate supplemental SFT data (disambiguation, self-correction, coding anchors). Also uses `{"messages": [...]}` format.
- **GRPO Phase 2**: `bfcl_grpo_align.py` generates DPO-style preference pairs from the SFT training data, then trains with `mlx_lm.lora --dpo`.
- **Evaluation**: `bfcl_eval.py` evaluates via Ollama `/api/generate` with `raw: true` and proper ChatML formatting. `benchmark.py` evaluates directly with `mlx_lm.generate`.
- **Config**: `config.py` holds tool schemas, system prompt formatting, and hyperparameters.
- **RAG**: `semantic_rag.py` provides HyDE-based retrieval for context-limited tool injection at eval time.

## Round 8 Fixes Applied (DO NOT RE-REPORT)

### ✅ Fix 1: GRPO Alignment Key Mismatch [CRITICAL → FIXED]
- **File**: `bfcl_grpo_align.py:94-125`
- **Was**: `ex.get("prompt", "")` and `ex.get("completion", "")` — but training data uses `{"messages": [...]}` from `unroll_multi_turn()`. All tool-calling and miss-func pairs were silently dropped (100% data loss).
- **Fix**: Now reads `ex.get("messages", [])`, extracts assistant content as completion, and builds ChatML prompt from non-assistant messages.

### ✅ Fix 2: SFT `--mask-prompt` Failure [HIGH → FIXED]
- **Files**: `generate_diverse_sft.py` (7 locations), `generate_sft_toolnames.py` (2 locations)
- **Was**: All training examples used `{"text": "<|im_start|>..."}` format. `mlx_lm.lora --mask-prompt` requires `{"messages": [...]}` to identify prompt/completion boundaries.
- **Fix**: All 9 locations converted to `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` format.

### ✅ Fix 3: Ollama Template Double-Wrapping [HIGH → FIXED]
- **File**: `bfcl_eval.py:584,729,847`
- **Was**: `call_ollama` and `call_ollama_best_of_n` sent prompts without `"raw": True`, causing Ollama to wrap everything in default chat template. `evaluate_test` built prompt as `f"{sys_prompt}\n\nUser: {prompt}"` without ChatML structure.
- **Fix**: Added `"raw": True` to both Ollama payload functions. Changed `evaluate_test` to format as `<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.

### ✅ Fix 4: Benchmark Missing Tool Schemas [HIGH → FIXED]
- **File**: `benchmark.py:253`
- **Was**: Hardcoded `sys_prompt` with tool name list only — no XML schema definitions. Model couldn't map parameters.
- **Fix**: Now loads full schemas via `format_system_prompt(tools)` from `config.py`.

### ✅ Fix 5: Boolean Bypasses Number Validation [LOW → FIXED]
- **File**: `bfcl_eval.py:683`
- **Was**: `isinstance(arg_val, (int, float))` — `bool` is subclass of `int`, so `True` passed number validation.
- **Fix**: Added `or isinstance(arg_val, bool)` guard, matching the R7 integer fix.

### ⏳ Deferred: Multi-Turn Eval (Bug 5 from R8)
- Multi-turn evaluation only checks first turn. Deferred because test data structure needs `followup` key design first.

## Previously Fixed (R7 — DO NOT RE-REPORT)

| # | Issue | Status |
|---|-------|--------|
| R7-1 | Boolean hallucinations bypass `integer` validation | FIXED in `bfcl_eval.py:681` |
| R7-2 | Best-of-N NO_TOOL candidates shadow valid tool calls | FIXED in `bfcl_eval.py:759` |
| R7-3 | GRPO prompts as raw strings instead of message arrays | FIXED in `generate_bfcl_training_data.py:591` |
| R7-4 | SM-CoT reasoning stripping | FALSE POSITIVE — `tc_text = f'{think_text}{TOKEN_TOOL_CALL_OPEN}...'` |

## Previously Passed (Rounds 1-6)

| # | Check | Status |
|---|-------|--------|
| 1 | RAG fallback efficiency | PASS |
| 2 | Circular import risk | PASS |
| 3 | `_TOOL_SCHEMAS` load graceful | PASS |
| 4 | Train/eval RAG distribution match | PASS |
| 5 | `_safe_lower` single-char skip | PASS |
| 6 | Evol-Instruct param corruption | PASS |
| 7 | RAG eval latency | PASS |
| 8 | Null-param bypass | PASS |
| 9 | `BEST_OF_N_TEMPERATURE` scope | PASS |
| 10 | Hardcoded paths | WARNING (acceptable) |
| 11 | Exception robustness | PASS |
| 12 | `_wb_sub` regex injection | PASS |

## Updated Checklist (R8 → R9)

| # | Check | Status |
|---|-------|--------|
| 13 | SM-CoT Pipeline Correctness | R7: PASS |
| 14 | GRPO Data Format Match | R8: FIXED |
| 15 | Best-of-N Candidate Override | R7: FIXED |
| 16 | bool/int type coercion | R8: FIXED (both `integer` AND `number`) |
| 17 | MLX-LM Loss Masking Integrity | R8: FIXED (`messages` format everywhere) |
| 18 | Inference System Prompt Integrity | R8: FIXED (`raw: true` + ChatML) |
| 19 | Multi-Turn Execution Evaluation | DEFERRED (needs test data structure) |
| 20 | Benchmark Schema Injection | R8: FIXED |

## Your Task

Review the attached repomix for Round 9 bugs. Focus on:

1. **Data flow integrity**: Does training data flow correctly from generation → SFT → GRPO → evaluation?
2. **Format parity**: Are `messages` arrays consumed correctly by all downstream consumers?
3. **Evaluation fidelity**: Does the eval harness match the training distribution?
4. **Silent failures**: Are there exception handlers that swallow critical errors?
5. **Distribution alignment**: Are train/eval system prompts identical?
6. **Hyperparameter correctness**: Are batch sizes, learning rates, and sampling temperatures reasonable?
7. **Multi-turn correctness**: Beyond the deferred first-turn-only issue, are there other multi-turn gaps?

Report ONLY NEW bugs. Reference exact file names and line numbers. Include severity (CRITICAL/HIGH/MEDIUM/LOW), the bug description, and a proposed fix.
