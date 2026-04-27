# BFCL V4 Training Pipeline — Round 10 Adversarial Review Prompt

You are an adversarial code reviewer auditing a fine-tuning pipeline for a Berkeley Function Calling Leaderboard (BFCL) submission. Your goal is to find bugs that silently corrupt data, destroy distributions, or deflate evaluation metrics.

## Pipeline Architecture
- **SFT Phase 1a**: `generate_bfcl_training_data.py` generates structured tool-calling data from BFCL V4 collections. Uses `unroll_multi_turn()` → `{"messages": [...]}` format. Includes `<|synalux_think|>` CoT blocks in all generators including Evol-Instruct.
- **SFT Phase 1b**: `generate_diverse_sft.py` and `generate_sft_toolnames.py` generate supplemental SFT data with system prompts included via `format_system_prompt()`. Uses `{"messages": [...]}` format.
- **GRPO Phase 2**: `bfcl_grpo_align.py` reads `{"messages": [...]}` from SFT data, extracts ChatML prompt + completion for DPO pairs.
- **Continuous Learning**: `continuous_learning.py` extracts preference signals from Prism SQLite DB, outputs `{"messages": [...]}` format.
- **Evaluation**: `bfcl_eval.py` evaluates via Ollama `/api/generate` with `raw: true` and proper ChatML. `benchmark.py` evaluates via `mlx_lm.generate` with full tool schemas from `format_system_prompt()`.
- **Config**: `config.py` holds tool schemas, system prompt formatting, and hyperparameters.
- **RAG**: `semantic_rag.py` provides HyDE-based retrieval for context-limited tool injection.

## Round 9 Fixes Applied (DO NOT RE-REPORT)

### ✅ Fix 1: Benchmark Regex Rejects 100% of Outputs [CRITICAL → FIXED]
- **File**: `benchmark.py:120`
- **Was**: `r'<|tool_call|>\s*(.*?)\s*</|tool_call|>'` — unescaped pipes treated as regex OR.
- **Fix**: Escaped to `r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>'`.

### ✅ Fix 2: Best-of-N Crashes on Hallucinated Arrays [HIGH → FIXED]
- **File**: `bfcl_eval.py:660`
- **Was**: `tool_args.items()` crashed on list/non-dict arguments.
- **Fix**: Added `if not isinstance(tool_args, dict): return False`.

### ✅ Fix 3: Diverse SFT Missing System Prompts [HIGH → FIXED]
- **Files**: `generate_diverse_sft.py` main(), `generate_sft_toolnames.py`
- **Was**: Tool-call training examples had no system prompt → model learned to call tools without seeing tool definitions.
- **Fix**: `format_system_prompt(_all_tools)` injected as system message in all training arrays.

### ✅ Fix 4: Continuous Learning Breaks Loss Masking [HIGH → FIXED]
- **File**: `continuous_learning.py:110`
- **Was**: Re-serialized messages into `{"text": ...}` format, breaking `--mask-prompt`.
- **Fix**: Direct passthrough: `{"messages": pair["messages"]}`.

### ✅ Fix 5: Evol-Instruct Missing CoT [MEDIUM → FIXED]
- **File**: `generate_bfcl_training_data.py:1700`
- **Was**: Evol-instruct examples jumped straight to `<|tool_call|>` without `<|synalux_think|>` block.
- **Fix**: Added `think_text = f'{TOKEN_THINK_OPEN}\n...\n{TOKEN_THINK_CLOSE}\n'` before tool call.

### ✅ Fix 6: Parallel Tool Call Stuttering [LOW → FIXED]
- **File**: `generate_bfcl_training_data.py:1060`
- **Was**: `f"{think}\n".join([""] + tool_blocks)` repeated the think block between every tool call.
- **Fix**: `think + "\n".join(tool_blocks)` — prepend once.

## Previously Fixed (R7-R8 — DO NOT RE-REPORT)

| Round | Issue | Status |
|-------|-------|--------|
| R7 | bool bypasses `integer` validation | FIXED |
| R7 | Best-of-N NO_TOOL shadows valid calls | FIXED |
| R7 | GRPO prompts as raw strings | FIXED |
| R7 | SM-CoT reasoning stripping | FALSE POSITIVE |
| R8 | GRPO discards all tool-calling pairs (key mismatch) | FIXED |
| R8 | SFT `"text"` key breaks `--mask-prompt` (9 locations) | FIXED |
| R8 | Ollama double-wraps prompts (missing `raw: true`) | FIXED |
| R8 | Benchmark hardcoded tool names | FIXED |
| R8 | bool bypasses `number` validation | FIXED |

## Cumulative Checklist (R1-R9)

| # | Check | Status |
|---|-------|--------|
| 1-12 | Core pipeline checks (RAG, imports, schemas, regex, exceptions) | ALL PASS |
| 13 | SM-CoT Pipeline Correctness | PASS |
| 14 | GRPO Data Format Match | FIXED (R8) |
| 15 | Best-of-N Candidate Override | FIXED (R7) |
| 16 | bool type coercion (integer + number) | FIXED (R7+R8) |
| 17 | MLX-LM Loss Masking Integrity | FIXED (R8+R9) |
| 18 | Inference System Prompt Integrity | FIXED (R8) |
| 19 | Multi-Turn Execution Evaluation | DEFERRED |
| 20 | Benchmark Schema Injection | FIXED (R8) |
| 21 | Benchmark Regex Syntax | FIXED (R9) |
| 22 | Best-of-N Array Args Guard | FIXED (R9) |
| 23 | Diverse SFT System Prompt | FIXED (R9) |
| 24 | Continuous Learning Masking | FIXED (R9) |
| 25 | Evol-Instruct CoT Compliance | FIXED (R9) |
| 26 | Parallel Call Think Stutter | FIXED (R9) |

## Your Task

Review the attached repomix for Round 10 bugs. Focus on:

1. **Data flow end-to-end**: Trace a training example from raw BFCL JSON → generation → SFT JSONL → GRPO alignment → Ollama evaluation. Are there any remaining format mismatches?
2. **System prompt consistency**: Is the same `format_system_prompt()` used identically in training, GRPO, and eval? Are there any drift points?
3. **Token/tag consistency**: Are all special tokens (`<|synalux_think|>`, `<|tool_call|>`, `<|im_start|>`) used consistently across generators, parsers, and evaluators?
4. **Hyperparameter alignment**: Are learning rates, batch sizes, temperatures, and LoRA ranks configured correctly for the model size?
5. **Edge case robustness**: What happens when the model outputs empty responses, partial JSON, or mixed tool/text?
6. **Multi-turn gaps**: Beyond the deferred first-turn issue, what else is missing for the 30% agentic scoring weight?

Report ONLY NEW bugs. Reference exact file names and line numbers. Include severity and proposed fix.
