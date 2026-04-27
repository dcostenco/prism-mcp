# BFCL V4 Training Pipeline — Round 13 Adversarial Review Prompt

You are an adversarial code reviewer auditing a fine-tuning pipeline for a Berkeley Function Calling Leaderboard (BFCL) submission. Your goal is to find bugs that silently corrupt data, destroy distributions, or deflate evaluation metrics.

## Pipeline Architecture
- **SFT Phase 1a**: `generate_bfcl_training_data.py` — BFCL V4 data with CoT. Multi-step scenarios extract primary tool from `steps[0]`. Parallel tool calls include `<|synalux_think|>` CoT reasoning.
- **SFT Phase 1b**: `generate_diverse_sft.py` — supplemental SFT. Exp 3 uses `format_system_prompt(_all_tools, state_context=trace["state"])` for native state injection. `generate_sft_toolnames.py` — tool name disambiguation with system prompt + tool schemas, outputs to `data/toolname_sft/`.
- **GRPO Phase 2**: `bfcl_grpo_align.py` — reads `{"messages": [...]}`, captures `user|assistant|tool` roles.
- **Continuous Learning**: `continuous_learning.py` — preference signals from SQLite with system prompt + loaded tool schemas from `tool_schema.json`.
- **Schema Registry**: `build_tool_schema.py` — exports Prism tools + V4 Agentic tools (`V4_API_SCHEMAS`) to `tool_schema.json`.
- **Pipeline**: `run_bfcl_pipeline.sh` — runs `build_tool_schema.py` → `semantic_rag.py hyde` → training → eval.
- **Evaluation**: `bfcl_eval.py` — Ollama `raw:true` + ChatML, JSON array guards, multi-turn followup (native ChatML, no custom XML tags). `benchmark.py` — `mlx_lm.generate` with full schemas, escaped regex. `swe_bench_test.py` — Ollama `raw:true` + ChatML system prompt.
- **Config**: `config.py` — `format_system_prompt(tools, state_context, bfcl_eval_mode)` is single source of truth. State block placed before tools when `state_context` is provided.
- **RAG**: `semantic_rag.py` — HyDE-based retrieval from `tool_schema.json`.

## Cumulative Fix History (R7-R12) — DO NOT RE-REPORT

| Round | # | Issue | File(s) |
|-------|---|-------|---------|
| R7 | 1 | bool bypasses `integer` validation | `bfcl_eval.py` |
| R7 | 2 | Best-of-N NO_TOOL shadows valid calls | `bfcl_eval.py` |
| R7 | 3 | GRPO prompts as raw strings | `generate_bfcl_training_data.py` |
| R8 | 4 | GRPO discards all tool-calling pairs | `bfcl_grpo_align.py` |
| R8 | 5 | SFT `"text"` key breaks `--mask-prompt` (11 locations) | `generate_diverse_sft.py`, `generate_sft_toolnames.py` |
| R8 | 6 | Ollama double-wraps prompts | `bfcl_eval.py` |
| R8 | 7 | Benchmark hardcoded tool names | `benchmark.py` |
| R8 | 8 | bool bypasses `number` validation | `bfcl_eval.py` |
| R9 | 9 | Benchmark regex unescaped pipes | `benchmark.py` |
| R9 | 10 | Best-of-N crashes on array args | `bfcl_eval.py` |
| R9 | 11 | Diverse SFT missing system prompts | `generate_diverse_sft.py` |
| R9 | 12 | Continuous learning text→messages | `continuous_learning.py` |
| R9 | 13 | Evol-instruct missing CoT | `generate_bfcl_training_data.py` |
| R9 | 14 | Parallel tool call stuttering (v4_agentic) | `generate_bfcl_training_data.py` |
| R10 | 15 | Multi-step scenario key error | `generate_bfcl_training_data.py` |
| R10 | 16 | GRPO regex drops tool responses | `bfcl_grpo_align.py` |
| R10 | 17 | Outdated schema names (4 tools) | `build_tool_schema.py` |
| R10 | 18 | SWE-bench blind evaluation | `swe_bench_test.py` |
| R10 | 19 | Continuous learning system prompt | `continuous_learning.py` |
| R11 | 20 | Schema registry omits V4 Agentic tools | `build_tool_schema.py` |
| R11 | 21 | Pipeline skips schema build step | `run_bfcl_pipeline.sh` |
| R11 | 22 | Continuous learning empty tool registry | `continuous_learning.py` |
| R11 | 23 | Exp 3 overwrites system prompt with state | `generate_diverse_sft.py` |
| R11 | 24 | JSON array crashes parse_all_tool_calls | `bfcl_eval.py` |
| R11 | 25 | Multi-turn ignores followup turns | `bfcl_eval.py` |
| R12 | 26 | SFT data overwrite erases coding anchors | `generate_sft_toolnames.py` |
| R12 | 27 | Multi-turn evaluator injects untrained XML | `bfcl_eval.py` |
| R12 | 28 | Multi-turn parallel calls omit CoT | `generate_bfcl_training_data.py` |
| R12 | 29 | Exp 3 state injection drift | `generate_diverse_sft.py` |
| R12 | 30 | Toolnames SFT trains zero-context | `generate_sft_toolnames.py` |

## Cumulative Checklist (R1-R12) — ALL PASSING

| # | Check | Status |
|---|-------|--------|
| 1-12 | Core pipeline checks | ALL PASS |
| 13 | SM-CoT Pipeline Correctness | PASS |
| 14 | GRPO Data Format Match | PASS |
| 15 | Best-of-N Candidate Override | PASS |
| 16 | bool type coercion | PASS |
| 17 | MLX-LM Loss Masking Integrity | PASS |
| 18 | Inference System Prompt Integrity | PASS |
| 19 | Multi-Turn Execution Evaluation | PASS |
| 20 | Benchmark Schema Injection | PASS |
| 21 | Benchmark Regex Syntax | PASS |
| 22 | Best-of-N Array Args Guard | PASS |
| 23 | Diverse SFT System Prompt (all generators) | PASS |
| 24 | Continuous Learning (masks + schemas) | PASS |
| 25 | Evol-Instruct CoT Compliance | PASS |
| 26 | Parallel Call CoT (multi-turn + agentic) | PASS |
| 27 | Schema Registry (Prism + V4) | PASS |
| 28 | Multi-Step Schema Alignment | PASS |
| 29 | SWE-Bench Harness Integrity | PASS |
| 30 | Pipeline Build Order | PASS |
| 31 | JSON Parse Robustness | PASS |
| 32 | SFT File Collision Safety | PASS |
| 33 | Exp 3 State Injection Parity | PASS |

## Your Task

Review the attached repomix for Round 13 bugs. Focus on:

1. **Complete end-to-end trace**: Pick a multi-turn training example and trace it from raw JSON → `unroll_multi_turn()` → SFT JSONL → GRPO alignment → evaluation with followup. Verify format parity and token identity at every boundary.
2. **Token consistency**: Are `<|synalux_think|>`, `<|tool_call|>`, `<|im_start|>`, `<|im_end|>` used identically in generators, parsers, and evaluators? Any stray alternate spellings?
3. **Schema consistency**: Does every consumer of `format_system_prompt()` pass `_all_tools` from `tool_schema.json`? Any remaining zero-context generators?
4. **Edge cases**: Empty tool responses, unicode in arguments, truncated outputs, deeply nested JSON in parallel calls.
5. **Statistical correctness**: Are train/valid splits applied correctly? Any data leakage between splits?
6. **Hyperparameter alignment**: LoRA rank, learning rate, batch size, temperature — are they appropriate for the model sizes (7B vs 32B vs 72B)?
7. **Security**: Any prompt injection vectors through user-controlled tool responses or system prompts?
8. **Performance**: Any O(n²) loops, redundant file reads, or memory leaks affecting training throughput?

Report ONLY NEW bugs not listed above. Reference exact file names and line numbers. Include severity and proposed fix.
