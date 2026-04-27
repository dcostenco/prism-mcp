# Adversarial Code Review — Round 14 (BFCL V4 Training Pipeline)

## Context & Objective
You are a senior adversarial code auditor performing Round 14 of an iterative hardening cycle on the Prism BFCL V4 training and evaluation pipeline. The objective is to achieve the **#1 ranking on the Berkeley Function Calling Leaderboard (BFCL)** by eliminating every bug that causes silent data corruption, evaluation deflation, or distribution drift.

## What Was Fixed in Round 13

### 1. [CRITICAL → RESOLVED] Fatal NameError Crash & Zero-Context Traces in Diverse SFT
- **File:** `generate_diverse_sft.py`
- **Root cause:** `generate_experiment_traces()` referenced `_all_tools` and `format_system_prompt` from `main()`'s local scope, causing a `NameError` on Experiment 3. Experiments 1, 2, 4, and coding anchors had no system prompt.
- **Fix:** Parameterized `generate_experiment_traces(sys_prompt, all_tools, format_fn)`. Injected `{"role": "system", "content": sys_prompt}` as the first message in ALL experiment traces.

### 2. [CRITICAL → RESOLVED] Fatal SLERP Shape Mismatch (GRPO vs SFT LoRA Layers)
- **Files:** `bfcl_grpo_align.py`, `run_bfcl_pipeline.sh`
- **Root cause:** SFT used `--lora-layers 24` but GRPO defaulted to `lora_layers=16` with no argparse exposure. Phase 3 SLERP merge crashes on tensor shape mismatch.
- **Fix:** Exposed `--lora-layers` in argparse, passed it to `train_dpo()`, and added `--lora-layers 24` to the pipeline invocation.

### 3. [HIGH → RESOLVED] Best-of-N Suppresses Valid Abstentions
- **File:** `bfcl_eval.py`
- **Root cause:** When model correctly abstains, `all_calls` is empty → marked `is_valid=False`. If a later candidate hallucinates a compliant tool call, Best-of-N returns the hallucination.
- **Fix:** Check for `<|synalux_answer|>` tags in no-tool responses. If present, mark as valid and break early.

### 4. [HIGH → RESOLVED] Continuous Learning Missing `--mask-prompt`
- **File:** `continuous_learning.py`
- **Root cause:** `run_training()` omitted `--mask-prompt`, causing loss computation over system/user tokens → catastrophic forgetting.
- **Fix:** Appended `"--mask-prompt"` to the training command.

### 5. [HIGH → RESOLVED] Continuous Learning Omits Mandatory XML Tags
- **File:** `continuous_learning.py`
- **Root cause:** Upvoted/correction completions used raw text without `<|synalux_think|>`/`<|synalux_answer|>` wrappers.
- **Fix:** Wrapped all completions in mandatory XML tags.

### 6. [HIGH → RESOLVED] Benchmark Crashes on Hallucinated Array Arguments
- **File:** `benchmark.py`
- **Root cause:** `.keys()` called on a `list` when model hallucinates array arguments.
- **Fix:** Added `isinstance(tool_args, dict)` type guard.

### 7. [MEDIUM → RESOLVED] Regex Chopping Bypasses JSON Repair
- **File:** `bfcl_eval.py`
- **Root cause:** Strategy 3 regex `\{[^}]*\}` chops nested JSON, then appends `(tool_name, {})`, bypassing `_repair_and_extract()`.
- **Fix:** Changed to `pass` on JSONDecodeError to allow the repair pipeline to handle it.

### 8. [MEDIUM → RESOLVED] SFT Toolnames Missing Answer Tags
- **File:** `generate_sft_toolnames.py`
- **Root cause:** Reasoning pair answers missing `<|synalux_answer|>` wrapper.
- **Fix:** Wrapped in `<|synalux_answer|>{answer}</|synalux_answer|>`.

## Cumulative Fix Log (R7–R13)
- **R7:** Fixed GRPO alignment to parse `messages` arrays; fixed benchmark regex escaping
- **R8:** Fixed schema registry to export V4 Agentic tools; fixed pipeline orchestration order
- **R9:** Fixed SFT data overwrite (toolname_sft/ collision); fixed multi-turn eval XML drift
- **R10:** Fixed multi-step scenario tool selection; fixed CoT missing in parallel calls
- **R11:** Fixed state injection to use centralized formatter; added benchmark array guard (bfcl_eval only)
- **R12:** Fixed system prompt injection in SFT toolname generator; fixed output directory
- **R13:** Fixed NameError + zero-context in diverse SFT; SLERP shape mismatch; Best-of-N abstention; --mask-prompt; continuous learning XML tags; benchmark array guard; regex chopping bypass; SFT toolname answer tags

## Your Task

The full, up-to-date pipeline source is attached below. Perform a focused adversarial review looking for:

1. **Silent data corruption** — Are any training examples still generated without system prompts, tool schemas, or mandatory XML tags?
2. **Cross-module translation bugs** — Are there any places where keys, formats, or schemas differ between the generator scripts and the evaluation/alignment scripts?
3. **Evaluation harness correctness** — Are there any edge cases in `bfcl_eval.py` or `benchmark.py` that would cause score deflation or crashes?
4. **Distribution drift** — Will the training data distribution match the inference-time prompt format exactly?
5. **Pipeline orchestration** — Does `run_bfcl_pipeline.sh` execute all phases in the correct order with the right inputs/outputs?
6. **Completeness** — Are there any remaining TODO/FIXME/HACK comments that indicate unfinished work?

For each bug found, provide:
- **File** and **line number**
- **Severity** (Critical / High / Medium)
- **Root cause** explanation
- **Proposed fix** with code snippet

If no bugs are found, explicitly state that the pipeline passes the Round 14 audit.
