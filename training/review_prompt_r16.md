# Adversarial Code Review — Round 16 (BFCL V4 Training Pipeline)

## Context & Objective
You are a senior adversarial code auditor performing Round 16 of an iterative hardening cycle on the Prism BFCL V4 training and evaluation pipeline. The objective is to achieve the **#1 ranking on the Berkeley Function Calling Leaderboard (BFCL)** by eliminating every bug that causes silent data corruption, evaluation deflation, or distribution drift.

## What Was Fixed in Round 15

### 1. [CRITICAL → RESOLVED] Silent Data Omission via Path Mismatch
- **Files:** `run_bfcl_pipeline.sh`, `config.py`
- **Root cause:** Bash script defined `AUX_DATA_DIR="data/aux_sft"` but never exported it. `config.py` defaults to `data/aux`. Python scripts wrote to `data/aux/`, bash merge phase looked in `data/aux_sft/` and silently dropped 1,200+ coding anchors + UX experiments.
- **Fix:** Added `export PRISM_AUX_DATA_DIR="$AUX_DATA_DIR"` after the definition, syncing with `config.py`'s `os.environ.get("PRISM_AUX_DATA_DIR", ...)`.

### 2. [HIGH → RESOLVED] Unescaped Pipes in Benchmark Regex
- **File:** `benchmark.py`
- **Root cause:** Three regex patterns using `<|synalux_think|>` had unescaped `|` chars, causing regex OR logic to wildly match angle brackets and corrupt think_length metrics.
- **Fix:** Escaped all pipe characters: `<\\|synalux_think\\|>` in all 3 patterns (lines 152, 197, 205).

## Cumulative Fix Log (R7–R15)
- **R7:** GRPO messages array parsing; benchmark regex escaping
- **R8:** Schema registry V4 tools; pipeline orchestration order
- **R9:** SFT data overwrite (toolname_sft/); multi-turn eval XML drift
- **R10:** Multi-step scenario tool selection; CoT in parallel calls
- **R11:** State injection centralized formatter; bfcl_eval array guard
- **R12:** System prompt injection in SFT toolname; output directory fix
- **R13:** NameError + zero-context diverse SFT; SLERP shape mismatch; Best-of-N abstention; --mask-prompt; continuous learning XML tags; benchmark array guard; regex chopping bypass; SFT toolname answer tags
- **R14:** Benchmark truthiness for zero-arg tools; Layer 3a dict guards; dynamic VALID_TOOLS sync
- **R15:** AUX_DATA_DIR path export; benchmark regex pipe escaping

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

If no bugs are found, explicitly state that the pipeline passes the Round 16 audit.
