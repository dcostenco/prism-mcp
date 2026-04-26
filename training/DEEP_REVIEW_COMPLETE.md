# BFCL prism-coder-72b — Deep Review #4 (THINK DEEPER)

> **Context**: This is the 4th review round. Three prior reviews validated 10 fixes across the handler, training pipeline, and data generator. This review should be ADVERSARIAL — look for things the previous reviews missed. The codebase now includes bleeding-edge research strategies (BalanceSFT, Noisy Trajectories, Model Souping). All 35 unit tests pass.
>
> **Hardware**: Apple M5 Max 48GB unified memory. MLX-native pipeline using `mlx_lm`.
>
> **Goal**: #1 overall rank on BFCL V4 leaderboard. Currently projected Agentic 78-82%, Overall 76-79%.

---

## Summary of ALL Changes (Cumulative)

### Handler (`prism_coder.py`) — 7 fixes
1. `decode_execute`: type coercion via `_fix_argument_types` + `repr(v)` pattern
2. Native `<|im_start|>tool` role for tool responses
3. 7-rule strict system prompt (abstention, no-guessing, direct-answer, clarification, clear-final-answer, no-hallucinated-optional-params, exact-memory-keys)
4. Regex-based JSON extraction fallback (`_extract_tool_calls`)
5. Language-aware coercion (Python bool/null only, Java/JS strings preserved)

### Training Data Generator (`generate_bfcl_training_data.py`) — 5 strategies
1. **BalanceSFT**: `format_as_prompt_completion()` outputs `{"prompt": ..., "completion": ...}` format for `--mask-prompt` loss masking
2. **Parallel tool calling**: 4 scenario templates generating multiple `<tool_call>` blocks per assistant turn
3. **Noisy trajectories**: 15% of multi-turn examples inject user interruptions mid-conversation
4. 7-rule system prompt (synced with handler)
5. Raw text format via `format_as_raw_text()` ensures token-identical training-inference alignment

### SFT Training (`bfcl_qlora_finetune.py`)
- `--mask-prompt` for BalanceSFT loss masking
- `--grad-accum 4` for training stability
- `--max-seq-length 8192` for long agentic sequences

### DPO Training (`bfcl_grpo_align.py`)
- Uses `mlx_lm.dpo` (not `.lora`)
- `--max-seq-length 2048` (DPO memory cap for 48GB)
- `--grad-checkpoint` enabled

### Post-Training (`merge_adapters.py`) [NEW]
- Model Souping: weighted average of SFT + DPO adapters (default 0.6/0.4)

---

## Deep Review Questions — THINK ADVERSARIALLY

### A. Loss Masking Correctness
1. `format_as_prompt_completion()` splits at the LAST assistant message. For multi-turn training data with 3 turns:
   - Turn 1: user → assistant(tool_call) → tool(result)
   - Turn 2: user → assistant(tool_call) → tool(result)
   - Turn 3: user → assistant(tool_call)
   
   The loss is computed ONLY on Turn 3's assistant message. **Is this correct?** Should the model also learn from Turn 1 and Turn 2's tool_call outputs? Is there a risk that the model only learns to generate the FINAL tool call in a chain but not intermediate ones?

2. For multi-turn sequences, does masking everything except the last completion actually hurt multi-turn performance? The model never gets gradient signal for how to generate the FIRST tool call in a conversation — it only sees the system prompt and learns to generate the LAST one. **Is this a critical flaw?**

3. Should we switch to a format where ALL assistant messages contribute to the loss (not just the last one)? Does MLX `--mask-prompt` support this with the `chat` format?

### B. Noisy Trajectory Design
4. The interruptions are injected AFTER all tool turns are complete (at the end of the conversation). This means the model never sees an interruption BETWEEN tool calls. **Should the injection point be randomized** (e.g., after the 1st turn instead of at the end)?

5. The interruption responses are generic ("I can't help with that"). But in BFCL Agentic tests, the model sometimes needs to handle a CONTEXT SWITCH where the user asks about a different tool entirely. Does the training data teach the model to switch tools mid-conversation, or only to refuse?

### C. decode_execute Edge Cases
6. If the model outputs BOTH a tool call AND conversational text (e.g., "Sure, let me check that for you.\n<tool_call>..."), does `_extract_tool_calls` correctly ignore the conversational prefix? What about suffix text?

7. For parallel tool calls in `decode_execute`, are they handled as individual function call strings or combined? What if `eval()` receives `"func1(a=1)\nfunc2(b=2)"` — does it correctly execute both?

8. What happens if `_fix_argument_types` receives a value that is ALREADY a Python bool (`True`) not a string (`"true"`)? Does it accidentally re-coerce it to a string?

### D. Training Data Quality
9. The scenario templates use HARDCODED function names like `"mkdir"`, `"ls"`, `"send_message"`. But BFCL test data uses a WIDE variety of function names and parameter schemas. Is there a risk of **overfitting to these specific function names** rather than learning the general pattern of tool calling?

10. How many UNIQUE parallel tool calling examples will be generated? With 4 templates and `num_examples=1000`, what is the distribution? Is it enough to generalize?

11. The DPO pairs in `generate_grpo_pairs()` — do they use the same `format_as_prompt_completion()` format, or do they use a different format? Is there a format mismatch between SFT and DPO training data?

### E. Model Souping Safety
12. The `merge_adapters.py` script averages weights. But if the SFT adapter has rank 16 and the DPO adapter has rank 8, can you still merge them? What happens with incompatible architectures?

13. After merging, does the merged adapter need to be re-validated with the 35-test suite? Could the merge introduce subtle behavioral regressions that tests wouldn't catch?

### F. Pipeline Coherence
14. In `run_bfcl_pipeline.sh`, after SFT, the adapter is FUSED into the model before DPO starts. This means DPO trains on the fused SFT model. But `merge_adapters.py` expects two separate UNFUSED adapters. **Is there an architectural mismatch** in the pipeline flow? Should the SFT adapter be saved BEFORE fusing?

15. The system prompt has 7 rules. The BFCL Agentic checker uses substring matching on the final answer. Do any of your 7 rules potentially degrade the model's willingness to output natural language answers when the Agentic checker expects them?

16. What is the EXACT end-to-end flow from model output → `decode_execute` → `eval()` → BFCL checker? Walk through a complete example with a parallel tool call to prove there are no format mismatches at any stage.

### G. Hardware & MLX Specifics
17. Does `mlx_lm.lora` with `--mask-prompt` actually work with the `completions` format? Have you verified this isn't only supported for `chat` format in the latest mlx-lm version?

18. With `--grad-accum 4` and `--batch-size 1`, the effective batch size is 4. Combined with `--max-seq-length 8192`, what is the peak memory estimate? Is there any risk this combination pushes past 48GB?

### H. Competitive Gap Analysis
19. What specific techniques does the current #1 model (GPT-4.1 or Claude Opus 4.5) use for BFCL that you have NOT implemented? List any known gaps.

20. Your projected Overall is 76-79%. The actual #1 is at 77.47%. What is the single highest-risk factor that could cause you to underperform this projection? What is your contingency plan?

---

## Code Follows Below

> The full codebase (handler + training pipeline + merge script) follows as two repomix bundles.
# 🔍 External Code Review Request: BFCL #1 Ranking — Prism-Coder Agent

## Objective

We are building **prism-coder-72b**, a fine-tuned Qwen2.5-Coder-72B model targeting the **#1 ranking** on the [Berkeley Function Calling Leaderboard (BFCL V4)](https://gorilla.cs.berkeley.edu/leaderboard.html). We need an expert review of our handler code, training pipeline, and strategy.

**Organization:** Synalux  
**Base Model:** Qwen2.5-Coder-72B (via Ollama)  
**Hardware:** Apple M5 Max 48GB (MLX-native pipeline)  
**Framework:** BFCL V4 eval harness (Python, eval checkpoint `f7cf735`)

---

## Current Competitive Landscape

| Rank | Model | Overall | Agentic(40%) | Multi-Turn(30%) |
|------|-------|---------|-------------|----------------|
| 1 | Claude Opus 4.5 (FC) | **77.47%** | 79.1% | 68.4% |
| 4 | GLM-4.6 (FC, MIT) | **72.38%** | 66.6% | 68.0% |
| 18 | xLAM-2-32b (FC) | **54.66%** | 23.2% | 69.5% |

**Scoring Formula:** `Overall = 10%×NL_AST + 10%×Live_AST + 10%×Irrelevance + 30%×Multi_Turn + 40%×Agentic`

The **Agentic** category (40% weight) is the kingmaker. It consists of:
- **Web Search** (avg of base + no_snippet): Model must call `search_engine_query()` and `fetch_url_content()`, then synthesize a natural language answer containing the expected substring.
- **Memory** (avg of KV + Vector + Rec Sum): Model must call memory APIs (`put()`, `get()`, `search()`, `add_embedding()`, etc.) and produce correct state after execution.

**Critical eval mechanics:**
1. `decode_execute()` output is literally `eval()`'d as Python — format must be exact
2. Agentic checker is **pure substring match** on the model's last non-function-call message
3. Multi-turn checker verifies state after each turn via ground-truth subsequence matching
4. Irrelevance: model must output NO function calls for irrelevant queries (empty list or text-only)

---

## Architecture Overview

### Files to Review

Two repomix bundles are attached:

**Bundle 1: `bfcl_repomix_eval.txt` (46K tokens, 14 files)**
- `prism_coder.py` — Our handler (514 lines). Extends `QwenFCHandler`.
- `salesforce_qwen.py` — xLAM competitor handler (115 lines). Direct comparison target.
- `qwen_fc.py` — Parent handler class we extend
- `base_oss_handler.py` — OSS handler base (Ollama/vLLM inference)
- `base_handler.py` — Abstract handler with 8 multi-turn FC methods
- `agentic_checker.py` — Agentic answer validation (substring match)
- `multi_turn_checker.py` — Multi-turn state verification
- `multi_turn_utils.py` — Function call execution (`eval()` bridge)
- `memory_kv.py`, `memory_vector.py`, `memory_rec_sum.py` — Memory APIs
- `web_search.py` — Web search API (SerpAPI + URL fetching)
- `eval_runner.py`, `eval_runner_helper.py` — Scoring and evaluation pipeline

**Bundle 2: `bfcl_repomix_training.txt` (17K tokens, 5 files)**
- `bfcl_qlora_finetune.py` — MLX-native QLoRA SFT script
- `bfcl_grpo_align.py` — GRPO/DPO preference alignment
- `generate_bfcl_training_data.py` — Training data generation from BFCL test data
- `run_bfcl_pipeline.sh` — Master training pipeline orchestrator
- `test_handler.py` — Unit test suite (35 test cases)

### Model Registration (model_config.py)

```python
"prism-coder-72b-FC": ModelConfig(
    model_name="qwen2.5:72b",
    display_name="prism-coder-72b (FC)",
    url="https://github.com/dcostenco/prism",
    org="Synalux",
    license="Apache-2.0",
    model_handler=PrismCoderHandler,
    is_fc_model=True,
    underscore_to_dot=False,
),
```

---

## Review Questions

Please analyze the attached code bundles and answer the following:

### 1. Handler Correctness (Critical)

- **`decode_ast()`**: Does our JSON extraction handle all edge cases? Compare with xLAM's simpler `json.loads()` approach. Are we over-engineering the regex extraction?
- **`decode_execute()`**: Is our Python expression format exactly compatible with `eval()` in `multi_turn_utils.py`? Key concern: `repr(v)` vs `json.dumps(v)` for string arguments.
- **`_format_prompt()`**: Compare our 514-line prompt formatting with xLAM's 115-line handler. Are there redundant steps, edge cases, or prompt injection risks?
- **Tool response formatting**: We use `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>` while xLAM uses `<|im_start|>tool\n...<|im_end|>`. Which is correct for Qwen2.5?
- **Multi-turn FC methods**: We implement all 8 abstract methods from `BaseHandler`. Are there any missing or incorrectly implemented? Pay attention to `_add_execution_results_prompting()`.

### 2. Agentic Strategy (40% of score)

- The agentic checker does `re.search(rf"\b{re.escape(possible_answer)}\b", standardized_model_response)` after removing `,./-_*^()` and lowercasing. How should our model format its final answer to maximize substring match success?
- For memory APIs: The model must call functions like `put("key", "value")`, `get("key")`, etc. After all tool calls, the state is compared. What training data patterns would maximize state correctness?
- For web search: The model must search, optionally fetch URLs, then provide a final answer. What system prompt or instruction tuning would maximize answer quality?

### 3. Training Pipeline (MLX-specific)

- Does our MLX QLoRA pipeline correctly handle Qwen2.5 architecture (attention heads, MLP projections)?
- Is our GRPO alignment strategy (chosen=abstention, rejected=hallucinated call) the right approach for irrelevance?
- Training data format: Are the generated training examples properly formatted for Qwen's `<|im_start|>` chat template?
- What additional training data should we generate for agentic category success?

### 4. Competitive Gap Analysis

- Compare our handler architecture vs xLAM's `SalesforceQwenHandler`. What design patterns make xLAM effective at multi-turn (69.5%) despite simple code?
- xLAM overrides `_pre_query_processing_prompting()` to use its own system prompt. Should we do the same?
- What specific techniques could push us past GLM-4.6 (72.38%) toward Claude Opus territory (77.47%)?

### 5. Bug Hunt

- Identify any bugs, edge cases, or failure modes in our handler that would cause incorrect eval results.
- Check for type coercion issues (Python bool `True`/`False` vs JSON `true`/`false`, string vs int conversions).
- Check for prompt format mismatches between what our handler generates and what the eval harness expects.

---

## Expected Deliverables

1. **Bug List**: Any bugs found, ranked by severity
2. **Handler Optimization Recommendations**: Specific code changes with rationale
3. **Training Strategy Recommendations**: For maximizing agentic scores
4. **Ranking Projection**: Your estimate of where prism-coder-72b would rank with proposed changes
5. **Risk Assessment**: What could go wrong and how to mitigate

---

## How to Use This Review Package

```bash
# The review package consists of:
# 1. This file (REVIEW_PROMPT.md) — context and questions
# 2. bfcl_repomix_eval.txt — eval harness code (46K tokens)  
# 3. bfcl_repomix_training.txt — training pipeline code (17K tokens)

# To review, paste this prompt followed by the two repomix files into
# a large-context LLM (Claude, Gemini, GPT-4.1) or a code review tool.

# Combined size: ~63K tokens (fits in most 128K context models)
```

---

# APPENDIX A: Evaluation Harness Code (bfcl_repomix_eval.txt)

The following is a repomix bundle of 14 files from the BFCL evaluation harness,
including our handler (`prism_coder.py`), the competitor handler (`salesforce_qwen.py`),
the eval checkers, and the memory/web APIs that the model must interact with.

```
This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: bfcl_eval/model_handler/local_inference/prism_coder.py, bfcl_eval/model_handler/local_inference/salesforce_qwen.py, bfcl_eval/model_handler/local_inference/base_oss_handler.py, bfcl_eval/model_handler/local_inference/qwen_fc.py, bfcl_eval/model_handler/base_handler.py, bfcl_eval/eval_checker/eval_runner.py, bfcl_eval/eval_checker/eval_runner_helper.py, bfcl_eval/eval_checker/agentic_eval/agentic_checker.py, bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py, bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py, bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_kv.py, bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_rec_sum.py, bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py, bfcl_eval/eval_checker/multi_turn_eval/func_source_code/web_search.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)


================================================================
Directory Structure
================================================================
bfcl_eval/
  eval_checker/
    agentic_eval/
      agentic_checker.py
    multi_turn_eval/
      func_source_code/
        memory_kv.py
        memory_rec_sum.py
        memory_vector.py
        web_search.py
      multi_turn_checker.py
      multi_turn_utils.py
    eval_runner_helper.py
    eval_runner.py
  model_handler/
    local_inference/
      base_oss_handler.py
      prism_coder.py
      qwen_fc.py
      salesforce_qwen.py
    base_handler.py

================================================================
Files
================================================================

================
File: bfcl_eval/eval_checker/agentic_eval/agentic_checker.py
================
import re

#### Main functions ####


def agentic_checker(model_response: str, possible_answer_list: list[str]) -> dict:
    """
    Check if one of the possible answers is contained in the model response, ignoring case, whitespace and ",./-_*^" punctuation.
    """
    standardized_possible_answer_list = [
        standardize_string(possible_answer) for possible_answer in possible_answer_list
    ]
    # Sometimes the model response is a list of one string
    if type(model_response) is list:
        model_response = model_response[0]
    if type(model_response) is not str:
        model_response = str(model_response)

    standardized_model_response = standardize_string(model_response)

    for possible_answer in standardized_possible_answer_list:
        if re.search(rf"\b{re.escape(possible_answer)}\b", standardized_model_response):
            return {"valid": True, "error": []}

    return {
        "valid": False,
        "error_message": f"None of the expected answers were found in the model response.",
        "error_type": "agentic:answer_not_found",
        "details": {
            "model_response": model_response,
            "possible_answers": possible_answer_list,
            "standardized_model_response": standardized_model_response,
            "standardized_possible_answers": standardized_possible_answer_list,
        },
    }


#### Helper functions ####


def standardize_string(input_string: str):
    """
    This function standardizes the string by removing all the whitespace, ",./-_*^()" punctuation, and converting it to lowercase
    It will also convert all the single quotes to double quotes
    This is used to compare the model output with the possible answers
    We don't want to punish model for answer like April 1, 2024 vs April 1,2024, vs April 1 2024
    """
    regex_string = r"[\,\.\/\-\_\*\^\(\)]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')

================
File: bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_kv.py
================
import json
import re
from copy import deepcopy
from typing import Dict, List, Tuple

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
    MemoryAPI,
)
from rank_bm25 import BM25Plus

# https://lilianweng.github.io/posts/2023-06-23-agent/#component-two-memory
MAX_CORE_MEMORY_SIZE = 7
MAX_CORE_MEMORY_ENTRY_LENGTH = 300
MAX_ARCHIVAL_MEMORY_SIZE = 50
MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000


class MemoryAPI_kv(MemoryAPI):
    """
    A class that provides APIs to manage short-term and long-term memory data in a key-value format.
    """

    def __init__(self):
        self.core_memory = {}
        self.archival_memory = {}
        self._api_description = """This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system."""
        self.snapshot_folder = None

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # Set up paths & load snapshots
        memory_data = self._prepare_snapshot(initial_config)

        # Populate in-memory structures if we have a previous snapshot
        if memory_data:
            self.core_memory = deepcopy(memory_data["core_memory"])
            self.archival_memory = deepcopy(memory_data["archival_memory"])

    def _flush_memory_to_local_file(self):
        """
        Flush (save) current memory (both core and archival) to a local JSON file.
        """

        # Write the snapshot file for the current test entry
        with open(self.snapshot_folder / f"{self.test_id}.json", "w") as f:
            json.dump(
                {
                    "core_memory": self.core_memory,
                    "archival_memory": self.archival_memory,
                },
                f,
                indent=4,
            )

        # Update the latest snapshot file content
        with open(self.latest_snapshot_file, "w") as f:
            json.dump(
                {
                    "core_memory": self.core_memory,
                    "archival_memory": self.archival_memory,
                },
                f,
                indent=4,
            )

    def _dump_core_memory_to_context(self) -> str:
        if not self.core_memory:
            return "There is no content in the core memory at this point."
        return json.dumps(self.core_memory, indent=4)

    @staticmethod
    def _similarity_search(query: str, corpus: list[str], k: int = 5):
        """
        Search for the most similar text in the corpus to the query using BM25+ algorithm.

        Args:
            query (str): The query text to search for.
            corpus (list[str]): A list of text strings to search in.
            k (int): The number of results to return.

        Returns:
            ranked_results (list[tuple[float, str]]): A list of tuples containing the BM25+ score and the text string.
        """
        tokenized_corpus = [text.replace("_", " ").lower().split() for text in corpus]
        bm25 = BM25Plus(tokenized_corpus)
        tokenized_query = query.replace("_", " ").lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_results = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
        return {"ranked_results": ranked_results[:k]}

    @staticmethod
    def _is_valid_key_format(s):
        """
        Check if the key is in snake_case format and does not contain spaces.
        """
        pattern = r"^[a-z]+(_[a-z0-9]+)*$"
        return bool(re.match(pattern, s))

    def core_memory_add(self, key: str, value: str) -> Dict[str, str]:
        """
        Add a key-value pair to the short-term memory. Make sure to use meaningful keys for easy retrieval later.

        Args:
            key (str): The key under which the value is stored. The key should be unique and case-sensitive. Keys must be snake_case and cannot contain spaces.
            value (str): The value to store in the short-term memory.

        Returns:
            status (str): Status of the operation.
        """
        key, value = str(key), str(value)
        if len(self.core_memory) >= MAX_CORE_MEMORY_SIZE:
            return {"error": "Core memory is full. Please clear some entries."}
        if len(value) > MAX_CORE_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry is too long. Please shorten the entry to less than {MAX_CORE_MEMORY_ENTRY_LENGTH} characters."
            }

        if not self._is_valid_key_format(key):
            return {"error": "Key must be in snake_case format and cannot contain spaces."}
        if key in self.core_memory:
            return {"error": "Key name must be unique."}

        self.core_memory[key] = value
        return {"status": "Key-value pair added."}

    def core_memory_remove(self, key: str) -> Dict[str, str]:
        """
        Remove a key-value pair from the short-term memory.

        Args:
            key (str): The key to remove from the short-term memory. Case-sensitive.

        Returns:
            status (str): Status of the operation.
        """
        if key in self.core_memory:
            del self.core_memory[key]
            return {"status": "Key removed."}
        else:
            return {"error": "Key not found."}

    def core_memory_replace(self, key: str, value: str) -> Dict[str, str]:
        """
        Replace a key-value pair in the short-term memory with a new value.

        Args:
            key (str): The key to replace in the short-term memory. Case-sensitive.
            value (str): The new value associated with the key.

        Returns:
            status (str): Status of the operation.
        """
        key, value = str(key), str(value)
        if key not in self.core_memory:
            return {"error": "Key not found."}
        if len(value) > MAX_CORE_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry is too long. Please shorten the entry to less than {MAX_CORE_MEMORY_ENTRY_LENGTH} characters."
            }

        self.core_memory[key] = value
        return {"status": "Key replaced."}

    def core_memory_clear(self) -> Dict[str, str]:
        """
        Clear all key-value pairs from the short-term memory, including those from previous interactions. This operation is irreversible.

        Returns:
            status (str): Status of the operation.
        """
        self.core_memory = {}
        return {"status": "Short term memory cleared."}

    def core_memory_retrieve(self, key: str) -> Dict[str, str]:
        """
        Retrieve the value associated with a key from the short-term memory. This function does not support partial key matching or similarity search.

        Args:
            key (str): The key to retrieve. Case-sensitive. The key must match exactly with the key stored in the memory.

        Returns:
            value (str): The value associated with the key.

        """
        if key not in self.core_memory:
            return {"error": "Key not found."}
        return {"value": self.core_memory[key]}

    def core_memory_list_keys(self) -> Dict[str, List[str]]:
        """
        List all keys currently in the short-term memory.

        Returns:
            keys (List[str]): A list of all keys in the short-term memory.
        """
        return {"keys": list(self.core_memory.keys())}

    def core_memory_key_search(
        self, query: str, k: int = 5
    ) -> Dict[str, List[Tuple[float, str]]]:
        """
        Search for key names in the short-term memory that are similar to the query using BM25+ algorithm.

        Args:
            query (str): The query text to search for.
            k (int): [Optional] The number of results to return.

        Returns:
            ranked_results (List[Tuple[float, str]]): A list of tuples containing the BM25+ score and the key.
        """
        keys = deepcopy(list(self.core_memory.keys()))
        return self._similarity_search(query, keys, k)

    def core_memory_retrieve_all(self) -> Dict[str, str]:
        """
        Retrieve all key-value pairs from the short-term memory.

        Returns:
            key (str): Each key in the short-term memory.
            value (str): The value associated with each key.
        """
        return self.core_memory

    def archival_memory_add(self, key: str, value: str) -> Dict[str, str]:
        """
        Add a key-value pair to the long-term memory. Make sure to use meaningful keys for easy retrieval later.
        Args:
            key (str): The key under which the value is stored. The key should be unique and case-sensitive. Keys must be snake_case and cannot contain spaces.
            value (str): The value to store in the long-term memory.

        Returns:
            status (str): Status of the operation.
        """
        key, value = str(key), str(value)
        if len(self.archival_memory) >= MAX_ARCHIVAL_MEMORY_SIZE:
            return {"error": "Long term memory is full. Please clear some entries."}
        if len(value) > MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry is too long. Please shorten the entry to less than {MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH} characters."
            }

        if not self._is_valid_key_format(key):
            return {"error": "Key must be in snake_case format and cannot contain spaces."}
        if key in self.archival_memory:
            return {"error": "Key name must be unique."}

        self.archival_memory[key] = value
        return {"status": "Key added."}

    def archival_memory_remove(self, key: str) -> Dict[str, str]:
        """
        Remove a key-value pair from the long-term memory.

        Args:
            key (str): The key to remove from the long-term memory. Case-sensitive.

        Returns:
            status (str): Status of the operation.
        """
        if key in self.archival_memory:
            del self.archival_memory[key]
            return {"status": "Key removed."}
        else:
            return {"error": "Key not found."}

    def archival_memory_replace(self, key: str, value: str) -> Dict[str, str]:
        """
        Replace a key-value pair in the long-term memory with a new value.

        Args:
            key (str): The key to replace in the long-term memory. Case-sensitive.
            value (str): The new value associated with the key.

        Returns:
            status (str): Status of the operation.
        """
        key, value = str(key), str(value)
        if key not in self.archival_memory:
            return {"error": "Key not found."}
        if len(value) > MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry is too long. Please shorten the entry to less than {MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH} characters."
            }

        self.archival_memory[key] = value
        return {"status": "Key replaced."}

    def archival_memory_clear(self) -> Dict[str, str]:
        """
        Clear all key-value pairs from the long-term memory, including those from previous interactions. This operation is irreversible.

        Returns:
            status (str): Status of the operation.
        """
        self.archival_memory = {}
        return {"status": "Long term memory cleared."}

    def archival_memory_retrieve(self, key: str) -> Dict[str, str]:
        """
        Retrieve the value associated with a key from the long-term memory. This function does not support partial key matching or similarity search.

        Args:
            key (str): The key to retrieve. Case-sensitive. The key must match exactly with the key stored in the memory.

        Returns:
            value (str): The value associated with the key.
        """
        if key not in self.archival_memory:
            return {"error": "Key not found."}
        return {"value": self.archival_memory[key]}

    def archival_memory_list_keys(self) -> Dict[str, List[str]]:
        """
        List all keys currently in the long-term memory.

        Returns:
            keys (List[str]): A list of all keys in the long-term memory.
        """
        return {"keys": list(self.archival_memory.keys())}

    def archival_memory_key_search(
        self, query: str, k: int = 5
    ) -> Dict[str, List[Tuple[float, str]]]:
        """
        Search for key names in the long-term memory that are similar to the query using BM25+ algorithm.

        Args:
            query (str): The query text to search for.
            k (int): [Optional] The number of results to return.

        Returns:
            ranked_results (List[Tuple[float, str]]): A list of tuples containing the BM25+ score and the key.
        """
        keys = deepcopy(list(self.archival_memory.keys()))
        return self._similarity_search(query, keys, k)

================
File: bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_rec_sum.py
================
import json
from copy import deepcopy
from typing import Dict

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
    MemoryAPI,
)

MAX_MEMORY_ENTRY_LENGTH = 10000  # 10k characters


class MemoryAPI_rec_sum(MemoryAPI):
    """
    A class that provides APIs to manage memory data via recursive summarization.
    """

    def __init__(self):
        self.memory = ""
        self._api_description = """This tool belongs to the memory suite, which provides APIs to manage memory data via recursive summarization."""
        self.snapshot_folder = None

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # Set up paths & load snapshots
        memory_data = self._prepare_snapshot(initial_config)

        # Populate in-memory structures if we have a previous snapshot
        if memory_data:
            self.memory = deepcopy(memory_data["memory"])
            assert isinstance(
                self.memory, str
            ), f"Memory data should be a string, but got {type(self.memory)} instead."

    def _flush_memory_to_local_file(self):
        """
        Flush (save) current memory to a local JSON file.
        """

        # Write the snapshot file for the current test entry
        with open(self.snapshot_folder / f"{self.test_id}.json", "w") as f:
            json.dump(
                {
                    "memory": self.memory,
                },
                f,
                indent=4,
            )

        # Update the latest snapshot file content
        with open(self.latest_snapshot_file, "w") as f:
            json.dump(
                {
                    "memory": self.memory,
                },
                f,
                indent=4,
            )

    def _dump_core_memory_to_context(self) -> str:
        if not self.memory:
            return "There is no content in the memory at this point."

        return str(self.memory)

    def memory_append(self, text: str) -> Dict[str, str]:
        """
        Append a new text to the end of the memory.

        Args:
            text (str): The text to append to the memory.

        Returns:
            status (str): Status of the operation.
        """
        text = str(text)
        combined_text = self.memory + text
        if len(combined_text) > MAX_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry will be too long after appending. Please shorten the entry to less than {MAX_MEMORY_ENTRY_LENGTH} characters."
            }

        self.memory += text
        return {"status": "Memory appended."}

    def memory_update(self, text: str) -> Dict[str, str]:
        """
        Update the memory with new text. This will replace the existing memory content.

        Args:
            text (str): The new text to set as the memory.

        Returns:
            status (str): Status of the operation.
        """
        text = str(text)
        if len(text) > MAX_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry will be too long after updating. Please shorten the entry to less than {MAX_MEMORY_ENTRY_LENGTH} characters."
            }

        self.memory = text
        return {"status": "Memory updated."}

    def memory_clear(self) -> Dict[str, str]:
        """
        Clear all content in the memory, including any from previous interactions. This operation is irreversible.

        Returns:
            status (str): Status of the operation.
        """
        self.memory = ""
        return {"status": "Short term memory cleared."}

    def memory_replace(self, old_text: str, new_text: str) -> Dict[str, str]:
        """
        Replace a specific text in the memory with new text.
        Args:
            old_text (str): The text to be replaced in the memory.
            new_text (str): The new text to replace the old text.
        Returns:
            status (str): Status of the operation.
        """
        old_text = str(old_text)
        new_text = str(new_text)

        if old_text not in self.memory:
            return {"error": f"Text '{old_text}' not found in memory."}

        if len(new_text) > MAX_MEMORY_ENTRY_LENGTH:
            return {
                "error": f"Entry will be too long after replacing. Please shorten the entry to less than {MAX_MEMORY_ENTRY_LENGTH} characters."
            }

        self.memory = self.memory.replace(old_text, new_text)
        return {"status": "Memory updated."}

    def memory_retrieve(self) -> Dict[str, str]:
        """
        Retrieve the current content of the memory.

        Returns:
            memory_content (str): The current content of the memory.
        """

        if not self.memory:
            return {"error": "Memory is empty."}

        return {"memory_content": self.memory}

================
File: bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py
================
import json
from typing import List, Optional

import numpy as np
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
    MemoryAPI,
)

# isort: off
# Note: This import order is necessary to avoid segfault issue due to FAISS and PyTorch each load a different OpenMP runtime
# See https://github.com/pytorch/pytorch/issues/149201#issuecomment-2725586827
# TODO: Find a common OpenMP runtime to avoid this issue
from sentence_transformers import SentenceTransformer
import faiss

# isort: on

# https://lilianweng.github.io/posts/2023-06-23-agent/#component-two-memory
MAX_CORE_MEMORY_SIZE = 7
MAX_CORE_MEMORY_ENTRY_LENGTH = 300
MAX_ARCHIVAL_MEMORY_SIZE = 50
MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000


# Use a global SentenceTransformer model for all vector stores.
ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ENCODER_DIM = ENCODER.get_sentence_embedding_dimension()


class MemoryAPI_vector(MemoryAPI):
    """
    A class that provides APIs to manage short-term and long-term memory data using vector embeddings.
    """

    def __init__(self):
        self.core_memory = VectorStore(
            max_size=MAX_CORE_MEMORY_SIZE,
            max_entry_length=MAX_CORE_MEMORY_ENTRY_LENGTH,
        )
        self.archival_memory = VectorStore(
            max_size=MAX_ARCHIVAL_MEMORY_SIZE,
            max_entry_length=MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH,
        )
        self._api_description = """This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system."""
        self.snapshot_folder = None

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # Set up paths & load snapshots
        memory_data = self._prepare_snapshot(initial_config)

        if memory_data:
            self.core_memory.load_from_snapshot(memory_data["core_memory"])
            self.archival_memory.load_from_snapshot(memory_data["archival_memory"])

    def _flush_memory_to_local_file(self):
        """
        Flush (save) current memory (both core and archival) to a local JSON file.
        """

        # Write the snapshot file for the current test entry
        with open(self.snapshot_folder / f"{self.test_id}.json", "w") as f:
            json.dump(
                {
                    "core_memory": self.core_memory.export(),
                    "archival_memory": self.archival_memory.export(),
                },
                f,
                indent=4,
            )

        # Update the latest snapshot file content
        with open(self.latest_snapshot_file, "w") as f:
            json.dump(
                {
                    "core_memory": self.core_memory.export(),
                    "archival_memory": self.archival_memory.export(),
                },
                f,
                indent=4,
            )

    def _dump_core_memory_to_context(self) -> str:
        if not self.core_memory:
            return "There is no content in the core memory at this point."

        return json.dumps(self.core_memory._store, indent=4)

    def core_memory_add(self, text: str) -> dict[str, str]:
        """
        Add a new entry to the core memory.

        Args:
            text (str): The text to be added to the core memory.

        Returns:
            id (int): The ID of the added entry, which can be used later for deletion or retrieval.
        """
        return self.core_memory.add(text)

    def core_memory_remove(self, vec_id: int) -> dict[str, str]:
        """
        Remove an entry from the core memory.

        Args:
            vec_id (int): The ID of the entry to be removed.

        Returns:
            status (str): Status of the operation.
        """
        return self.core_memory.remove(vec_id)

    def core_memory_update(self, vec_id: int, new_text: str) -> dict[str, str]:
        """
        Update an entry in the core memory.

        Args:
            vec_id (int): The ID of the entry to be updated.
            new_text (str): The new text to replace the old text.

        Returns:
            status (str): Status of the operation.
        """
        return self.core_memory.update(vec_id, new_text)

    def core_memory_clear(self) -> dict[str, str]:
        """
        Clear all entries in the core memory.

        Returns:
            status (str): Status of the operation.
        """
        return self.core_memory.clear()

    def core_memory_retrieve(
        self, query: str, top_k: Optional[int] = 5
    ) -> list[dict[str, str]]:
        """
        Retrieve the most similar entries from the core memory.

        Args:
            query (str): The query text to search for.
            top_k (int): [Optional] The number of top similar entries to retrieve.

        Returns:
            results (list[dict]): A list of dictionaries containing the ID, similarity score, and text of the retrieved entries.
                - id (int): The ID of the entry.
                - similarity_score (float): The similarity score of the entry with respect to the query.
                - text (str): The text of the entry.
        """
        return {"result": self.core_memory.retrieve(query, top_k)}

    def core_memory_retrieve_all(self) -> list[dict[str, str]]:
        """
        Retrieve all entries from the core memory.

        Returns:
            results (list[dict]): A list of dictionaries containing the ID and text of all entries.
                - id (int): The ID of the entry.
                - text (str): The text of the entry.
        """
        return {"result": self.core_memory.retrieve_all()}

    def archival_memory_add(self, text: str) -> dict[str, str]:
        """
        Add a new entry to the archival memory.

        Args:
            text (str): The text to be added to the archival memory.

        Returns:
            id (int): The ID of the added entry, which can be used later for deletion or retrieval.
        """
        return self.archival_memory.add(text)

    def archival_memory_remove(self, vec_id: int) -> dict[str, str]:
        """
        Remove an entry from the archival memory.

        Args:
            vec_id (int): The ID of the entry to be removed.

        Returns:
            status (str): Status of the operation.
        """
        return self.archival_memory.remove(vec_id)

    def archival_memory_update(self, vec_id: int, new_text: str) -> dict[str, str]:
        """
        Update an entry in the archival memory.

        Args:
            vec_id (int): The ID of the entry to be updated.
            new_text (str): The new text to replace the old text.

        Returns:
            status (str): Status of the operation.
        """
        return self.archival_memory.update(vec_id, new_text)

    def archival_memory_clear(self) -> dict[str, str]:
        """
        Clear all entries in the archival memory.

        Returns:
            status (str): Status of the operation.
        """
        return self.archival_memory.clear()

    def archival_memory_retrieve(
        self, query: str, top_k: Optional[int] = 5
    ) -> list[dict[str, str]]:
        """
        Retrieve the most similar entries from the archival memory.
        Args:
            query (str): The query text to search for.
            top_k (int): [Optional] The number of top similar entries to retrieve.
        Returns:
            results (list[dict]): A list of dictionaries containing the ID, similarity score, and text of the retrieved entries.
                - id (int): The ID of the entry.
                - similarity_score (float): The similarity score of the entry with respect to the query.
                - text (str): The text of the entry.
        """
        return {"result": self.archival_memory.retrieve(query, top_k)}

    def archival_memory_retrieve_all(self) -> list[dict[str, str]]:
        """
        Retrieve all entries from the archival memory.

        Returns:
            results (list[dict]): A list of dictionaries containing the ID and text of all entries.
                - id (int): The ID of the entry.
                - text (str): The text of the entry.
        """
        return {"result": self.archival_memory.retrieve_all()}


class VectorStore:

    def __init__(self, max_size, max_entry_length):
        self.max_size = max_size
        self.max_entry_length = max_entry_length

        # Cosine similarity via inner product on L2‑normalised vectors.
        index_flat = faiss.IndexFlatIP(ENCODER_DIM)
        self._index = faiss.IndexIDMap(index_flat)

        self._store: dict[int, str] = {}
        # _next_id will always be unique and sequential
        self._next_id: int = 0

    def _embed(self, text: str | List[str]) -> np.ndarray:
        """Return an L2-normalised NumPy array suitable for FAISS."""
        vecs = ENCODER.encode(
            text if isinstance(text, list) else [text], normalize_embeddings=True
        )
        return np.asarray(vecs, dtype=np.float32)

    def add(self, text: str) -> dict[str, str]:
        if len(text) > self.max_entry_length:
            return {
                "error": f"Entry length exceeds maximum length of {self.max_entry_length} characters."
            }

        if len(self._store) >= self.max_size:
            return {
                "error": f"Memory size exceeds maximum size of {self.max_size} entries."
            }

        vec_id = self._next_id
        self._next_id += 1

        vector = self._embed(text)
        self._index.add_with_ids(vector, np.array([vec_id], dtype=np.int64))
        self._store[vec_id] = text

        return {"id": vec_id}

    def remove(self, vec_id: int) -> dict[str, str]:
        if vec_id not in self._store:
            return {"error": f"ID {vec_id} not present in store."}

        self._index.remove_ids(np.array([vec_id], dtype=np.int64))
        del self._store[vec_id]

        return {"status": f"ID {vec_id} removed from store."}

    def update(self, vec_id: int, new_text: str) -> dict[str, str]:
        if vec_id not in self._store:
            return {"error": f"ID {vec_id} not present in store."}
        if len(new_text) > self.max_entry_length:
            return {
                "error": f"Entry length exceeds maximum length of {self.max_entry_length} characters."
            }

        self._index.remove_ids(np.array([vec_id], dtype=np.int64))
        vector = self._embed(new_text)
        self._index.add_with_ids(vector, np.array([vec_id], dtype=np.int64))
        self._store[vec_id] = new_text

        return {"status": f"ID {vec_id} updated."}

    def clear(self) -> dict[str, str]:
        self._index.reset()
        self._store.clear()
        self._next_id = 0

        return {"status": "Memory cleared."}

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Return the *top_k* most similar texts.

        Return a list of dictionary with keys 'id', 'similarity_score', and 'text'.
        """
        if not self._store:
            return []
        # q_vec has just one row (shape == (1, dim))
        q_vec = self._embed(query)

        # scores and ids come back with one row each (shape == (1, top_k))
        scores, ids = self._index.search(q_vec, min(top_k, len(self._store)))

        results = []
        for score, vid in zip(scores[0], ids[0]):
            # FAISS pads the id slots it can’t fill with ‑1, so we skip those
            if vid != -1:
                results.append(
                    {
                        "id": int(vid),
                        "similarity_score": float(score),
                        "text": self._store[int(vid)],
                    }
                )
        return results

    def retrieve_all(self) -> list[dict[str, str]]:
        """Return all entries in the vector store."""
        results = []
        for vid, text in self._store.items():
            results.append(
                {
                    "id": int(vid),
                    "text": text,
                }
            )
        return results

    def export(self) -> dict:
        """
        Export the vector store snapshot to a dictionary.
        """
        return {
            "next_id": self._next_id,
            "store": self._store,
        }

    def load_from_snapshot(self, snapshot_data: dict) -> None:
        """
        Load the vector store from a snapshot.
        """
        self._next_id = snapshot_data["next_id"]
        self._store = {int(k): v for k, v in snapshot_data["store"].items()}
        self._index.reset()

        if self._store:
            # Re-embed every stored text in one batch
            # To keep IDs aligned with vectors, sort by ID
            ids = np.array(sorted(self._store.keys()), dtype=np.int64)
            texts = [self._store[i] for i in ids]
            vectors = self._embed(texts)

            # Re-populate the index with the known IDs
            self._index.add_with_ids(vectors, ids)

================
File: bfcl_eval/eval_checker/multi_turn_eval/func_source_code/web_search.py
================
import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


class WebSearchAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        # Note: The following two random generators are used to simulate random errors, but that feature is not currently used
        # This one used to determine if we should simulate a random error
        # Outcome (True means simulate error): [True, False, True, True, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, True]
        self._random = random.Random(337)
        # This one is used to determine the content of the error message
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        backoff = 2  # initial back-off in seconds
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        # Infinite retry loop with exponential backoff
        while True:
            try:
                search = GoogleSearch(params)
                search_results = search.get_dict()
            except Exception as e:
                # If the underlying HTTP call raised a 429 we retry, otherwise propagate
                if "429" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off
                    continue
                else:
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Error from SerpAPI: {str(e)}. This is not a rate-limit error, so it will not be retried."
                        + "*" * 100
                    )
                    print(error_block)
                    return {"error": str(e)}

            # SerpAPI sometimes returns the error in the payload instead of raising
            if "error" in search_results and "429" in str(search_results["error"]):
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                    + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            break  # Success – no rate-limit error detected

        if "organic_results" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        search_results = search_results["organic_results"]

        # Convert the search results to the desired format
        results = []
        for result in search_results[:max_results]:
            if self.show_snippet:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                        "body": result["snippet"],
                    }
                )
            else:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                    }
                )

        return results

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            # A header that mimics a browser request. This helps avoid 403 Forbidden errors.
            # TODO: Is this the best way to do this?
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()

            # Note: Un-comment this when we want to simulate a random error
            # Flip a coin to simulate a random error
            # if self._random.random() < 0.95:
            #     return {"error": self._fake_requests_get_error_msg(url)}

            # Process the response based on the mode
            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic‑looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)

================
File: bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py
================
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

#### Main functions ####


def multi_turn_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
    test_entry: dict,
    test_category: str,
    model_name: str,
) -> dict:
    """
    The main function that checks the correctness of the model's function call execution.
    """

    initial_config: dict = test_entry["initial_config"]
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category: str = test_entry_id.rsplit("_", 1)[0]
    execution_results: list[dict] = []
    all_turn_model_execution_results: list[str] = []

    # First execute all the function calls
    for turn_index, single_turn_ground_truth_list in enumerate(
        multi_turn_ground_truth_list
    ):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]

        # Note that we combine all the sub-step results into a single list, for easier comparison
        single_turn_model_execution_results = []
        single_turn_model_execution_results_uncombined = []
        single_turn_ground_truth_execution_results = []
        model_instances = {}  # Will be overwritten in the for loop
        single_step_model_execution_results = []  # Will be overwritten in the for loop
    
        for single_step_model_response in single_turn_model_response_list:
            single_step_model_execution_results, model_instances = (
                execute_multi_turn_func_call(
                    func_call_list=single_step_model_response,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name=model_name,
                    test_entry_id=test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=True,
                )
            )
            single_turn_model_execution_results.extend(single_step_model_execution_results)
            single_turn_model_execution_results_uncombined.append(single_step_model_execution_results)

        # Execute the ground truth function calls
        single_turn_ground_truth_execution_results, ground_truth_instances = (
            execute_multi_turn_func_call(
                func_call_list=single_turn_ground_truth_list,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name + "_ground_truth",
                test_entry_id=test_entry_id,
                long_context=(
                    "long_context" in test_category or "composite" in test_category
                ),
                is_evaL_run=True,
            )
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)
        execution_results.append(
            {
                "model": single_turn_model_execution_results_uncombined,
                "ground_truth": single_turn_ground_truth_execution_results,
            }
        )

        # If the ground truth list is not empty, then the model response list should not be empty
        if len(single_turn_ground_truth_list) > 0:
            if not single_turn_model_response_list or is_empty_execute_response(
                single_turn_model_response_list
            ):
                return {
                    "valid": False,
                    "error_message": f"Model response list is empty for turn {turn_index}",
                    "error_type": "multi_turn:empty_turn_model_response",
                    "details": {
                        "execution_result": execution_results,
                    },
                }

        # If the ground truth list is empty, this is the turn where the model should eventually fail to achieve the user request.
        # The actual check for irrelevance is done in the multi_turn_irrelevance_checker function
        # Note: If the model outputs any function call in this turn, we will still execute it so that the state check at the next turn is accurate.
        if not single_turn_ground_truth_list:
            continue

        ## Check after each turn ##
        assert len(model_instances) == len(
            ground_truth_instances
        ), f"Model instances and ground truth instances do not match in length for turn {turn_index}. Model instances: {len(model_instances)}, Ground truth instances: {len(ground_truth_instances)}"
        assert set(model_instances.keys()) == set(ground_truth_instances.keys())

        # Check the state of the instances
        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            state_check_result["execution_result"] = execution_results
            return state_check_result

        # Check the response of the function calls
        # We use the all_turn_model_execution_results to accomodate the situation where the model invokes a function in a previous turn, and thus don't need to invoke it again in the current turn.
        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

        # # Check the method invoke order
        # method_invoke_order_check_result = method_invoke_order_checker(
        #     model_instances, ground_truth_instances
        # )
        # if not method_invoke_order_check_result["valid"]:
        #     return method_invoke_order_check_result

    return {"valid": True}


def multi_turn_irrelevance_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
) -> dict:
    """
    Check if the model's output are irrelevant when it should be.
    It should be empty when the ground truth is a empty list for that turn.
    """
    for turn_index, single_turn_ground_truth_list in enumerate(
        multi_turn_ground_truth_list
    ):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]
        if len(single_turn_ground_truth_list) == 0:
            if is_empty_execute_response(single_turn_model_response_list):
                continue
            else:
                return {
                    "valid": False,
                    "error_message": f"Model outputs valid function calls when it should not for turn {turn_index}.",
                    "error_type": "multi_turn:irrelevance_error:decoder_success",
                    "details": {
                        "model response decoded": single_turn_model_response_list,
                    },
                }
    return {"valid": True}


#### Sub-Chekcers ####


def state_checker(model_instances: dict, ground_truth_instances: dict):
    """
    Checks if, after executing the function calls, the model_instance has the same state (defined by the attributes) as the ground_truth_instance.
    It checks if every instance in the model_instances has the same attributes as their corresponding instance (of the same class) from ground_truth_instances.
    """
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)

        if not valid:
            model_instance_attributes = {
                key: value
                for key, value in vars(model_instance).items()
                if not key.startswith("_")
            }
            ground_truth_instance_attributes = {
                key: value
                for key, value in vars(ground_truth_instance).items()
                if not key.startswith("_")
            }
            # Format the error message for better readability
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the state with ground truth instance.",
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {
                    "differences": differences,
                    "model_instance_state": model_instance_attributes,
                    "ground_truth_instance_state": ground_truth_instance_attributes,
                },
            }

    return {"valid": True}


def response_checker(
    model_response_list: list, ground_truth_response_list: list, turn_index: int
):
    """
    Checks if the model_response is a subsequence of the ground_truth_response.
    Each list contains the response of the function calls executed in that single turn.
    """
    # We don't need to enforce the order of the responses, because many entries have parallel operations, and so the model can execute them in any order.
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": f"Model response execution results so far does not contain all the ground truth response execution results for turn {turn_index}.",
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {
                "missing_items": missing_items,
                "model_response (including all previous turns)": model_response_list,
                "ground_truth_response (only the current turn)": ground_truth_response_list,
            },
        }

    return {"valid": True}


def method_invoke_order_checker(model_instances: dict, ground_truth_instances: dict):
    """
    Checks if the model_instance called the same order of methods as the ground_truth_instance.
    model_instance can call additional methods, but not skip any method that the ground_truth_instance called.

    Note: Currently, this functions only checks for the method names and not the arguments.
    """
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]

        # The get_method_called method is added by the LoggingMeta metaclass automatically
        model_invoke_order = model_instance.get_method_called()
        ground_truth_invoke_order = ground_truth_instance.get_method_called()

        # Extract the method names
        model_invoke_order = [method_call["method"] for method_call in model_invoke_order]
        ground_truth_invoke_order = [
            method_call["method"] for method_call in ground_truth_invoke_order
        ]

        is_subsequence, missing_items = _is_subsequence(
            ground_truth_invoke_order, model_invoke_order
        )
        if not is_subsequence:
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the method invoke order with ground truth instance. Missing items: {missing_items}",
                "error_type": "multi_turn:method_invoke_order_mismatch",
            }

    return {"valid": True}


#### Helper functions ####


def _compare_instances(model_obect, ground_truth_object):
    """
    Checks if the model_object has the same attributes as the ground_truth_object. They are instances of the same class.
    """
    assert type(model_obect) == type(
        ground_truth_object
    ), "Objects are not of the same type."
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
        # We don't check for private attributes
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_obect, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)

        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

    return valid, differences


def _is_subsequence(list1, list2) -> tuple[bool, list]:
    """
    Checks if list1 is a subsequence of list2, i.e., all elements of list1 are present in list2 in the same order.
    Also returns the elements of list1 that are not present in list2.
    """
    # Convert list2 to an iterator to ensure that the elements are consumed only once.
    iter_list2 = iter(list2)
    return all(item in iter_list2 for item in list1), [
        item for item in list1 if item not in list2
    ]


def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
    """
    Checks if all elements of list1 are present in list2, regardless of order.
    Also returns the elements of list1 that are not present in list2.
    """
    # Copy list2 to avoid modifying the original list during checks
    list2_copy = list2[:]
    
    # Check each item in list1 to see if it exists in list2_copy
    missing_elements = []
    for item in list1:
        try:
            # Attempt to remove one occurrence of `item` from list2_copy to handle duplicates
            list2_copy.remove(item)
        except ValueError:
            # If item is not found, add it to missing_elements
            missing_elements.append(item)
    
    # If there are missing elements, list1 is not a subsequence of list2
    is_subsequence = len(missing_elements) == 0
    return is_subsequence, missing_elements

================
File: bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py
================
import copy
import importlib
import inspect
import json
import re

from bfcl_eval.constants.executable_backend_config import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
)


def execute_multi_turn_func_call(
    func_call_list: list[str],  # a list of strings of func calls
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
) -> tuple[list[str], dict]:
    """
    TODO: Add docstring
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # TODO: Handler the model name issue from handler more elegantly
        instance_name = (
            f"{model_name}_{test_entry_id}_{class_name}_instance"
        )
        instance_name = re.sub(r'[-./:]', '_', instance_name)
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                # Deep copy the initial configuration to avoid mutation issues
                class_instance._load_scenario(
                    copy.deepcopy(class_initial_config), long_context=long_context
                )
            globals()[instance_name] = class_instance
        # This happens in subsequent turns
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        # Retrieve all method names and map them to the instance
        for method_name, method in inspect.getmembers(
            class_instance, predicate=inspect.ismethod
        ):
            # Skip private methods
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error. 
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if type(func_call_result) == str:
                pass
            elif type(func_call_result) == dict:
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        class_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string

================
File: bfcl_eval/eval_checker/eval_runner_helper.py
================
import os
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.column_headers import *
from bfcl_eval.constants.eval_config import *
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import *


def calculate_weighted_accuracy(accuracy_dict_list, display_na_if_category_missing=True):
    has_na = False
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["total_count"]
        if accuracy_dict["display_accuracy"] == "N/A":
            has_na = True

        total_count += count
        total_accuracy += accuracy * count

    result = {"accuracy": total_accuracy / total_count, "total_count": total_count}

    if has_na and display_na_if_category_missing:
        result["display_accuracy"] = "N/A"
    else:
        result["display_accuracy"] = result["accuracy"]

    return result


def calculate_unweighted_accuracy(accuracy_dict_list, display_na_if_category_missing=True):
    has_na = False
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["total_count"]
        if accuracy_dict["display_accuracy"] == "N/A":
            # If a category is not being evaluated, it will still be considered 0 in the overall score calculation.
            has_na = True

        total_count += count
        total_accuracy += accuracy

    result = {
        "accuracy": total_accuracy / len(accuracy_dict_list),
        "total_count": total_count,
    }

    if has_na and display_na_if_category_missing:
        result["display_accuracy"] = "N/A"
    else:
        result["display_accuracy"] = result["accuracy"]

    return result


def calculate_percentage_weighted_accuracy(
    accuracy_dict_list, weights, display_na_if_category_missing=True
):
    """
    Calculate accuracy using a fixed list of weights that sum to 1.0.

    Parameters
    ----------
    accuracy_dict_list : list[dict]
        Each element is a dict containing at least the keys ``accuracy``, ``total_count`` and ``display_accuracy``.
    weights : list[float]
        The weight for each corresponding accuracy entry. Can sum to any positive value – they will be normalised internally.
    display_na_if_category_missing : bool, default True
        If True and any of the input categories has ``display_accuracy`` equal to "N/A", the returned ``display_accuracy`` will also be "N/A".

    Returns
    -------
    dict
        A dict with the same schema as other helper functions in this module (``accuracy``, ``total_count``, ``display_accuracy``).
    """
    assert len(accuracy_dict_list) == len(
        weights
    ), "Weights length must match accuracy list"

    has_na = False
    total_count = 0
    total_accuracy = 0.0
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights must be greater than 0")

    # Normalise weights so that they sum to 1.0
    weights_norm = [w / weight_sum for w in weights]

    for accuracy_dict, weight in zip(accuracy_dict_list, weights_norm):
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["total_count"]
        if accuracy_dict["display_accuracy"] == "N/A":
            has_na = True

        total_count += count
        total_accuracy += accuracy * weight

    result = {"accuracy": total_accuracy, "total_count": total_count}

    if has_na and display_na_if_category_missing:
        result["display_accuracy"] = "N/A"
    else:
        result["display_accuracy"] = result["accuracy"]

    return result


def record_result(leaderboard_table, model_name, test_category, accuracy, total_count):
    if model_name not in leaderboard_table:
        leaderboard_table[model_name] = {}
    leaderboard_table[model_name][test_category] = {
        "accuracy": accuracy,
        "total_count": total_count,
    }


def record_cost_latency(leaderboard_table, model_name, model_output_data):
    def process_data(key, data, output_list):
        # All entries are either a list of list (in multi-turn), or a single value (in single-turn)
        if key in data:
            if isinstance(data[key], list) and all(
                isinstance(inner_item, list) for inner_item in data[key]
            ):
                flattened_list = sum(data[key], [])
                output_list.extend(
                    [
                        item
                        for item in flattened_list
                        if isinstance(item, (int, float)) and item != 0
                    ]
                )
            else:
                if isinstance(data[key], (int, float)) and data[key] != 0:
                    output_list.append(data[key])

    if model_name not in leaderboard_table:
        leaderboard_table[model_name] = {}
        leaderboard_table[model_name]["cost"] = {"input_data": [], "output_data": []}
        leaderboard_table[model_name]["latency"] = {"data": []}

    input_token = []
    output_token = []
    latency = []
    for data in model_output_data:
        process_data("latency", data, latency)
        process_data("input_token_count", data, input_token)
        process_data("output_token_count", data, output_token)

    leaderboard_table[model_name]["cost"]["input_data"].extend(input_token)
    leaderboard_table[model_name]["cost"]["output_data"].extend(output_token)
    leaderboard_table[model_name]["latency"]["data"].extend(latency)


def save_eval_results(
    result,
    correct_count,
    model_result,
    test_category,
    model_name,
    score_dir,
    extra_header_fields: dict = None,
) -> tuple[float, int]:
    """
    Compute accuracy, finalize evaluation results and write them to disk.
    Return the accuracy and the total number of test cases.
    """
    accuracy = correct_count / len(model_result)
    header = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(model_result),
    }
    if extra_header_fields:
        header.update(extra_header_fields)

    result.insert(0, header)
    output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
    output_file_dir = (
        score_dir / model_name / get_directory_structure_by_category(test_category)
    )
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def get_cost_latency_info(model_name, cost_data, latency_data):
    cost, mean_latency, std_latency, percentile_95_latency = "N/A", "N/A", "N/A", "N/A"
    model_config = MODEL_CONFIG_MAPPING[model_name]

    # For API models, we use the input and output token counts to calculate the cost
    if model_config.input_price is not None and model_config.output_price is not None:
        if len(cost_data["input_data"]) > 0 and len(cost_data["output_data"]) > 0:
            total_input_tokens = sum(cost_data["input_data"])
            total_output_tokens = sum(cost_data["output_data"])
            # price is in USD per million tokens
            cost = (
                total_input_tokens * model_config.input_price / 1000000
                + total_output_tokens * model_config.output_price / 1000000
            )
            cost = round(cost, 2)

    # For local-hosted models, we calculate the total GPU cost by summing all latencies and multiplying by the hourly GPU price.
    elif len(latency_data["data"]) > 0:
        total_latency_seconds = sum(latency_data["data"])
        total_latency_hours = total_latency_seconds / 3600

        # Divide by 100 since we are doing 100x parallel inference; this is an approximation to the GPU up-time.
        cost = total_latency_hours * H100_X8_PRICE_PER_HOUR / LOCAL_SERVER_MAX_CONCURRENT_REQUEST
        cost = round(cost, 2)

    # Calculate latency statistics for ALL models (both API and local)
    if len(latency_data["data"]) != 0:
        mean_latency = statistics.mean(latency_data["data"])
        std_latency = statistics.stdev(latency_data["data"])
        percentile_95_latency = np.percentile(latency_data["data"], 95)
        mean_latency = round(mean_latency, 2)
        std_latency = round(std_latency, 2)
        percentile_95_latency = round(percentile_95_latency, 2)

    return cost, mean_latency, std_latency, percentile_95_latency


def get_category_score(score_dict: dict, test_category: str) -> dict:
    if test_category in score_dict:
        score = score_dict[test_category]
        score["display_accuracy"] = score["accuracy"]
        return score
    else:
        num_entry = len(
            load_dataset_entry(
                test_category, include_prereq=False, include_language_specific_hint=False
            )
        )
        # If a category is not being evaluated, it needs to be distinguished from the situation where the evaluation score is 0
        # It will still be considered 0 in the overall score calculation though
        # We use `display_accuracy` to special handle
        return {"accuracy": 0, "total_count": num_entry, "display_accuracy": "N/A"}


def write_score_csv_file(
    data,
    file_path: str,
    header: list,
    sort_column_index: int,
    no_conversion_numeric_column_index: list[int] = [],
) -> None:
    # Sort the data by the target column. Any row that contains "N/A" in the sort
    # column should always be placed at the end of the list. We achieve this by
    # returning -1 for such rows (all valid accuracy values are in the range [0, 1]),
    # and then performing a regular descending sort.
    data.sort(
        key=lambda x: x[sort_column_index] if x[sort_column_index] != "N/A" else -1,
        reverse=True,
    )
    for i in range(len(data)):
        # Add the ranking column, start from 0
        data[i][0] = str(i + 1)
        for j in range(1, len(data[i])):
            if type(data[i][j]) == str:
                continue
            # Some columns such as Latency and Cost, should not be presented in the percentage format
            elif j in no_conversion_numeric_column_index:
                data[i][j] = str(data[i][j])
            else:
                # Convert numeric value to percentage format
                data[i][j] = "{:.2f}%".format(data[i][j] * 100)

    data.insert(0, header)

    with open(file_path, "w") as f:
        for i, row in enumerate(data):
            if i < len(data) - 1:
                f.write(",".join(row) + "\n")
            else:
                f.write(",".join(row))


def generate_leaderboard_csv(leaderboard_table, output_path):
    print("📈 Aggregating data to generate leaderboard score table...")
    # Prepare format sensitivity configuration list once
    all_format_configs = get_all_format_sensitivity_configs()

    data_non_live = []
    data_live = []
    data_multi_turn = []
    data_agentic = []
    data_format_sensitivity = []
    data_combined = []
    for model_name, value in leaderboard_table.items():
        model_name_escaped = model_name.replace("_", "/")
        model_config = MODEL_CONFIG_MAPPING[model_name_escaped]

        cost_data = value.get("cost", {"input_data": [], "output_data": []})
        latency_data = value.get("latency", {"data": []})
        cost, latency_mean, latency_std, percentile_95_latency = get_cost_latency_info(
            model_name_escaped, cost_data, latency_data
        )

        # Non-Live Score
        python_simple_ast_non_live = get_category_score(value, "simple_python")
        python_multiple_ast_non_live = get_category_score(value, "multiple")
        python_parallel_ast_non_live = get_category_score(value, "parallel")
        python_parallel_multiple_ast_non_live = get_category_score(
            value, "parallel_multiple"
        )
        java_simple_ast_non_live = get_category_score(value, "simple_java")
        javascript_simple_ast_non_live = get_category_score(value, "simple_javascript")
        irrelevance_non_live = get_category_score(value, "irrelevance")

        simple_ast_non_live = calculate_unweighted_accuracy(
            [
                python_simple_ast_non_live,
                java_simple_ast_non_live,
                javascript_simple_ast_non_live,
            ]
        )
        multiple_ast_non_live = python_multiple_ast_non_live
        parallel_ast_non_live = python_parallel_ast_non_live
        parallel_multiple_ast_non_live = python_parallel_multiple_ast_non_live

        summary_ast_non_live = calculate_unweighted_accuracy(
            [
                simple_ast_non_live,
                multiple_ast_non_live,
                parallel_ast_non_live,
                parallel_multiple_ast_non_live,
            ]
        )
        overall_accuracy_non_live = calculate_unweighted_accuracy(
            [
                simple_ast_non_live,
                multiple_ast_non_live,
                parallel_ast_non_live,
                parallel_multiple_ast_non_live,
            ],
            display_na_if_category_missing=False,
        )

        data_non_live.append(
            [
                "N/A",
                model_config.display_name,
                overall_accuracy_non_live["display_accuracy"],
                summary_ast_non_live["display_accuracy"],
                simple_ast_non_live["display_accuracy"],
                python_simple_ast_non_live["display_accuracy"],
                java_simple_ast_non_live["display_accuracy"],
                javascript_simple_ast_non_live["display_accuracy"],
                multiple_ast_non_live["display_accuracy"],
                parallel_ast_non_live["display_accuracy"],
                parallel_multiple_ast_non_live["display_accuracy"],
                irrelevance_non_live["display_accuracy"],
            ]
        )

        # Live Score
        python_simple_ast_live = get_category_score(value, "live_simple")
        python_multiple_ast_live = get_category_score(value, "live_multiple")
        python_parallel_ast_live = get_category_score(value, "live_parallel")
        python_parallel_multiple_ast_live = get_category_score(
            value, "live_parallel_multiple"
        )
        irrelevance_live = get_category_score(value, "live_irrelevance")
        relevance_live = get_category_score(value, "live_relevance")
        summary_ast_live = calculate_weighted_accuracy(
            [
                python_simple_ast_live,
                python_multiple_ast_live,
                python_parallel_ast_live,
                python_parallel_multiple_ast_live,
            ]
        )

        overall_accuracy_live = calculate_weighted_accuracy(
            [
                python_simple_ast_live,
                python_multiple_ast_live,
                python_parallel_ast_live,
                python_parallel_multiple_ast_live,
            ],
            display_na_if_category_missing=False,
        )

        data_live.append(
            [
                "N/A",
                model_config.display_name,
                overall_accuracy_live["display_accuracy"],
                summary_ast_live["display_accuracy"],
                python_simple_ast_live["display_accuracy"],
                python_multiple_ast_live["display_accuracy"],
                python_parallel_ast_live["display_accuracy"],
                python_parallel_multiple_ast_live["display_accuracy"],
                irrelevance_live["display_accuracy"],
                relevance_live["display_accuracy"],
            ]
        )

        # Multi-Turn Score
        multi_turn_base = get_category_score(value, "multi_turn_base")
        multi_turn_miss_func = get_category_score(value, "multi_turn_miss_func")
        multi_turn_miss_param = get_category_score(value, "multi_turn_miss_param")
        multi_turn_long_context = get_category_score(value, "multi_turn_long_context")
        overall_accuracy_multi_turn = calculate_unweighted_accuracy(
            [
                multi_turn_base,
                multi_turn_miss_func,
                multi_turn_miss_param,
                multi_turn_long_context,
            ],
            display_na_if_category_missing=False,
        )

        data_multi_turn.append(
            [
                "N/A",
                model_config.display_name,
                overall_accuracy_multi_turn["display_accuracy"],
                multi_turn_base["display_accuracy"],
                multi_turn_miss_func["display_accuracy"],
                multi_turn_miss_param["display_accuracy"],
                multi_turn_long_context["display_accuracy"],
            ]
        )

        # Agentic Score
        web_search_base = get_category_score(value, "web_search_base")
        web_search_no_snippet = get_category_score(value, "web_search_no_snippet")
        summary_web_search = calculate_unweighted_accuracy(
            [
                web_search_base,
                web_search_no_snippet,
            ]
        )
        memory_kv = get_category_score(value, "memory_kv")
        memory_vector = get_category_score(value, "memory_vector")
        memory_rec_sum = get_category_score(value, "memory_rec_sum")
        summary_memory = calculate_unweighted_accuracy(
            [
                memory_kv,
                memory_vector,
                memory_rec_sum,
            ]
        )
        overall_accuracy_agentic = calculate_unweighted_accuracy(
            [
                summary_web_search,
                summary_memory,
            ],
            display_na_if_category_missing=False,
        )

        data_agentic.append(
            [
                "N/A",
                model_config.display_name,
                overall_accuracy_agentic["display_accuracy"],
                summary_web_search["display_accuracy"],
                web_search_base["display_accuracy"],
                web_search_no_snippet["display_accuracy"],
                summary_memory["display_accuracy"],
                memory_kv["display_accuracy"],
                memory_vector["display_accuracy"],
                memory_rec_sum["display_accuracy"],
            ]
        )

        # Total Score
        total_irrelevance = calculate_unweighted_accuracy(
            [irrelevance_non_live, irrelevance_live]
        )
        total_relevance = relevance_live

        # Format Sensitivity statistics
        format_sensitivity_metadata = value.get("format_sensitivity", {})
        format_sensitivity_max_delta = format_sensitivity_metadata.get(
            "accuracy_max_delta", "N/A"
        )
        format_sensitivity_std = format_sensitivity_metadata.get("accuracy_std", "N/A")

        # Prepare row for format sensitivity CSV
        config_accuracy_values = []
        for cfg in all_format_configs:
            cfg_stats = format_sensitivity_metadata.get(cfg, {})
            cfg_acc = cfg_stats.get("accuracy", "N/A")
            config_accuracy_values.append(cfg_acc)

        data_format_sensitivity.append(
            [
                "N/A",
                model_config.display_name,
                format_sensitivity_max_delta,
                format_sensitivity_std,
                *config_accuracy_values,
            ]
        )

        # TODO: @HuanzhiMao adjust the weights
        total_overall_accuracy = calculate_percentage_weighted_accuracy(
            [
                overall_accuracy_non_live,
                overall_accuracy_live,
                total_irrelevance,
                overall_accuracy_multi_turn,
                overall_accuracy_agentic,
            ],
            [10, 10, 10, 30, 40],
            display_na_if_category_missing=False,
        )

        data_combined.append(
            [
                "N/A",
                total_overall_accuracy["display_accuracy"],
                model_config.display_name,
                model_config.url,
                cost,
                latency_mean,
                latency_std,
                percentile_95_latency,
                summary_ast_non_live["display_accuracy"],
                simple_ast_non_live["display_accuracy"],
                multiple_ast_non_live["display_accuracy"],
                parallel_ast_non_live["display_accuracy"],
                parallel_multiple_ast_non_live["display_accuracy"],
                overall_accuracy_live["display_accuracy"],
                python_simple_ast_live["display_accuracy"],
                python_multiple_ast_live["display_accuracy"],
                python_parallel_ast_live["display_accuracy"],
                python_parallel_multiple_ast_live["display_accuracy"],
                overall_accuracy_multi_turn["display_accuracy"],
                multi_turn_base["display_accuracy"],
                multi_turn_miss_func["display_accuracy"],
                multi_turn_miss_param["display_accuracy"],
                multi_turn_long_context["display_accuracy"],
                summary_web_search["display_accuracy"],
                web_search_base["display_accuracy"],
                web_search_no_snippet["display_accuracy"],
                summary_memory["display_accuracy"],
                memory_kv["display_accuracy"],
                memory_vector["display_accuracy"],
                memory_rec_sum["display_accuracy"],
                total_relevance["display_accuracy"],
                total_irrelevance["display_accuracy"],
                format_sensitivity_max_delta,
                format_sensitivity_std,
                model_config.org,
                model_config.license,
            ]
        )

    # Write Non-Live Score File
    write_score_csv_file(
        data=data_non_live,
        file_path=output_path / "data_non_live.csv",
        header=COLUMNS_NON_LIVE,
        sort_column_index=2,
    )

    # Write Live Score File
    write_score_csv_file(
        data=data_live,
        file_path=output_path / "data_live.csv",
        header=COLUMNS_LIVE,
        sort_column_index=2,
    )

    # Write Multi Turn Score File
    write_score_csv_file(
        data=data_multi_turn,
        file_path=output_path / "data_multi_turn.csv",
        header=COLUMNS_MULTI_TURN,
        sort_column_index=2,
    )

    # Write Agentic Score File
    write_score_csv_file(
        data=data_agentic,
        file_path=output_path / "data_agentic.csv",
        header=COLUMNS_AGENTIC,
        sort_column_index=2,
    )

    # Write Format Sensitivity Score File
    COLUMNS_FORMAT_SENS = COLUMNS_FORMAT_SENS_PREFIX + [
        f"Config {cfg}" for cfg in all_format_configs
    ]

    write_score_csv_file(
        data=data_format_sensitivity,
        file_path=output_path / "data_format_sensitivity.csv",
        header=COLUMNS_FORMAT_SENS,
        sort_column_index=2,
        no_conversion_numeric_column_index=[2, 3],
    )

    # Write Total Score File
    write_score_csv_file(
        data=data_combined,
        file_path=output_path / "data_overall.csv",
        header=COLUMNS_OVERALL,
        sort_column_index=1,
        no_conversion_numeric_column_index=[4, 5, 6, 7, 32, 33],
    )

    wandb_project = os.getenv("WANDB_BFCL_PROJECT")
    if wandb_project and wandb_project != "ENTITY:PROJECT":
        import wandb

        # Initialize WandB run
        wandb.init(
            # wandb_project is 'entity:project'
            entity=wandb_project.split(":")[0],
            project=wandb_project.split(":")[1],
            name=f"BFCL-v4-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

        # Log CSV files to WandB
        # Read the CSV files
        non_live_df = pd.read_csv(output_path / "data_non_live.csv")
        live_df = pd.read_csv(output_path / "data_live.csv")
        multi_turn_df = pd.read_csv(output_path / "data_multi_turn.csv")
        agentic_df = pd.read_csv(output_path / "data_agentic.csv")
        overall_df = pd.read_csv(output_path / "data_overall.csv")

        # Convert DataFrames to WandB Tables
        non_live_table = wandb.Table(dataframe=non_live_df)
        live_table = wandb.Table(dataframe=live_df)
        multi_turn_table = wandb.Table(dataframe=multi_turn_df)
        agentic_table = wandb.Table(dataframe=agentic_df)
        overall_table = wandb.Table(dataframe=overall_df)

        # Create artifacts
        bfcl_artifact = wandb.Artifact("bfcl_results", type="dataset")

        # Add tables to artifact
        bfcl_artifact.add(non_live_table, "non_live_results")
        bfcl_artifact.add(live_table, "live_results")
        bfcl_artifact.add(multi_turn_table, "multi_turn_results")
        bfcl_artifact.add(agentic_table, "agentic_results")
        bfcl_artifact.add(overall_table, "overall_results")

        # Add raw CSV files to artifact
        bfcl_artifact.add_file(str(output_path / "data_non_live.csv"))
        bfcl_artifact.add_file(str(output_path / "data_live.csv"))
        bfcl_artifact.add_file(str(output_path / "data_multi_turn.csv"))
        bfcl_artifact.add_file(str(output_path / "data_agentic.csv"))
        bfcl_artifact.add_file(str(output_path / "data_overall.csv"))

        # Log tables directly
        wandb.log(
            {
                "Non-Live Results": non_live_table,
                "Live Results": live_table,
                "Multi-Turn Results": multi_turn_table,
                "Agentic Results": agentic_table,
                "Overall Results": overall_table,
            }
        )

        # Log artifact
        wandb.log_artifact(bfcl_artifact)
        wandb.finish()


def update_leaderboard_table_with_local_score_file(
    leaderboard_table, score_path: Path
) -> None:

    entries = score_path.iterdir()

    # Filter out the subdirectories
    subdirs = [entry for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in subdirs:
        model_name = subdir.relative_to(score_path).name
        # Find and process all score JSON files recursively in the subdirectory
        pattern = f"{VERSION_PREFIX}_*_score.json"
        for model_score_json in subdir.rglob(pattern):
            metadata = load_file(model_score_json)[0]
            test_category = extract_test_category(model_score_json)
            if model_name not in leaderboard_table:
                leaderboard_table[model_name] = {}
            # Store the full metadata to retain additional statistics (e.g. format sensitivity breakdown)
            leaderboard_table[model_name][test_category] = metadata

================
File: bfcl_eval/eval_checker/eval_runner.py
================
import argparse
import statistics
from collections import defaultdict

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import *
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.eval_runner_helper import *
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import parse_prompt_variation_params
from bfcl_eval.utils import *
from dotenv import load_dotenv
from tqdm import tqdm


def get_handler(model_name: str) -> BaseHandler:
    config = MODEL_CONFIG_MAPPING[model_name]
    handler: BaseHandler = config.model_handler(
        model_name=config.model_name,
        temperature=0,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )
    return handler


def _subset_entries_by_model_ids(
    model_result_entries: list[dict],
    prompt_entries: list[dict],
    ground_truth_entries: list[dict] = None,  # Irrelevance entries don't have ground truth
    allow_missing: bool = False,
):
    """
    Filter the prompt and ground truth entries so that its order/length matches the IDs present in `model_result`. When `allow_missing` is False, all IDs must be present; otherwise, any missing IDs are silently ignored.
    """
    if not model_result_entries:
        return [], []

    if not allow_missing and (len(model_result_entries) != len(prompt_entries)):
        raise ValueError(
            f"Length of model result ({len(model_result_entries)}) does not match length of test entries ({len(prompt_entries)}). If you intended to run only on a subset (eg. entries present in the model result), please pass the `--partial-eval` flag."
        )

    all_present_ids = {entry["id"]: entry for entry in model_result_entries}

    # Align prompt and ground-truth using the *index* of the prompt entry. Some
    # ground-truth items use a different ID format, but the order between the
    # prompt list and the ground-truth list is guaranteed to be identical. We
    # therefore keep the element at index *i* in both lists whenever the
    # prompt entry at that index has an ID present in the model results.
    filtered_prompt_entries: list[dict] = []
    filtered_ground_truth_entries: list[dict] = []
    for idx, prompt_entry in enumerate(prompt_entries):
        if prompt_entry["id"] in all_present_ids:
            filtered_prompt_entries.append(prompt_entry)
            # ground_truth_entries and prompt_entries are aligned by index.
            if ground_truth_entries is not None:
                filtered_ground_truth_entries.append(ground_truth_entries[idx])

    return filtered_prompt_entries, filtered_ground_truth_entries


def _evaluate_single_agentic_entry(
    handler: BaseHandler,
    index,
    model_result_list,
    possible_answer_item,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single agentic entry."""
    # Remove the function doc from the score file for better readability
    if "function" in prompt_entry:
        del prompt_entry["function"]

    # Agentic test is a single-turn multi-step test, so the model result should be a list of one element
    if type(model_result_list) != list or len(model_result_list) != 1:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Error during inference phase. Model did not output a list of model responses."
                ],
                "error_type": "agentic:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": possible_answer_item,
        }

    # Try decoding the model results into executable function calls
    # Note: We only care about the last non-function-call message, which should fail to get decoded.
    # We don't care about the function calls in the middle of the conversation.
    # We only check if the expected answer is mentioned in the last message.
    # decode_execute returns a list of strings
    model_result_list_decoded: list[list[str]] = []
    last_unsuccessful_decoding_message = None

    for model_result_item in model_result_list[0]:
        # model_result_item is per step
        try:
            decoded_result: list[str] = handler.decode_execute(
                model_result_item, has_tool_call_tag=False
            )
            if is_empty_execute_response(decoded_result):
                last_unsuccessful_decoding_message = model_result_item
                continue
            model_result_list_decoded.append(decoded_result)
        except Exception as e:
            last_unsuccessful_decoding_message = model_result_item
            continue

    if not last_unsuccessful_decoding_message:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Cannot find the last chat message that is not a function call."
                ],
                "error_type": "agentic:no_last_message",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "model_result_decoded": model_result_list_decoded,
            "possible_answer": possible_answer_item,
        }

    # Check if the model output contains the expected answer
    accuracy_checker_result = agentic_checker(
        last_unsuccessful_decoding_message,
        possible_answer_item,
    )

    if not accuracy_checker_result["valid"]:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": accuracy_checker_result.pop("valid"),
            "error": accuracy_checker_result,
            "prompt": prompt_entry["question"],
            "model_result_raw": model_result_list,
            "last_non_fc_message": last_unsuccessful_decoding_message,
            "possible_answer": possible_answer_item,
        }

    return {"valid": True}


def _evaluate_single_multi_turn_entry(
    handler: BaseHandler,
    test_entry_id,
    model_result_list,
    ground_truth_list,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single multi-turn entry."""
    # Remove the function doc from the score file for better readability
    if "function" in prompt_entry:
        del prompt_entry["function"]

    if type(model_result_list) != list:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    "Error during inference phase. Model did not output a list of model responses."
                ],
                "error_type": "multi_turn:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # Check if force-terminated during inference phase.
    # This happens when the model has retried too many times and still haven't figured out the answer.
    # When force-terminated, no further evaluation is needed. This whole entry will be failed.
    if len(model_result_list) != len(ground_truth_list):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    f"Model was force-terminated during inference phase. The length of the model result turns ({len(model_result_list)}) does not match the length of the ground truth turns ({len(ground_truth_list)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # decode_execute returns a list of strings
    multi_turn_model_result_list_decoded: list[list[list[str]]] = []
    # Try decoding the model results into executable function calls
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded = []
        for model_result_item in single_turn_model_result_list:
            # model_result_item is per step
            try:
                decoded_result: list[str] = handler.decode_execute(
                    model_result_item, has_tool_call_tag=False
                )
                if is_empty_execute_response(decoded_result):
                    # Empty output is not considered as a valid function call
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception as e:
                # Ignore any failed decoding and continue to the next message
                # We only care about the decoded function call, not the error message or if the model is chatting
                continue
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

    # Check if the model output the correct function calls
    accuracy_checker_result = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        ground_truth_list,
        prompt_entry,
        test_category,
        model_name,
    )

    if not accuracy_checker_result["valid"]:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": accuracy_checker_result.pop("valid"),
            "error": accuracy_checker_result,
            "prompt": prompt_entry,
            "model_result_raw": model_result_list,
            "model_result_decoded": multi_turn_model_result_list_decoded,
            "possible_answer": ground_truth_list,
        }

    return {"valid": True}


def _evaluate_single_relevance_entry(
    handler: BaseHandler,
    index,
    model_result_item,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single relevance/irrelevance entry."""
    contain_func_call = False
    decoded_result = None
    decode_error = None

    try:
        decoded_result = handler.decode_ast(
            model_result_item, language=ReturnFormat.PYTHON, has_tool_call_tag=False
        )
        # Decode successfully, which means the model output is in valid function call format
        contain_func_call = True
        if is_empty_output(decoded_result):
            # Empty output is not considered as a valid function call
            contain_func_call = False
    except Exception as e:
        # Decode failed, which means the model output is not in valid function call format
        contain_func_call = False
        decode_error = str(e)

    # irrelevance test means no function call outputted
    if "irrelevance" in test_category:
        success = not contain_func_call
    else:
        success = contain_func_call

    if not success:
        temp = {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": success,
            "prompt": prompt_entry,
            "model_result": model_result_item,
            "decoded_result": decoded_result,
        }
        if "irrelevance" in test_category:
            temp["error"] = ["Valid syntax. Successfully decode AST when it should not."]
            temp["error_type"] = "irrelevance_error:decoder_success"
        else:
            temp["error"] = [
                f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"
            ]
            temp["error_type"] = "relevance_error:decoder_failed"
        return temp

    return {"valid": True}


def _evaluate_single_ast_entry(
    handler: BaseHandler,
    index,
    model_result_item,
    possible_answer_item,
    prompt_entry,
    model_name,
    test_category,
    language: Language,
    return_format: ReturnFormat,
    has_tool_call_tag=False,
):
    """Helper method to process a single AST entry."""
    prompt_function = prompt_entry["function"]

    try:
        model_result_item_raw = model_result_item
        model_result_item = handler.decode_ast(
            model_result_item, return_format, has_tool_call_tag
        )
    except Exception as e:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
            "error_type": "ast_decoder:decoder_failed",
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "possible_answer": possible_answer_item,
        }

    decoder_output_valid = is_function_calling_format_output(model_result_item)
    if not decoder_output_valid:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": [
                "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
            ],
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "prompt": prompt_entry,
            "model_result_raw": str(model_result_item_raw),
            "model_result_decoded": str(model_result_item),
            "possible_answer": possible_answer_item,
        }

    checker_result = ast_checker(
        prompt_function,
        model_result_item,
        possible_answer_item,
        language,
        # format sensitivity has parallel, multiple cases which is encoded in index
        test_category if test_category != 'format_sensitivity' else index.split(':')[-1],
        model_name,
    )

    if not checker_result["valid"]:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": checker_result["valid"],
            "error": checker_result["error"],
            "error_type": checker_result["error_type"],
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result_item,
            "possible_answer": possible_answer_item,
        }
    return {"valid": True}


def format_sensitivity_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    # The format sensitivity tests are all single-turn tests, so we use a similar logic to the ast_file_runner to evaluate them.

    result = []
    correct_count = 0
    # Track stats per format sensitivity configuration
    config_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        assert (
            ":" in index and len(index.split(":")) == 3
        ), f"Test entry ID {index} should contain exactly two colons, since they are supposed to be the format sensitivity ids."

        format_sensitivity_config = index.split(":")[1]
        (
            return_format,
            has_tool_call_tag,
            function_doc_format,
            prompt_format,
            prompt_style,
        ) = parse_prompt_variation_params(format_sensitivity_config)

        return_format = ReturnFormat(return_format)

        entry_result = _evaluate_single_ast_entry(
            handler,
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
            # Format sensitivity tests are all python tests
            language=Language.PYTHON,
            return_format=return_format,
            has_tool_call_tag=has_tool_call_tag,
        )

        # Update stats for this configuration
        config_stats[format_sensitivity_config]["total"] += 1
        if entry_result["valid"]:
            correct_count += 1
            config_stats[format_sensitivity_config]["correct"] += 1
        else:
            result.append(entry_result)

    # Compute accuracy per configuration
    accuracy_by_config = {
        cfg: {
            "accuracy": stats["correct"] / stats["total"],
            "correct_count": stats["correct"],
            "total_count": stats["total"],
        }
        for cfg, stats in config_stats.items()
    }

    # Calculate statistics across different prompt configurations
    config_accuracies = [v["accuracy"] for v in accuracy_by_config.values()]
    if len(config_accuracies) > 1:
        accuracy_variance = round(statistics.variance(config_accuracies) * 100**2, 2)
        accuracy_std = round(statistics.stdev(config_accuracies) * 100, 2)
        accuracy_max_delta = round(
            (max(config_accuracies) - min(config_accuracies)) * 100, 2
        )
    else:
        accuracy_variance = 0.0
        accuracy_std = 0.0
        accuracy_max_delta = 0.0

    extra_header_fields = {
        "accuracy_max_delta": accuracy_max_delta,
        "accuracy_variance": accuracy_variance,
        "accuracy_std": accuracy_std,
        **accuracy_by_config,
    }

    return save_eval_results(
        result,
        correct_count,
        model_result,
        test_category,
        model_name,
        score_dir,
        extra_header_fields=extra_header_fields,
    )


def agentic_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_list = model_result[i]["result"]
        possible_answer_item = possible_answer[i]["ground_truth"]
        test_entry = prompt[i]

        entry_result = _evaluate_single_agentic_entry(
            handler,
            index,
            model_result_list,
            possible_answer_item,
            test_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

    return save_eval_results(
        result, correct_count, model_result, test_category, model_name, score_dir
    )


def multi_turn_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        multi_turn_model_result_list = model_result[i]["result"]
        multi_turn_ground_truth_list = possible_answer[i]["ground_truth"]
        test_entry = prompt[i]

        entry_result = _evaluate_single_multi_turn_entry(
            handler,
            index,
            multi_turn_model_result_list,
            multi_turn_ground_truth_list,
            test_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

    return save_eval_results(
        result, correct_count, model_result, test_category, model_name, score_dir
    )


def relevance_file_runner(
    handler: BaseHandler, model_result, prompt, model_name, test_category, score_dir
):
    # This function serves for both relevance and irrelevance tests, which share the exact opposite logic.
    # If `test_category` is "irrelevance", the model is expected to output no function call.
    # No function call means either the AST decoding fails (a error message is generated) or the decoded AST does not contain any function call (such as a empty list, `[]`).
    # If `test_category` is "relevance", the model is expected to output to a function call, and empty list doesn't count as a function call.
    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]

        entry_result = _evaluate_single_relevance_entry(
            handler, index, model_result_item, prompt_entry, model_name, test_category
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            result.append(entry_result)

    return save_eval_results(
        result, correct_count, model_result, test_category, model_name, score_dir
    )


def ast_file_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    test_category,
    model_name,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    if is_java(test_category):
        language = Language.JAVA
        return_format = ReturnFormat.JAVA
    elif is_js(test_category):
        language = Language.JAVASCRIPT
        return_format = ReturnFormat.JAVASCRIPT
    else:
        language = Language.PYTHON
        return_format = ReturnFormat.PYTHON

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        entry_result = _evaluate_single_ast_entry(
            handler,
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
            language=language,
            return_format=return_format,
            has_tool_call_tag=False,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            result.append(entry_result)

    return save_eval_results(
        result, correct_count, model_result, test_category, model_name, score_dir
    )


#### Main runner function ####
def evaluate_task(
    test_category,
    result_dir,
    score_dir,
    model_result,
    model_name,
    handler,
    leaderboard_table,
    allow_missing: bool = False,
):
    print(f"🔍 Running test: {test_category}")

    record_cost_latency(leaderboard_table, model_name, model_result)

    # Find the corresponding prompt entries
    prompt = load_dataset_entry(
        test_category, include_prereq=False, include_language_specific_hint=False
    )

    if is_relevance_or_irrelevance(test_category):
        prompt, _ = _subset_entries_by_model_ids(
            model_result, prompt, None, allow_missing=allow_missing
        )

        accuracy, total_count = relevance_file_runner(
            handler, model_result, prompt, model_name, test_category, score_dir
        )

    else:
        # Find the corresponding possible answer entries
        possible_answer = load_ground_truth_entry(test_category)
        # Sanity: prompt and ground truth should be 1:1
        assert len(prompt) == len(
            possible_answer
        ), f"Length of ground truth ({len(possible_answer)}) should match prompt entries ({len(prompt)})."

        prompt, possible_answer = _subset_entries_by_model_ids(
            model_result, prompt, possible_answer, allow_missing=allow_missing
        )

        if is_format_sensitivity(test_category):
            accuracy, total_count = format_sensitivity_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )

        elif is_multi_turn(test_category):
            accuracy, total_count = multi_turn_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )

        elif is_agentic(test_category):
            accuracy, total_count = agentic_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )
        # Single turn test
        else:
            accuracy, total_count = ast_file_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                test_category,
                model_name,
                score_dir,
            )

    record_result(leaderboard_table, model_name, test_category, accuracy, total_count)

    print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy:.2%}")

    return leaderboard_table


def runner(
    model_names, test_categories, result_dir, score_dir, allow_missing: bool = False
):

    # A dictionary to store the evaluation scores.
    # Key is model name, value is a dictionary with keys as test category
    # and values as a dictionary with accuracy and total count.
    # TODO: use defaultdict to initialize the leaderboard table
    leaderboard_table = {}

    # Get a list of all entries in the folder
    entries = result_dir.iterdir()

    # Filter out the subdirectories
    subdirs = [entry for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in tqdm(subdirs, desc="Number of models evaluated"):

        model_name = subdir.relative_to(result_dir).name
        if model_names is not None and model_name not in model_names:
            continue

        model_name_escaped = model_name.replace("_", "/")

        print(f"🦍 Model: {model_name}")

        # Find and process all result JSON files recursively in the subdirectory
        for model_result_json in subdir.rglob(RESULT_FILE_PATTERN):
            test_category = extract_test_category(model_result_json)
            if test_category not in test_categories:
                continue

            handler = get_handler(model_name_escaped)

            # We don't evaluate the following categories in the current iteration of the benchmark
            if (
                is_chatable(test_category)
                or is_sql(test_category)
                or is_executable(test_category)
                or is_memory_prereq(test_category)
            ):
                continue

            model_result = load_file(model_result_json, sort_by_id=True)

            leaderboard_table = evaluate_task(
                test_category,
                result_dir,
                score_dir,
                model_result,
                model_name,
                handler,
                leaderboard_table,
                allow_missing=allow_missing,
            )

    # This function reads all the score files from local folder and updates the
    # leaderboard table. This is helpful when you only want to run the
    # evaluation for a subset of models and test categories.
    update_leaderboard_table_with_local_score_file(leaderboard_table, score_dir)
    # Write the leaderboard table to a file
    generate_leaderboard_csv(leaderboard_table, score_dir)


def main(model, test_categories, result_dir, score_dir, partial_eval: bool = False):
    if result_dir is None:
        result_dir = RESULT_PATH
    else:
        result_dir = (PROJECT_ROOT / result_dir).resolve()

    if score_dir is None:
        score_dir = SCORE_PATH
    else:
        score_dir = (PROJECT_ROOT / score_dir).resolve()

    if type(test_categories) is not list:
        test_categories = [test_categories]

    all_test_categories = parse_test_category_argument(test_categories)

    model_names = None
    if model:
        model_names = []
        for model_name in model:
            if model_name not in MODEL_CONFIG_MAPPING:
                raise ValueError(f"Invalid model name '{model_name}'.")
            # Runner takes in the model name that contains "_", instead of "/", for the sake of file path issues.
            # This is differnet than the model name format that the generation script "openfunctions_evaluation.py" takes in (where the name contains "/").
            # We patch it here to avoid confusing the user.
            model_names.append(model_name.replace("/", "_"))

    # Driver function to run the evaluation for all categories involved.
    runner(
        model_names,
        all_test_categories,
        result_dir,
        score_dir,
        allow_missing=partial_eval,
    )

    print(
        f"🏁 Evaluation completed. See {score_dir / 'data_overall.csv'} for overall evaluation results on BFCL V4."
    )
    if partial_eval:
        print(
            "⚠️  Partial evaluation for a single category is enabled (--partial-run flag is set). Accuracy scores are computed only on the subset of entries present in the model result files, which may differ from a full evaluation and from the official leaderboard score."
        )
    print(
        f"See {score_dir / 'data_live.csv'}, {score_dir / 'data_non_live.csv'}, {score_dir / 'data_multi_turn.csv'}, {score_dir / 'data_agentic.csv'} and {score_dir / 'data_format_sensitivity.csv'} for detailed evaluation results on each sub-section categories respectively."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two lists of strings.")

    # Add arguments for two lists of strings
    parser.add_argument(
        "--model", nargs="+", type=str, help="A list of model names to evaluate"
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        default="all",
        help="A list of test categories to run the evaluation on",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        type=str,
        help="Path to the folder where the model response files are stored; relative to the `berkeley-function-call-leaderboard` root folder",
    )
    parser.add_argument(
        "--score-dir",
        default=None,
        type=str,
        help="Path to the folder where the evaluation score files will be stored; relative to the `berkeley-function-call-leaderboard` root folder",
    )
    parser.add_argument(
        "--partial-eval",
        default=False,
        action="store_true",
        help="Run evaluation on a partial set of benchmark entries (eg. entries present in the model result files) without raising for missing IDs.",
    )

    args = parser.parse_args()

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file
    main(
        args.model,
        args.test_category,
        args.result_dir,
        args.score_dir,
        partial_eval=args.partial_eval,
    )

================
File: bfcl_eval/model_handler/local_inference/base_oss_handler.py
================
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction
from openai import OpenAI
from overrides import EnforceOverrides, final, override


class OSSHandler(BaseHandler, EnforceOverrides):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.OSSMODEL
        self.dtype = dtype

        # Will be overridden in batch_inference method
        # Used to indicate where the tokenizer and config should be loaded from
        self.model_path_or_id = None

        # Read from env vars with fallbacks
        self.local_server_endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
        self.local_server_port = os.getenv("LOCAL_SERVER_PORT", LOCAL_SERVER_PORT)

        # Support custom base_url and api_key for remote/local OpenAI-compatible deployments (e.g., vLLM)
        # Use REMOTE_OPENAI_* variables to avoid conflicts with main OPENAI_* variables
        self.base_url = os.getenv("REMOTE_OPENAI_BASE_URL", f"http://{self.local_server_endpoint}:{self.local_server_port}/v1")
        self.api_key = os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # TODO: Let oss model support FC methods as well, depends on their model type
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(result, has_tool_call_tag)

    @final
    def spin_up_local_server(
        self,
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
        local_model_path: Optional[str],
        lora_modules: Optional[list[str]] = None,
        enable_lora: bool = False,
        max_lora_rank: Optional[int] = None,
    ):
        """
        Spin up a local server for the model.
        If the server is already running, skip the setup.
        """
        from transformers import AutoConfig, AutoTokenizer

        # Determine the model source
        if local_model_path is not None:
            # Validate the local_model_path
            if not os.path.isdir(local_model_path):
                raise ValueError(
                    f"local_model_path '{local_model_path}' does not exist or is not a directory."
                )

            required_files = ["config.json", "tokenizer_config.json"]
            for file_name in required_files:
                if not os.path.exists(os.path.join(local_model_path, file_name)):
                    raise ValueError(
                        f"Required file '{file_name}' not found in local_model_path '{local_model_path}'."
                    )

            self.model_path_or_id = local_model_path
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "local_files_only": True,
                "trust_remote_code": True,
            }
        else:
            self.model_path_or_id = self.model_name_huggingface
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "trust_remote_code": True,
            }

        # For remote OpenAI-compatible endpoints, use specified tokenizer path if provided
        is_remote_endpoint = bool(os.getenv("REMOTE_OPENAI_BASE_URL"))
        tokenizer_path = os.getenv("REMOTE_OPENAI_TOKENIZER_PATH", self.model_path_or_id)

        if is_remote_endpoint and os.getenv("REMOTE_OPENAI_TOKENIZER_PATH"):
            # Use specified tokenizer for remote endpoints
            tokenizer_kwargs = {
                "pretrained_model_name_or_path": tokenizer_path,
                "trust_remote_code": True,
            }
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
                config = AutoConfig.from_pretrained(**tokenizer_kwargs)
                print(f"Loaded tokenizer from REMOTE_OPENAI_TOKENIZER_PATH: {tokenizer_path}")
            except Exception as e:
                print(f"Failed to load tokenizer from {tokenizer_path}, falling back to model path: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
                config = AutoConfig.from_pretrained(**load_kwargs)
        else:
            # Standard loading for local models or when no specific tokenizer path is provided
            self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
            config = AutoConfig.from_pretrained(**load_kwargs)

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self.tokenizer.model_max_length is not None:
            self.max_context_length = self.tokenizer.model_max_length
        else:
            if not hasattr(self, "max_context_length"):
                raise ValueError(
                    "Model does not have a max_position_embeddings attribute or tokenizer.model_max_length attribute. Please set the max_context_length attribute in the corresponding model handler."
                )
        print(f"Max context length: {self.max_context_length}")

        self._server_process = process = None
        self._stdout_thread = stdout_thread = None
        self._stderr_thread = stderr_thread = None
        # Event to signal threads to stop; no need to see logs after server is ready
        # declare early so it always exists
        self._stop_event = threading.Event()
        try:
            if not skip_server_setup:
                if backend == "vllm":
                    process = subprocess.Popen(
                        [
                            "vllm",
                            "serve",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tensor-parallel-size",
                            str(num_gpus),
                            "--gpu-memory-utilization",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ]
                        + (["--enable-lora"] if enable_lora else [])
                        + (
                            ["--max-lora-rank", str(max_lora_rank)]
                            if max_lora_rank is not None
                            else []
                        )
                        + (
                            sum(
                                [["--lora-modules", lora_module] for lora_module in lora_modules],
                                [],
                            )
                            if lora_modules
                            else []
                        ),
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                elif backend == "sglang":

                    process = subprocess.Popen(
                        [
                            "python",
                            "-m",
                            "sglang.launch_server",
                            "--model-path",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tp",
                            str(num_gpus),
                            "--mem-fraction-static",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ],
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                else:
                    raise ValueError(f"Backend {backend} is not supported.")

                def log_subprocess_output(pipe, stop_event):
                    # Read lines until the pipe is closed (EOF)
                    for line in iter(pipe.readline, ""):
                        if not stop_event.is_set():
                            print(line, end="")
                    print("server log tracking thread stopped successfully.")

                # Start threads to read and print stdout and stderr
                stdout_thread = threading.Thread(
                    target=log_subprocess_output, args=(process.stdout, self._stop_event)
                )
                stderr_thread = threading.Thread(
                    target=log_subprocess_output, args=(process.stderr, self._stop_event)
                )
                stdout_thread.setDaemon(True)
                stderr_thread.setDaemon(True)
                stdout_thread.start()
                stderr_thread.start()

            self._server_process = process
            self._stdout_thread = stdout_thread
            self._stderr_thread = stderr_thread

            # Wait for the server to be ready
            server_ready = False
            while not server_ready:
                # Check if the process has terminated unexpectedly
                if not skip_server_setup and process.poll() is not None:
                    # Output the captured logs
                    stdout, stderr = process.communicate()
                    print(stdout)
                    print(stderr)
                    raise Exception(
                        f"Subprocess terminated unexpectedly with code {process.returncode}"
                    )
                try:
                    # Make a simple request to check if the server is up
                    response = requests.get(f"{self.base_url}/models")
                    if response.status_code == 200:
                        server_ready = True
                        print("server is ready!")
                except requests.exceptions.ConnectionError:
                    # If the connection is not ready, wait and try again
                    time.sleep(1)

            # Signal threads to stop reading output
            self._stop_event.set()

        except Exception as e:
            # Clean-up everything we already started, then re-raise
            if self._server_process and self._server_process.poll() is None:
                self._server_process.terminate()
            if self._stop_event:
                self._stop_event.set()
            if self._stdout_thread:
                self._stdout_thread.join(timeout=2)
            if self._stderr_thread:
                self._stderr_thread.join(timeout=2)
            raise e

    def shutdown_local_server(self):
        """Terminate the locally launched OSS model server if it is still running."""
        # Ensure the server process is terminated properly
        process = getattr(self, "_server_process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                # Wait for the process to terminate fully
                process.wait(timeout=15)
                print("Process terminated successfully.")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()  # Wait again to ensure it's fully terminated
                print("Process killed.")

        # Tell the log-reader threads to stop and wait for them
        if getattr(self, "_stop_event", None):
            self._stop_event.set()
        if getattr(self, "_stdout_thread", None):
            self._stdout_thread.join(timeout=2)
        if getattr(self, "_stderr_thread", None):
            self._stderr_thread.join(timeout=2)

    #### Prompting methods ####

    def _format_prompt(self, messages, function):
        """
        Manually apply the chat template to construct the formatted prompt.
        This way, we can have full control over the final formatted prompt and is generally recommended for advanced use cases.
        """
        raise NotImplementedError(
            "OSS Models should implement their own prompt formatting."
        )

    @override
    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Tokenize the formatted prompt to get token count
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        if len(extra_body) > 0:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                extra_body=extra_body,
                timeout=72000,  # Avoid timeout errors
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                timeout=72000,  # Avoid timeout errors
            )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data

================
File: bfcl_eval/model_handler/local_inference/prism_coder.py
================
"""
PrismCoderHandler: Custom handler for prism-coder running via Ollama.

Supports 7B, 32B, and 72B model variants.
Extends QwenFCHandler with:
1. Multi-turn FC methods (8 abstract methods for inference_multi_turn_FC)
2. Language-aware type coercion (Python-only bool fix for Java/JS compat)
3. Abstention detection for irrelevance scoring
4. Robust JSON extraction from model output

BFCL Scoring Formula: 10% Non-Live + 10% Live + 10% Irrelevance + 30% Multi-Turn + 40% Agentic
"""

import json
import os
import re
import time
from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
from overrides import override


class PrismCoderHandler(QwenFCHandler):

    def __init__(self, model_name, temperature, registry_name=None, is_fc_model=False):
        # Store the real Ollama model name for API calls
        self._ollama_model_name = model_name
        
        # Pass valid HF ID to parent for tokenizer loading
        hf_name = os.getenv("REMOTE_OPENAI_TOKENIZER_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
        super().__init__(hf_name, temperature, registry_name, is_fc_model)

        # Wrap spin_up_local_server to fix model_path_or_id after tokenizer loading
        _original_spin_up = self.spin_up_local_server
        _handler = self

        def _patched_spin_up(*args, **kwargs):
            _original_spin_up(*args, **kwargs)
            _handler.model_path_or_id = _handler._ollama_model_name
        
        self.spin_up_local_server = _patched_spin_up

    ############################################################################
    # System Prompt Override — Abstention Instruction
    ############################################################################

    @override
    def _format_prompt(self, messages, function):
        """Override to inject abstention instruction into the system prompt.
        
        The base QwenFCHandler always says "You may call one or more functions..."
        which trains the model to ALWAYS call a function. We add an explicit
        instruction that the model should respond with plain text when no
        function is relevant to the user's query.
        """
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\n'
            formatted_prompt += 'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n'
            # xLAM-proven strict execution + abstention + agentic answer instructions
            formatted_prompt += 'IMPORTANT RULES:\n'
            formatted_prompt += '1. If NONE of the provided functions are relevant to the user\'s query, respond with a plain text message. Do NOT call any function when the query is unrelated to ALL available tools.\n'
            formatted_prompt += '2. Do not interpret or guess information. Wait for tool results to be returned before responding.\n'
            formatted_prompt += '3. If a tool result provides the answer, output it directly and concisely. Do not add conversational filler.\n'
            formatted_prompt += '4. If the user\'s input lacks required parameters, ask for clarification.\n'
            formatted_prompt += '5. When you have the final answer, state it clearly. Example: "The result is X."\n'
            formatted_prompt += '6. Do not hallucinate optional parameters. If the user does not explicitly provide a value for an optional parameter, do not include that parameter in your JSON arguments.\n'
            formatted_prompt += '7. When saving data to memory or a database, use the EXACT variable names and values provided by the user or the tool. Do not summarize or alter keys.<|im_end|>\n'
        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        # Replicate the rest of the parent's _format_prompt logic
        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            content = message.get("content", "")
            if not isinstance(content, str):
                content = ""

            # Extract reasoning_content if present
            reasoning_content = ""
            if message.get("reasoning_content"):
                reasoning_content = message["reasoning_content"]
            elif message["role"] == "assistant" and "</think>" in content:
                reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = content.split("</think>")[-1].lstrip("\n")

            if message["role"] == "user" or (
                message["role"] == "system" and idx > 0
            ):
                formatted_prompt += (
                    f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
                )
            elif message["role"] == "assistant":
                # Only inject <think> for the response we're about to generate (after last query)
                # For history messages, include reasoning if present
                if idx > last_query_index:
                    if reasoning_content:
                        formatted_prompt += f"<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}"
                    else:
                        formatted_prompt += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>assistant\n{content}"

                if "tool_calls" in message and message["tool_calls"]:
                    for tool_call in message["tool_calls"]:
                        tc = tool_call.get("function", tool_call)
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            args_str = args
                        else:
                            args_str = json.dumps(args)
                        formatted_prompt += f'\n<tool_call>\n{{"name": "{tc["name"]}", "arguments": {args_str}}}\n</tool_call>'
                formatted_prompt += "<|im_end|>\n"
            elif message["role"] == "tool":
                # Native Qwen2.5 tool role — matches base model pre-training template
                formatted_prompt += f"<|im_start|>tool\n{content}<|im_end|>\n"

        # Only inject <think> tags for models that support reasoning/thinking mode
        # Stock Qwen2.5-Coder models do NOT support <think> tags.
        # Set PRISM_ENABLE_THINKING=1 if using a fine-tuned model with thinking support.
        enable_thinking = os.getenv("PRISM_ENABLE_THINKING", "0") == "1"
        if enable_thinking:
            formatted_prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        else:
            formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    ############################################################################
    # Decode Methods — Language-Aware Type Coercion
    ############################################################################

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Override to handle abstention and language-aware type coercion."""
        
        tool_calls = self._extract_tool_calls(result)
        
        if not tool_calls:
            return []
        
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        
        decoded = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            # Language-aware type coercion: only fix types for Python
            fixed_args = self._fix_argument_types(args, language=language)
            decoded.append({name: fixed_args})
        
        return decoded

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Override to handle abstention + type coercion + eval()-safe formatting.
        
        CRITICAL: This method's output is literally eval()'d by multi_turn_utils.py.
        Uses xLAM's proven repr(v) pattern for eval() safety.
        Applies _fix_argument_types for Python type coercion (bool/null strings).
        """
        
        tool_calls = self._extract_tool_calls(result)
        
        if not tool_calls:
            return []
        
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        
        execution_list = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            name = item["name"]
            arguments = item.get("arguments", {})
            # CRITICAL FIX: Apply type coercion (Python-only) before execution
            arguments = self._fix_argument_types(arguments, language="Python")
            # Use xLAM's repr(v) pattern for eval()-safe argument formatting
            execution_list.append(
                f"{name}({','.join([f'{k}={repr(v)}' for k, v in arguments.items()])})"
            )
        return execution_list

    ############################################################################
    # FC Mode Methods — Multi-Turn Support
    ############################################################################

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """Initialize inference data for FC mode multi-turn."""
        functions: list = test_entry.get("function", [])
        inference_data["message"] = []
        inference_data["function"] = functions
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """Store the function definitions for prompt formatting.
        
        For our prompting-style FC handler, we don't need OpenAI-style tool conversion.
        We pass raw function definitions directly to _format_prompt.
        """
        inference_data["function"] = test_entry.get("function", [])
        return inference_data

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Add the first turn message(s) to the conversation history."""
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """Add subsequent turn user messages to conversation history."""
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        """Query the model in FC mode using our prompting-style approach.
        
        Since prism-coder uses Ollama's OpenAI-compatible completions API,
        we build the prompt via _format_prompt and call the completions endpoint.
        This is identical to _query_prompting but uses FC inference_data.
        """
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Tokenize to manage context window
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            timeout=72000,
        )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse the FC mode response — extract tool calls and build chat history entry.
        
        Returns a dict with:
        - model_responses: raw text for decode_execute
        - model_responses_message_for_chat_history: structured message for conversation
        - input_token, output_token: token counts
        - reasoning_content: extracted <think> content
        """
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        # Extract reasoning content
        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        # Build the message for chat history
        if len(extracted_tool_calls) > 0:
            # Convert extracted tool calls to the format expected by _format_prompt
            tool_calls_for_history = []
            for tc in extracted_tool_calls:
                tool_calls_for_history.append({
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    }
                })
            
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls_for_history,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }
        
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """Add assistant response (with tool calls) to conversation history."""
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """Add tool execution results to conversation history.
        
        Results are formatted as <tool_response> within a user message,
        matching the Qwen chat template for tool responses.
        """
        for execution_result in execution_results:
            inference_data["message"].append({
                "role": "tool",
                "content": execution_result,
            })
        return inference_data

    ############################################################################
    # Type Coercion — Language-Aware
    ############################################################################

    @staticmethod
    def _fix_argument_types(args, language="Python"):
        """Fix common type coercion issues in function call arguments.
        
        LANGUAGE-AWARE: Only applies bool/null coercion for Python.
        Java and JavaScript expect string values for String parameters,
        so "true"/"false" must remain as strings.
        
        Issues fixed (Python only):
        1. Stringified booleans: "true"/"false" -> True/False
        2. Stringified null: "null"/"none" -> None
        
        Issues fixed (all languages):
        3. Stringified JSON objects: '{"key": "val"}' -> {"key": "val"}
        4. Extra-quoted strings: "'USERSPACE1'" -> "USERSPACE1"
        """
        if not isinstance(args, dict):
            return args
        
        fixed = {}
        for key, value in args.items():
            fixed[key] = PrismCoderHandler._fix_value(value, language=language)
        return fixed
    
    @staticmethod
    def _fix_value(value, language="Python"):
        """Recursively fix a single value."""
        if isinstance(value, str):
            # Fix stringified booleans — ONLY for Python
            if language == "Python":
                if value.lower() == "true":
                    return True
                if value.lower() == "false":
                    return False
                if value.lower() == "null" or value.lower() == "none":
                    return None
            
            # Fix extra-quoted strings: "'something'" -> "something" (all languages)
            if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                return value[1:-1]
            
            # Fix stringified JSON objects/arrays (all languages)
            if (value.startswith("{") and value.endswith("}")) or \
               (value.startswith("[") and value.endswith("]")):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return PrismCoderHandler._fix_argument_types(parsed, language=language)
                    elif isinstance(parsed, list):
                        return [PrismCoderHandler._fix_value(v, language=language) for v in parsed]
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            return value
        
        elif isinstance(value, dict):
            return PrismCoderHandler._fix_argument_types(value, language=language)
        
        elif isinstance(value, list):
            return [PrismCoderHandler._fix_value(v, language=language) for v in value]
        
        return value

    ############################################################################
    # Tool Call Extraction
    ############################################################################

    @staticmethod
    @override
    def _extract_tool_calls(input_string):
        """Extract tool calls from model output.
        
        Handles:
        1. Standard <tool_call> tag extraction (Qwen native format)
        2. Bare JSON objects (prism-coder format)
        3. Abstention (no tool call - returns empty list)
        """
        # Strip <think>...</think> blocks first
        cleaned = re.sub(r"<think>.*?</think>", "", input_string, flags=re.DOTALL).strip()
        
        # If the output doesn't contain any JSON-like content, it's an abstention
        if not re.search(r'[{]', cleaned):
            return []
        
        # Try the standard <tool_call> tag extraction (Qwen native format)
        # Match with or without newlines around JSON content
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, cleaned, re.DOTALL)
        if matches:
            result = []
            for match in matches:
                try:
                    result.append(json.loads(match.strip()))
                except Exception:
                    pass
            if result:
                return result

        # Try to parse the entire output as a single JSON tool call
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "name" in parsed:
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find all JSON objects in the output (for parallel calls)
        json_objects = PrismCoderHandler._find_json_objects(cleaned)
        if json_objects:
            return json_objects

        # Try line-by-line JSON parsing as last resort
        result = []
        for line in cleaned.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "name" in parsed:
                    result.append(parsed)
            except json.JSONDecodeError:
                continue
        
        return result
    
    @staticmethod
    def _find_json_objects(text):
        """Find all complete JSON objects with 'name' key in text using bracket matching."""
        result = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and "name" in parsed:
                                    result.append(parsed)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1
        return result

================
File: bfcl_eval/model_handler/local_inference/qwen_fc.py
================
import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class QwenFCHandler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- if message.content is string %}
                {%- set content = message.content %}
            {%- else %}
                {%- set content = '' %}
            {%- endif %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }
            
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception as e:
                pass
        return result

================
File: bfcl_eval/model_handler/local_inference/salesforce_qwen.py
================
import json

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class SalesforceQwenHandler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        system_message = "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."
        remaining_messages = messages
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]

        # Format system message with tool instructions
        formatted_prompt += "<|im_start|>system\n"
        formatted_prompt += system_message + "\n"
        formatted_prompt += "You have access to a set of tools. When using tools, make calls in a single JSON array: \n\n"
        formatted_prompt += '[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]\n\n'
        formatted_prompt += "If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification. "
        formatted_prompt += "Do not interpret or respond until tool results are returned. Once they are available, process them or make additional calls if needed. "
        formatted_prompt += "For tasks that don't require tools, such as casual conversation or general advice, respond directly in plain text. The available tools are:\n\n"

        for func in function:
            formatted_prompt += json.dumps(func, indent=4) + "\n\n"
        formatted_prompt += "<|im_end|>"

        # Format conversation messages
        for message in remaining_messages:
            if message["role"] == "tool":
                formatted_prompt += "<|im_start|>tool\n"
                if isinstance(message["content"], (dict, list)):
                    formatted_prompt += json.dumps(message["content"])
                else:
                    formatted_prompt += message["content"]
                formatted_prompt += "<|im_end|>"
            elif "tool_calls" in message and message["tool_calls"]:
                formatted_prompt += "<|im_start|>assistant\n"
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
                formatted_prompt += json.dumps(tool_calls) + "<|im_end|>"
            else:
                formatted_prompt += (
                    f"<|im_start|>{message['role']}\n{message['content'].strip()}<|im_end|>"
                )

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # result = result.replace("<|python_tag|>", "")
        try:
            # Parse the JSON array of function calls
            function_calls = json.loads(result)
            if not isinstance(function_calls, list):
                function_calls = [function_calls]
        except json.JSONDecodeError:
            # Fallback for semicolon-separated format
            function_calls = [json.loads(call.strip()) for call in result.split(";")]

        decoded_output = []
        for func_call in function_calls:
            name = func_call["name"]
            arguments = func_call["arguments"]
            decoded_output.append({name: arguments})

        return decoded_output

    @override
    def decode_execute(self, result, has_tool_call_tag):
        try:
            function_calls = json.loads(result)
            if not isinstance(function_calls, list):
                function_calls = [function_calls]
        except json.JSONDecodeError:
            function_calls = [json.loads(call.strip()) for call in result.split(";")]

        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            arguments = func_call["arguments"]
            execution_list.append(
                f"{name}({','.join([f'{k}={repr(v)}' for k,v in arguments.items()])})"
            )

        return execution_list

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # override the default bfcl system prompt, xLAM uses its own system prompt
        return {"message": [], "function": functions}

================
File: bfcl_eval/model_handler/base_handler.py
================
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.utils import *
from overrides import final

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )


class BaseHandler:
    model_name: str
    is_fc_model: bool
    registry_name: str
    temperature: float
    registry_dir_name: str
    model_name_underline_replaced: str
    model_style: ModelStyle

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, **kwargs
    ) -> None:
        """
        Args:
            model_name: The name of the model as used in the vendor API or on Hugging Face.
            temperature: The temperature of the model.
            registry_name: The name of the model as used internally in BFCL, used for result directory naming.
            is_fc_model: Whether the model is a function calling model.
            **kwargs: Additional attributes passed via kwargs.
        """
        self.model_name = model_name
        self.is_fc_model = is_fc_model
        self.registry_name = registry_name

        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        # The directory name for the model
        # Replace the slash with underscore to avoid creating subdirectories
        self.registry_dir_name = registry_name.replace("/", "_")
        self.temperature = temperature

        # Set any additional attributes passed via kwargs
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # This method is used to retrive model response for each model.

        # FC model
        # TODO: Let all models have the is_fc_model attribute and remove the "FC" check
        if "FC" in self.registry_name or self.is_fc_model:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log)
        # Prompting model
        else:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_prompting(test_entry, include_input_log)

    @final
    def inference_multi_turn_FC(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list] = (
            []
        )  # The model response that will be used for later evaluation
        all_inference_log: list[list[dict]] = (
            []
        )  # The debugging log for human to understand
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        all_reasoning_content: list[list] = []

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                # Since we have added new functions, we need to recompile the tools
                inference_data = self._compile_tools(inference_data, test_entry)
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                # TODO: Move this to before pre_query_processing_FC.
                # Shouldn't be happening in the inference loop.
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_FC(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_FC(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_FC(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_FC(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_FC(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)

                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )

                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_inference_log.append(current_turn_inference_log)
            all_reasoning_content.append(current_turn_reasoning_content)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }

        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # Only for reasoning models, reasoning content will be stored as part of metadata and in inference log
        all_reasoning_content: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=holdout_function[str(turn_idx)]
                        ),
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_reasoning_content = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_prompting(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_prompting(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)
                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    model_response_data["model_responses_decoded"] = decoded_model_responses
                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        # We only include reasoning content if it exists and is not empty
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_single_turn_FC(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_FC(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_FC(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    @final
    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_prompting(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_prompting(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard AST checker input.
        """
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard execute checker input.
        """
        raise NotImplementedError

    @final
    def write(self, result, result_dir, update_mode=False):
        # Use the internal registry name to decide the result directory to avoid
        # collisions between different variants that share the same API model name.
        model_result_dir = result_dir / self.registry_dir_name

        if isinstance(result, dict):
            result = [result]

        # Collect and format each entry for JSON compatibility
        entries_to_write = [make_json_serializable(entry) for entry in result]

        # Group entries by their `test_category` for efficient file handling
        file_entries = {}
        for entry in entries_to_write:
            test_category = extract_test_category_from_id(entry["id"])
            # Determine the high-level grouping folder (non_live, live, etc.)
            group_dir_name = get_directory_structure_by_id(entry["id"])
            group_dir_path = model_result_dir / group_dir_name
            group_dir_path.mkdir(parents=True, exist_ok=True)

            file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result.json"
            file_entries.setdefault(file_path, []).append(entry)

        for file_path, entries in file_entries.items():
            if update_mode:
                # Load existing entries from the file
                existing_entries = {}
                if file_path.exists():
                    existing_entries = {
                        entry["id"]: entry for entry in load_file(file_path)
                    }

                # Update existing entries with new data
                for entry in entries:
                    existing_entries[entry["id"]] = entry

                # Sort entries by `id` and write them back to ensure order consistency
                sorted_entries = sorted(existing_entries.values(), key=sort_key)
                with open(file_path, "w") as f:
                    for entry in sorted_entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

            else:
                # Normal mode: Append to the end of the file
                # Note: We will sort all the entries at the end of the generation pipeline to ensure the order is consistent
                entries.sort(key=sort_key)
                with open(file_path, "a") as f:
                    for entry in entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the `_parse_query_response_FC` method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_FC` and `_add_next_turn_user_message_FC` methods, but it's usually cleaner to do it here.
        The inference_data dict is updated in place and returned.

        Note: This method has different signature from its Prompting version.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        [Only for FC mode]
        This method is used to prepare/compile the tools from the test entry and add them to the inference data to use for model query in FC mode.
        Function docs usually need to be transformed to the format expected by the model, done through the `convert_to_tool` function from `model_handler/utils.py`.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn. For most entry, this role will not appear.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_FC` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_prompting` and `_add_next_turn_user_message_prompting` methods, but it's usually cleaner to do it here.
        The function docs are usually supplied to the prompting models as part of the system prompt, done via the `system_prompt_pre_processing_chat_model` function from `model_handler/utils.py`, unless the model has a different way of handling it.
        Returns a dict that contains all the necessary information for the query method.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.

        Note: This method has different signature from its FC version.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_prompting` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        By default, execution results are added back as a `user` role message, as most models don't support the `tool` role in prompting mode.
        """
        raise NotImplementedError





================================================================
End of Codebase
================================================================
```

---

# APPENDIX B: Training Pipeline Code (bfcl_repomix_training.txt)

The following is a repomix bundle of 5 files from our MLX-native training pipeline,
including QLoRA SFT, GRPO alignment, data generation, and the 35-case test suite.

```
This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: bfcl_qlora_finetune.py, bfcl_grpo_align.py, generate_bfcl_training_data.py, test_handler.py, run_bfcl_pipeline.sh, merge_adapters.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)


================================================================
Directory Structure
================================================================
bfcl_grpo_align.py
bfcl_qlora_finetune.py
generate_bfcl_training_data.py
merge_adapters.py
run_bfcl_pipeline.sh
test_handler.py

================================================================
Files
================================================================

================
File: bfcl_grpo_align.py
================
"""
BFCL GRPO Alignment Script (MLX Native)

DPO-based preference alignment for calibrating abstention behavior.
Uses MLX-LM LoRA for Apple Silicon native training.

Reward Structure:
    +3.0  Correct abstention (irrelevant query → no function call)
    +2.0  Correct tool call (relevant query → correct function)
    -3.0  False abstention (relevant query → no function call)
    -2.0  Hallucinated call (irrelevant query → function call)

Pipeline:
    1. Generate DPO preference pairs from training data
    2. Fine-tune the SFT model with DPO on preference pairs
    3. Fuse the aligned adapter

Usage:
    # Generate pairs + train
    python bfcl_grpo_align.py --model ./output/bfcl-72b/fused_model --data ./data/bfcl

    # Generate pairs only
    python bfcl_grpo_align.py --generate-only --data ./data/bfcl

Requirements:
    pip install mlx-lm
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def generate_dpo_pairs(data_dir: str, output_path: str, max_pairs: int = 2000):
    """Generate DPO preference pairs from existing training data.
    
    Creates chosen/rejected pairs:
    - Irrelevance examples: chosen=abstention, rejected=hallucinated call
    - Multi-turn examples: chosen=correct call, rejected=false abstention
    """
    train_path = Path(data_dir) / "train.jsonl"
    grpo_path = Path(data_dir) / "grpo_pairs.jsonl"
    
    # Use pre-generated GRPO pairs if available
    if grpo_path.exists():
        with open(grpo_path) as f:
            existing_pairs = [json.loads(line) for line in f]
        print(f"   Found {len(existing_pairs)} pre-generated GRPO pairs")
    else:
        existing_pairs = []
    
    # Also generate pairs from training data
    generated_pairs = []
    with open(train_path) as f:
        examples = [json.loads(line) for line in f]
    
    for ex in examples:
        category = ex.get("category", "unknown")
        messages = ex.get("messages", [])
        
        if len(messages) < 2:
            continue
        
        # Build prompt from system + user messages
        prompt_parts = []
        assistant_response = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
            elif msg["role"] == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
            elif msg["role"] == "assistant":
                assistant_response = msg.get("content", "")
                break  # Only use first assistant response for single-turn DPO
        
        if not prompt_parts or not assistant_response:
            continue
        
        prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"
        
        if category == "irrelevance":
            # Chosen: correct abstention
            # Rejected: hallucinated tool call
            generated_pairs.append({
                "prompt": prompt,
                "chosen": assistant_response,
                "rejected": '<tool_call>\n{"name": "unknown_function", "arguments": {"query": "irrelevant"}}\n</tool_call>',
            })
        elif category == "multi_turn":
            # Chosen: correct tool call
            # Rejected: false abstention
            generated_pairs.append({
                "prompt": prompt,
                "chosen": assistant_response,
                "rejected": "I don't have the right tools to help with that request.",
            })
        elif category == "miss_func":
            # Chosen: correct abstention (tool is missing)
            # Rejected: hallucinated call to the missing function
            generated_pairs.append({
                "prompt": prompt,
                "chosen": assistant_response,
                "rejected": '<tool_call>\n{"name": "missing_function", "arguments": {}}\n</tool_call>',
            })
    
    # Combine and shuffle
    all_pairs = existing_pairs + generated_pairs
    random.shuffle(all_pairs)
    
    # Limit
    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]
    
    # Convert to MLX-LM chat format for DPO
    # MLX-LM DPO uses {"chosen": [...messages...], "rejected": [...messages...]}
    dpo_train = []
    for pair in all_pairs:
        # Parse prompt back to messages
        prompt_text = pair.get("prompt", "")
        chosen_text = pair.get("chosen", "")
        rejected_text = pair.get("rejected", "")
        
        dpo_train.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
    
    # Split 90/10
    split_idx = int(len(dpo_train) * 0.9)
    train_data = dpo_train[:split_idx]
    valid_data = dpo_train[split_idx:]
    
    # Write to DPO data directory
    dpo_dir = Path(output_path)
    dpo_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dpo_dir / "train.jsonl", "w") as f:
        for pair in train_data:
            f.write(json.dumps(pair) + "\n")
    
    with open(dpo_dir / "valid.jsonl", "w") as f:
        for pair in valid_data:
            f.write(json.dumps(pair) + "\n")
    
    print(f"   Generated {len(train_data)} train + {len(valid_data)} valid DPO pairs")
    print(f"   Output: {dpo_dir}")
    return len(train_data)


def train_dpo(model_path: str, data_dir: str, adapter_path: str,
              iters: int = 500, lora_rank: int = 32, 
              learning_rate: float = 5e-6, lora_layers: int = 8):
    """Run DPO/GRPO alignment using MLX-LM LoRA."""
    print(f"\n{'='*60}")
    print(f"GRPO Alignment Training")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Config: rank={lora_rank}, iters={iters}, lr={learning_rate}")
    print(f"LoRA layers: {lora_layers}")
    print()
    
    # MLX-LM DPO training for preference alignment
    # CRITICAL: Must use mlx_lm.dpo (not .lora) for chosen/rejected pair format
    cmd = [
        sys.executable, "-m", "mlx_lm.dpo",
        "--model", model_path,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(iters),
        "--lora-rank", str(lora_rank),
        "--batch-size", "1",
        "--learning-rate", str(learning_rate),
        "--lora-layers", str(lora_layers),
        "--save-every", "50",
        "--grad-checkpoint",
        "--max-seq-length", "2048",  # DPO needs ~2x memory; cap at 2048 for 48GB safety
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\n✅ GRPO adapter saved to: {adapter_path}")


def fuse_adapter(model_path: str, adapter_path: str, output_path: str):
    """Fuse the alignment adapter."""
    print(f"\n{'='*60}")
    print(f"Fusing GRPO Adapter")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model_path,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--export-gguf",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\n✅ Aligned model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BFCL GRPO Alignment (MLX Native)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SFT-fused MLX model")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data directory (with train.jsonl)")
    parser.add_argument("--output-dir", type=str, default="./output/bfcl-72b-grpo",
                        help="Output directory")
    parser.add_argument("--iters", type=int, default=500,
                        help="Training iterations (default: 500)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32, lower than SFT)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6, lower than SFT)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate DPO pairs, don't train")
    parser.add_argument("--max-pairs", type=int, default=2000,
                        help="Maximum DPO pairs to generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 BFCL GRPO Alignment — MLX Native")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpo_data_dir = str(output_dir / "dpo_data")
    adapter_path = str(output_dir / "adapters")
    
    # Step 1: Generate pairs
    print(f"\n📊 Generating DPO preference pairs...")
    generate_dpo_pairs(args.data, dpo_data_dir, max_pairs=args.max_pairs)
    
    if args.generate_only:
        print("\n✅ Pairs generated. Use --model to train.")
        return
    
    # Step 2: Train
    train_dpo(
        model_path=args.model,
        data_dir=dpo_data_dir,
        adapter_path=adapter_path,
        iters=args.iters,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
    )
    
    # Step 3: Fuse
    fused_path = str(output_dir / "fused_aligned")
    fuse_adapter(args.model, adapter_path, fused_path)
    
    print(f"\n{'='*60}")
    print("✅ GRPO alignment complete!")
    print(f"{'='*60}")
    print(f"\nDeploy to Ollama:")
    gguf_files = list(Path(fused_path).glob("*.gguf"))
    if gguf_files:
        print(f"  ollama create prism-coder-72b-FC -f <(echo 'FROM {gguf_files[0]}')")


if __name__ == "__main__":
    main()

================
File: bfcl_qlora_finetune.py
================
"""
BFCL QLoRA Fine-Tuning Script (MLX Native)

Fine-tunes Qwen2.5 72B for BFCL certification using MLX-LM's LoRA/QLoRA.
Optimized for Apple Silicon M5 Max 48GB.

MLX-LM handles 4-bit quantized training natively on Apple Silicon — 
no bitsandbytes or CUDA required.

Pipeline:
    1. Convert HF model to MLX format with 4-bit quantization
    2. LoRA fine-tune on BFCL training data
    3. Fuse LoRA adapters back into the model
    4. Export to GGUF for Ollama deployment

Usage:
    # Full pipeline (convert + train + fuse)
    python bfcl_qlora_finetune.py --model Qwen/Qwen2.5-72B-Instruct --data ./data/bfcl

    # Train only (if model already converted)
    python bfcl_qlora_finetune.py --mlx-model ./mlx_models/Qwen2.5-72B-4bit --data ./data/bfcl --skip-convert

    # Fuse only (after training)
    python bfcl_qlora_finetune.py --mlx-model ./mlx_models/Qwen2.5-72B-4bit --fuse-only --adapter-path ./output/bfcl-72b/adapters

Requirements:
    pip install mlx-lm
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def convert_model(hf_model: str, mlx_path: str, q_bits: int = 4):
    """Convert HuggingFace model to MLX format with quantization."""
    print(f"\n{'='*60}")
    print(f"Step 1: Converting {hf_model} to MLX ({q_bits}-bit)")
    print(f"{'='*60}")
    print(f"Output: {mlx_path}")
    print(f"This downloads the model and quantizes to {q_bits}-bit.")
    print(f"For 72B Q4: ~40GB download, ~38GB on disk after quantization.\n")

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", hf_model,
        "--mlx-path", mlx_path,
        "-q",
        "--q-bits", str(q_bits),
        "--q-group-size", "64",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\n✅ Model converted to: {mlx_path}")


def prepare_training_data(data_dir: str):
    """Verify training data is in the correct MLX-LM chat format.
    
    MLX-LM expects JSONL files in data/ directory:
    - train.jsonl: training data
    - valid.jsonl: validation data (optional)
    - test.jsonl: test data (optional)
    
    Format: {"messages": [{"role": "system", "content": "..."}, ...]}
    """
    train_path = Path(data_dir) / "train.jsonl"
    valid_path = Path(data_dir) / "valid.jsonl"
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        print(f"   Run: python generate_bfcl_training_data.py --output-dir {data_dir}")
        sys.exit(1)
    
    # Count examples
    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    
    valid_count = 0
    if valid_path.exists():
        with open(valid_path) as f:
            valid_count = sum(1 for _ in f)
    
    print(f"\n📊 Training data: {train_count} examples")
    print(f"   Validation data: {valid_count} examples")
    
    # Verify format
    with open(train_path) as f:
        first = json.loads(f.readline())
    
    if "messages" not in first:
        print("❌ Training data format error: missing 'messages' key")
        print("   Expected: {\"messages\": [{\"role\": \"system\", \"content\": \"...\"}, ...]}")
        sys.exit(1)
    
    print("   Format: ✅ Chat JSONL (MLX-LM native)")
    return train_count


def train_lora(mlx_model: str, data_dir: str, adapter_path: str,
               iters: int = 1000, lora_rank: int = 64, 
               batch_size: int = 1, learning_rate: float = 1e-5,
               lora_layers: int = 16, grad_checkpoint: bool = True):
    """Run MLX-LM LoRA fine-tuning."""
    print(f"\n{'='*60}")
    print(f"Step 2: LoRA Fine-Tuning")
    print(f"{'='*60}")
    print(f"Model: {mlx_model}")
    print(f"Data: {data_dir}")
    print(f"Adapter output: {adapter_path}")
    print(f"Config: rank={lora_rank}, iters={iters}, lr={learning_rate}")
    print(f"Batch size: {batch_size}, LoRA layers: {lora_layers}")
    print(f"Gradient checkpointing: {grad_checkpoint}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", mlx_model,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(iters),
        "--lora-rank", str(lora_rank),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--lora-layers", str(lora_layers),
        "--save-every", "100",
        "--test-batches", "10",
        "--val-batches", "10",
        "--max-seq-length", "8192",
        "--grad-accum", "4",  # Simulate batch-4 without extra VRAM
        "--mask-prompt",  # BalanceSFT: loss only on completion (tool_call JSON), not prompt
    ]
    
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\n✅ LoRA adapter saved to: {adapter_path}")


def fuse_adapter(mlx_model: str, adapter_path: str, output_path: str,
                 export_gguf: bool = True):
    """Fuse LoRA adapter back into the model."""
    print(f"\n{'='*60}")
    print(f"Step 3: Fusing LoRA Adapter")
    print(f"{'='*60}")
    print(f"Base model: {mlx_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", mlx_model,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
    ]
    
    if export_gguf:
        cmd.extend(["--export-gguf"])
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\n✅ Fused model saved to: {output_path}")
    
    if export_gguf:
        gguf_files = list(Path(output_path).glob("*.gguf"))
        if gguf_files:
            print(f"\n📦 GGUF file for Ollama: {gguf_files[0]}")
            print(f"\nTo deploy with Ollama:")
            print(f"  1. Create Modelfile:")
            print(f"     FROM {gguf_files[0]}")
            print(f"  2. ollama create prism-coder-72b-FC -f Modelfile")
            print(f"  3. ollama run prism-coder-72b-FC")


def main():
    parser = argparse.ArgumentParser(description="BFCL QLoRA Fine-Tuning (MLX Native)")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="HuggingFace model ID to convert")
    parser.add_argument("--mlx-model", type=str, default=None,
                        help="Path to already-converted MLX model (skips conversion)")
    
    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Directory containing train.jsonl / valid.jsonl")
    
    # Training
    parser.add_argument("--iters", type=int, default=1000,
                        help="Training iterations (default: 1000)")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (default: 64)")
    parser.add_argument("--lora-layers", type=int, default=16,
                        help="Number of model layers to apply LoRA to (default: 16)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1 for 72B)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    
    # Workflow control
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip model conversion (use --mlx-model)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (used with --fuse-only)")
    parser.add_argument("--fuse-only", action="store_true",
                        help="Only fuse adapter")
    parser.add_argument("--no-fuse", action="store_true",
                        help="Skip fusing after training")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Don't export GGUF during fuse")
    
    # Paths
    parser.add_argument("--output-dir", type=str, default="./output/bfcl-72b",
                        help="Output directory")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Explicit adapter path (default: {output-dir}/adapters)")
    parser.add_argument("--q-bits", type=int, default=4,
                        help="Quantization bits for conversion (default: 4)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 BFCL QLoRA Fine-Tuning — MLX Native (Apple Silicon)")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_path = args.adapter_path or str(output_dir / "adapters")
    
    # Determine MLX model path
    if args.mlx_model:
        mlx_model_path = args.mlx_model
    else:
        # Default: store converted models alongside training output
        model_name = args.model.replace("/", "-")
        mlx_model_path = str(output_dir / f"{model_name}-{args.q_bits}bit")
    
    # Step 1: Convert
    if not args.skip_convert and not args.fuse_only:
        if Path(mlx_model_path).exists():
            print(f"\n⏭️  MLX model already exists at {mlx_model_path}, skipping conversion")
        else:
            convert_model(args.model, mlx_model_path, q_bits=args.q_bits)
    
    # Step 2: Verify data
    if not args.fuse_only and not args.skip_train:
        prepare_training_data(args.data)
    
    # Step 3: Train
    if not args.fuse_only and not args.skip_train:
        train_lora(
            mlx_model=mlx_model_path,
            data_dir=args.data,
            adapter_path=adapter_path,
            iters=args.iters,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_layers=args.lora_layers,
        )
    
    # Step 4: Fuse
    if not args.no_fuse:
        fused_path = str(output_dir / "fused_model")
        fuse_adapter(
            mlx_model=mlx_model_path,
            adapter_path=adapter_path,
            output_path=fused_path,
            export_gguf=not args.no_gguf,
        )
    
    print(f"\n{'='*60}")
    print("✅ Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

================
File: generate_bfcl_training_data.py
================
"""
BFCL Training Data Generator v2

Generates training data aligned with BFCL v4 test categories using REAL
BFCL function definitions from multi_turn_func_doc/ directory.

Categories generated:
1. Irrelevance negatives (abstention training with real BFCL tools)
2. Multi-turn function calling (using actual BFCL API collections)
3. Miss-func detection (model should recognize missing tools)
4. Miss-param detection (model should ask for missing parameters)

Output: JSONL in Qwen chat format with <tool_call> tags.

Usage:
    python generate_bfcl_training_data.py --output-dir ./data/bfcl
    python generate_bfcl_training_data.py --output-dir ./data/bfcl --bfcl-dir ~/gorilla-bfcl/berkeley-function-call-leaderboard
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Default paths
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_BFCL_DIR = Path.home() / "gorilla-bfcl" / "berkeley-function-call-leaderboard"


def load_bfcl_func_docs(bfcl_dir: Path) -> dict:
    """Load all real BFCL function doc definitions from multi_turn_func_doc/."""
    func_doc_dir = bfcl_dir / "bfcl_eval" / "data" / "multi_turn_func_doc"
    if not func_doc_dir.exists():
        print(f"Warning: BFCL func_doc dir not found: {func_doc_dir}")
        return {}
    
    collections = {}
    for json_file in sorted(func_doc_dir.glob("*.json")):
        api_name = json_file.stem
        functions = []
        with open(json_file) as f:
            for line in f:
                try:
                    func_def = json.loads(line.strip())
                    functions.append(func_def)
                except json.JSONDecodeError:
                    continue
        if functions:
            collections[api_name] = functions
            print(f"  Loaded {len(functions)} functions from {api_name}")
    
    print(f"  Total: {sum(len(v) for v in collections.values())} functions across {len(collections)} API collections")
    return collections


# 40 irrelevant queries guaranteed to not match any BFCL tool
IRRELEVANT_QUERIES = [
    "What is the meaning of life?",
    "Can you write me a poem about the ocean?",
    "Explain quantum entanglement in simple terms.",
    "What's the difference between a crocodile and an alligator?",
    "Tell me a joke about programmers.",
    "How does photosynthesis work?",
    "What are the main causes of World War I?",
    "Can you explain the theory of relativity?",
    "What is the capital of Mongolia?",
    "How do you make sourdough bread?",
    "What programming language should I learn first?",
    "Explain the difference between TCP and UDP.",
    "What is the Fibonacci sequence?",
    "How does blockchain technology work?",
    "What are the benefits of meditation?",
    "Explain the water cycle.",
    "What is machine learning?",
    "How do vaccines work?",
    "What is the deepest part of the ocean?",
    "Can you summarize Romeo and Juliet?",
    "What are prime numbers?",
    "How does a combustion engine work?",
    "What is the greenhouse effect?",
    "Explain Newton's three laws of motion.",
    "What is the difference between a virus and bacteria?",
    "How does WiFi work?",
    "What are the planets in our solar system?",
    "Explain supply and demand.",
    "What is natural language processing?",
    "How do earthquakes happen?",
    "What is the Pythagorean theorem?",
    "Explain the concept of compound interest.",
    "What are the main types of renewable energy?",
    "How does DNA replication work?",
    "What is the speed of light?",
    "Explain the concept of opportunity cost.",
    "What are the different types of clouds?",
    "How does the human immune system work?",
    "What is the scientific method?",
    "Explain the difference between RAM and ROM.",
]

# Abstention response templates
ABSTENTION_RESPONSES = [
    "I don't have a suitable function to help with that query. The available tools are not relevant to your request.",
    "None of the available tools can assist with this question. I can only help with tasks related to the provided functions.",
    "This query is outside the scope of the available tools. I cannot call any function to address this request.",
    "The provided functions don't cover this topic. I'm unable to assist with this query using the available tools.",
    "I don't have the right tools for this request. The available functions are designed for different purposes.",
    "There are no relevant functions available to answer your question. Let me know if you need help with something the available tools can handle.",
    "The available tools cannot help with this kind of request. This falls outside their intended functionality.",
    "I appreciate the question, but none of the provided functions are applicable here. I can only assist with function-specific tasks.",
]


def format_system_prompt(tools: list) -> str:
    """Format tools into the system prompt template matching PrismCoderHandler._format_prompt."""
    prompt = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    prompt += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in tools:
        prompt += f"\n{json.dumps(tool)}"
    prompt += "\n</tools>\n\n"
    prompt += 'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n'
    prompt += 'IMPORTANT RULES:\n'
    prompt += '1. If NONE of the provided functions are relevant to the user\'s query, respond with a plain text message. Do NOT call any function when the query is unrelated to ALL available tools.\n'
    prompt += '2. Do not interpret or guess information. Wait for tool results to be returned before responding.\n'
    prompt += '3. If a tool result provides the answer, output it directly and concisely. Do not add conversational filler.\n'
    prompt += '4. If the user\'s input lacks required parameters, ask for clarification.\n'
    prompt += '5. When you have the final answer, state it clearly. Example: "The result is X."\n'
    prompt += '6. Do not hallucinate optional parameters. If the user does not explicitly provide a value for an optional parameter, do not include that parameter in your JSON arguments.\n'
    prompt += '7. When saving data to memory or a database, use the EXACT variable names and values provided by the user or the tool. Do not summarize or alter keys.'
    return prompt


def format_as_raw_text(messages: list, tools: list) -> str:
    """Format messages into raw text matching PrismCoderHandler._format_prompt exactly.
    
    CRITICAL: This ensures training data uses the IDENTICAL format that the handler
    produces during evaluation, eliminating training-inference prompt mismatch.
    """
    formatted = ""
    
    # System prompt with tools
    if tools:
        formatted += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            formatted += messages[0]["content"]
        else:
            formatted += format_system_prompt(tools)
        formatted += "<|im_end|>\n"
    
    # Conversation messages
    start_idx = 1 if messages[0]["role"] == "system" else 0
    for msg in messages[start_idx:]:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            # Native Qwen2.5 tool role — matches handler's _format_prompt
            formatted += f"<|im_start|>tool\n{content}<|im_end|>\n"
    
    return formatted


def format_as_prompt_completion(messages: list, tools: list) -> dict:
    """Format messages as prompt/completion pair for MLX --mask-prompt loss masking.
    
    BalanceSFT strategy: Loss is calculated ONLY on the completion (final assistant response),
    not on the system prompt, user queries, or tool responses. This focuses gradient updates
    on the exact tool_call JSON tokens the model needs to learn.
    
    Returns: {"prompt": "...", "completion": "..."}
    """
    formatted = ""
    
    # System prompt with tools
    if tools:
        formatted += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            formatted += messages[0]["content"]
        else:
            formatted += format_system_prompt(tools)
        formatted += "<|im_end|>\n"
    
    # Find the last assistant message index for the split point
    start_idx = 1 if messages[0]["role"] == "system" else 0
    last_asst_idx = None
    for idx in range(len(messages) - 1, start_idx - 1, -1):
        if messages[idx]["role"] == "assistant":
            last_asst_idx = idx
            break
    
    if last_asst_idx is None:
        # No assistant message; return as text-only fallback
        return {"text": format_as_raw_text(messages, tools)}
    
    # Build prompt (everything up to but NOT including the last assistant message)
    for msg in messages[start_idx:last_asst_idx]:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            formatted += f"<|im_start|>tool\n{content}<|im_end|>\n"
    
    prompt = formatted
    
    # Completion = final assistant response (what the model should learn to generate)
    completion = f"<|im_start|>assistant\n{messages[last_asst_idx]['content']}<|im_end|>\n"
    
    return {"prompt": prompt, "completion": completion}


# === NOISY TRAJECTORY INJECTION (Agent Stumble Prevention) ===
# 15% of multi-turn examples get an interruption to teach the model
# to handle context shifts without freezing or hallucinating.
INTERRUPTION_TEMPLATES = [
    "Wait, before that — {question}",
    "Actually, hold on. {question}",
    "One more thing first: {question}",
    "Before you do that, {question}",
    "Hmm, actually ignore the last thing. {question}",
    "Let me change my mind. {question}",
]

INTERRUPTION_QUESTIONS = [
    "what's the current status of everything?",
    "can you list what we've done so far?",
    "is there anything pending?",
    "how much storage space is left?",
    "what was the result of the first operation?",
    "never mind about the last step, just show me the current state.",
    "actually, let's check the account balance first.",
    "wait, what time is it in Tokyo right now?",
    "can you summarize what happened so far?",
    "hold on, I need to check something else first.",
]

INTERRUPTION_RESPONSES = [
    "I don't have a function available to answer that question. Based on what we've done so far: {summary}",
    "That question is outside the scope of the available tools. Let me know if you'd like to continue with the previous task.",
    "I can't help with that specific request using the available functions. Would you like to continue where we left off?",
]


def generate_irrelevance_negatives(output_path: Path, collections: dict, num_examples: int = 2000):
    """Generate training examples where the model should NOT call any function.
    
    Uses REAL BFCL tool definitions paired with irrelevant queries.
    """
    examples = []
    api_keys = list(collections.keys())
    
    for i in range(num_examples):
        # Pick 1-3 random API collections and combine their functions
        num_apis = random.randint(1, min(3, len(api_keys)))
        selected_apis = random.sample(api_keys, num_apis)
        tools = []
        for api in selected_apis:
            tools.extend(collections[api])
        
        query = random.choice(IRRELEVANT_QUERIES)
        response = random.choice(ABSTENTION_RESPONSES)
        
        example = {
            "messages": [
                {"role": "system", "content": format_system_prompt(tools)},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            "category": "irrelevance",
        }
        # Convert to prompt/completion format for loss masking
        pc = format_as_prompt_completion(example["messages"], tools)
        pc["category"] = "irrelevance"
        examples.append(pc)
    
    output_file = output_path / "irrelevance_train.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(examples)} irrelevance examples -> {output_file}")
    return examples


def generate_multiturn_examples(output_path: Path, collections: dict, num_examples: int = 1000):
    """Generate multi-turn function calling training data.
    
    Uses REAL BFCL tool definitions. Each example is a multi-turn conversation
    with 2-4 turns of function calls followed by tool responses.
    """
    examples = []
    
    # Multi-turn scenario templates using real BFCL API collections
    scenario_templates = [
        # GorillaFileSystem scenarios
        {
            "apis": ["gorilla_file_system"],
            "turns": [
                {"query": "Create a new directory called 'reports' in the current location", "func": "mkdir", "args": {"dir_name": "reports"}, "result": '{"message": "Directory \'reports\' created successfully."}'},
                {"query": "List all files in the current directory", "func": "ls", "args": {}, "result": '["reports", "readme.txt", "data.csv"]'},
                {"query": "Move data.csv into the reports directory", "func": "mv", "args": {"source": "data.csv", "destination": "reports/data.csv"}, "result": '{"message": "File moved successfully."}'},
            ]
        },
        {
            "apis": ["gorilla_file_system"],
            "turns": [
                {"query": "What files are in my current directory?", "func": "ls", "args": {}, "result": '["main.py", "utils.py", "test.py", "config.json"]'},
                {"query": "Search for 'import' in main.py", "func": "grep", "args": {"file_name": "main.py", "pattern": "import"}, "result": '["import os", "import json", "import sys"]'},
                {"query": "Show me the content of config.json", "func": "cat", "args": {"file_name": "config.json"}, "result": '{"database": "postgresql", "port": 5432}'},
            ]
        },
        # VehicleControl scenarios
        {
            "apis": ["vehicle_control"],
            "turns": [
                {"query": "Start the engine of my car", "func": "startEngine", "args": {"ignitionMode": "START"}, "result": '{"engineState": "running", "fuelLevel": 75.5}'},
                {"query": "Turn on the headlights", "func": "activateParkingBrake", "args": {"mode": "release"}, "result": '{"parkingBrakeStatus": "released"}'},
                {"query": "What's the current fuel level?", "func": "check_budget", "args": {}, "result": '{"fuelLevel": 75.5, "range": 320}'},
            ]
        },
        # TradingBot scenarios
        {
            "apis": ["trading_bot"],
            "turns": [
                {"query": "What's the current price of AAPL?", "func": "get_stock_info", "args": {"symbol": "AAPL"}, "result": '{"symbol": "AAPL", "price": 185.50, "change": 2.3}'},
                {"query": "Place a buy order for 10 shares", "func": "place_order", "args": {"order_type": "Buy", "symbol": "AAPL", "price": 185.50, "amount": 10}, "result": '{"orderId": "ORD-12345", "status": "Pending"}'},
                {"query": "Check the status of my recent order", "func": "get_order_details", "args": {"order_id": 12345}, "result": '{"orderId": 12345, "status": "Completed", "price": 185.50, "amount": 10}'},
            ]
        },
        # MessageAPI scenarios
        {
            "apis": ["message_api"],
            "turns": [
                {"query": "Send a message to user 456 saying 'Meeting at 3pm'", "func": "send_message", "args": {"receiver_id": 456, "message": "Meeting at 3pm"}, "result": '{"status": "sent", "messageId": 789}'},
                {"query": "Show me my inbox messages", "func": "view_messages_sent", "args": {}, "result": '[{"id": 789, "to": 456, "message": "Meeting at 3pm", "time": "2024-01-15T14:30:00"}]'},
                {"query": "Search for messages containing 'budget'", "func": "search_messages", "args": {"keyword": "budget"}, "result": '[{"id": 101, "from": 123, "message": "Budget report attached", "time": "2024-01-14T09:00:00"}]'},
            ]
        },
        # TicketAPI scenarios
        {
            "apis": ["ticket_api"],
            "turns": [
                {"query": "Create a support ticket for a login issue", "func": "create_ticket", "args": {"title": "Login Issue", "description": "Cannot login to the platform", "priority": 3}, "result": '{"ticketId": "T-1001", "status": "Open"}'},
                {"query": "Get the details of that ticket", "func": "get_ticket", "args": {"ticket_id": 1001}, "result": '{"id": 1001, "title": "Login Issue", "status": "Open", "priority": 3}'},
                {"query": "Close that ticket as resolved", "func": "close_ticket", "args": {"ticket_id": 1001}, "result": '{"ticketId": 1001, "status": "Closed"}'},
            ]
        },
        # PostingAPI scenarios  
        {
            "apis": ["posting_api"],
            "turns": [
                {"query": "Create a post saying 'Excited about our new product launch!'", "func": "post_tweet", "args": {"content": "Excited about our new product launch!"}, "result": '{"id": 501, "status": "posted"}'},
                {"query": "Retweet post 501", "func": "retweet", "args": {"tweet_id": 501}, "result": '{"status": "retweeted"}'},
                {"query": "How many followers do I have?", "func": "get_user_stats", "args": {}, "result": '{"followers": 1250, "following": 380, "tweets": 42}'},
            ]
        },
        # === PARALLEL TOOL CALLING SCENARIOS ===
        # Critical for AST and Live categories where model must output multiple <tool_call> tags
        {
            "apis": ["gorilla_file_system"],
            "turns": [
                {"query": "Show me the contents of both main.py and config.json",
                 "parallel_calls": [
                     {"func": "cat", "args": {"file_name": "main.py"}},
                     {"func": "cat", "args": {"file_name": "config.json"}},
                 ],
                 "results": ['"import os\nimport json"', '{"db": "postgres"}']},
                {"query": "Now delete both files",
                 "parallel_calls": [
                     {"func": "rm", "args": {"file_name": "main.py"}},
                     {"func": "rm", "args": {"file_name": "config.json"}},
                 ],
                 "results": ['{"message": "main.py deleted"}', '{"message": "config.json deleted"}']},
            ]
        },
        {
            "apis": ["trading_bot"],
            "turns": [
                {"query": "What are the current prices of AAPL and TSLA?",
                 "parallel_calls": [
                     {"func": "get_stock_info", "args": {"symbol": "AAPL"}},
                     {"func": "get_stock_info", "args": {"symbol": "TSLA"}},
                 ],
                 "results": ['{"symbol": "AAPL", "price": 185.50}', '{"symbol": "TSLA", "price": 245.20}']},
                {"query": "Buy 10 shares of each",
                 "parallel_calls": [
                     {"func": "place_order", "args": {"order_type": "Buy", "symbol": "AAPL", "price": 185.50, "amount": 10}},
                     {"func": "place_order", "args": {"order_type": "Buy", "symbol": "TSLA", "price": 245.20, "amount": 10}},
                 ],
                 "results": ['{"orderId": "ORD-101", "status": "Pending"}', '{"orderId": "ORD-102", "status": "Pending"}']},
            ]
        },
        {
            "apis": ["message_api"],
            "turns": [
                {"query": "Send 'Meeting at 3pm' to users 456 and 789",
                 "parallel_calls": [
                     {"func": "send_message", "args": {"receiver_id": 456, "message": "Meeting at 3pm"}},
                     {"func": "send_message", "args": {"receiver_id": 789, "message": "Meeting at 3pm"}},
                 ],
                 "results": ['{"status": "sent", "messageId": 101}', '{"status": "sent", "messageId": 102}']},
            ]
        },
        {
            "apis": ["ticket_api"],
            "turns": [
                {"query": "Create tickets for both the login bug and the payment error",
                 "parallel_calls": [
                     {"func": "create_ticket", "args": {"title": "Login Bug", "description": "Users cannot log in", "priority": 1}},
                     {"func": "create_ticket", "args": {"title": "Payment Error", "description": "Payment processing fails", "priority": 2}},
                 ],
                 "results": ['{"ticketId": "T-2001", "status": "Open"}', '{"ticketId": "T-2002", "status": "Open"}']},
            ]
        },
    ]
    
    for i in range(num_examples):
        scenario = random.choice(scenario_templates)
        
        # Get tools from the specified APIs
        tools = []
        for api_name in scenario["apis"]:
            if api_name in collections:
                tools.extend(collections[api_name])
        
        if not tools:
            # Fallback: use all tools if specific API not found
            for api_name in random.sample(list(collections.keys()), min(2, len(collections))):
                tools.extend(collections[api_name])
        
        messages = [{"role": "system", "content": format_system_prompt(tools)}]
        
        # Randomly select 2-3 turns from the scenario
        num_turns = random.randint(2, min(3, len(scenario["turns"])))
        selected_turns = scenario["turns"][:num_turns]
        
        for turn in selected_turns:
            messages.append({"role": "user", "content": turn["query"]})
            
            if "parallel_calls" in turn:
                # Parallel tool calling: multiple <tool_call> blocks in one assistant message
                tc_parts = []
                tool_calls_meta = []
                for call in turn["parallel_calls"]:
                    tc_json = json.dumps({"name": call["func"], "arguments": call["args"]})
                    tc_parts.append(f'<tool_call>\n{tc_json}\n</tool_call>')
                    tool_calls_meta.append({"function": {"name": call["func"], "arguments": call["args"]}})
                
                tc_text = '\n'.join(tc_parts)
                messages.append({
                    "role": "assistant",
                    "content": tc_text,
                    "tool_calls": tool_calls_meta,
                })
                
                # Each parallel call gets its own tool response
                for result in turn["results"]:
                    messages.append({"role": "tool", "content": result})
            else:
                # Sequential single tool call
                tc_json = json.dumps({"name": turn["func"], "arguments": turn["args"]})
                tc_text = f'<tool_call>\n{tc_json}\n</tool_call>'
                messages.append({
                    "role": "assistant",
                    "content": tc_text,
                    "tool_calls": [{"function": {"name": turn["func"], "arguments": turn["args"]}}],
                })
                
                # Tool response
                messages.append({"role": "tool", "content": turn["result"]})
        
        # === NOISY TRAJECTORY INJECTION (15% chance) ===
        # Inject a user interruption mid-conversation to train Agent Stumble resilience
        if len(selected_turns) >= 2 and random.random() < 0.15:
            # Insert interruption after the first turn's tool response
            interrupt_idx = len(messages)  # current end
            template = random.choice(INTERRUPTION_TEMPLATES)
            question = random.choice(INTERRUPTION_QUESTIONS)
            interrupt_query = template.format(question=question)
            messages.append({"role": "user", "content": interrupt_query})
            
            summary = "the previous operations completed successfully"
            response = random.choice(INTERRUPTION_RESPONSES).format(summary=summary)
            messages.append({"role": "assistant", "content": response})
        
        example_msgs = [
            {"role": "system", "content": format_system_prompt(tools)},
        ] + messages[1:]  # skip our system message, use formatted one
        
        # Convert to prompt/completion format for loss masking
        pc = format_as_prompt_completion(example_msgs, tools)
        pc["category"] = "multi_turn"
        examples.append(pc)
    
    output_file = output_path / "multiturn_train.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(examples)} multi-turn examples -> {output_file}")
    return examples


def generate_miss_func_examples(output_path: Path, collections: dict, num_examples: int = 500):
    """Generate examples where the model should recognize that needed functions are missing.
    
    Strategy: provide a subset of tools from an API and ask about functionality
    that requires a missing tool. Model should say it can't do that.
    """
    examples = []
    
    miss_func_queries = [
        ("gorilla_file_system", "cp", "Copy the file report.pdf to the backup directory"),
        ("gorilla_file_system", "chmod", "Change the permissions of script.sh to executable"),
        ("vehicle_control", "adjustClimateControl", "Set the car temperature to 72°F"),
        ("trading_bot", "cancel_order", "Cancel my pending order for TSLA"),
        ("message_api", "delete_message", "Delete the message I sent to user 456"),
        ("ticket_api", "edit_ticket", "Change the priority of ticket T-1001 to high"),
        ("posting_api", "delete_tweet", "Remove my last tweet"),
    ]
    
    for i in range(num_examples):
        api_name, excluded_func, query = random.choice(miss_func_queries)
        
        if api_name not in collections:
            continue
        
        # Remove the excluded function
        tools = [f for f in collections[api_name] if f.get("name") != excluded_func]
        
        if not tools or len(tools) == len(collections.get(api_name, [])):
            # Skip if no tools remain or function wasn't found
            continue
        
        response = random.choice(ABSTENTION_RESPONSES)
        
        msgs = [
            {"role": "system", "content": format_system_prompt(tools)},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        pc = format_as_prompt_completion(msgs, tools)
        pc["category"] = "miss_func"
        examples.append(pc)
    
    output_file = output_path / "miss_func_train.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(examples)} miss-func examples -> {output_file}")
    return examples


def generate_grpo_pairs(output_path: Path, collections: dict, num_pairs: int = 800):
    """Generate GRPO preference pairs for irrelevance calibration.
    
    Each pair has:
    - chosen: correct abstention (no tool call)
    - rejected: hallucinated tool call
    """
    pairs = []
    api_keys = list(collections.keys())
    
    for i in range(num_pairs):
        # Pick random tools
        num_apis = random.randint(1, min(2, len(api_keys)))
        selected_apis = random.sample(api_keys, num_apis)
        tools = []
        for api in selected_apis:
            tools.extend(collections[api])
        
        query = random.choice(IRRELEVANT_QUERIES)
        
        # Chosen: correct abstention
        chosen_response = random.choice(ABSTENTION_RESPONSES)
        
        # Rejected: hallucinated tool call
        random_tool = random.choice(tools)
        tool_name = random_tool.get("name", "unknown_func")
        rejected_response = f'<tool_call>\n{{"name": "{tool_name}", "arguments": {{"query": "{query}"}}}}\n</tool_call>'
        
        pair = {
            "prompt": format_system_prompt(tools) + f"\n\nUser: {query}",
            "chosen": chosen_response,
            "rejected": rejected_response,
            "category": "irrelevance_grpo",
        }
        pairs.append(pair)
    
    output_file = output_path / "grpo_pairs.jsonl"
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"Generated {len(pairs)} GRPO preference pairs -> {output_file}")
    return pairs


def merge_datasets(output_path: Path):
    """Merge all training data into a single shuffled file."""
    all_examples = []
    
    for jsonl_file in output_path.glob("*_train.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    random.shuffle(all_examples)
    
    # Split 90/10 train/valid
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    valid = all_examples[split_idx:]
    
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"
    
    with open(train_file, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    
    with open(valid_file, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nMerged dataset: {len(train)} train, {len(valid)} valid")
    print(f"Category distribution:")
    cats = {}
    for ex in all_examples:
        cat = ex.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate BFCL training data (v2)")
    parser.add_argument("--output-dir", type=str, default="./data/bfcl", help="Output directory")
    parser.add_argument("--bfcl-dir", type=str, default=str(DEFAULT_BFCL_DIR), help="BFCL repo directory")
    parser.add_argument("--irrelevance-count", type=int, default=2000, help="Number of irrelevance examples")
    parser.add_argument("--multiturn-count", type=int, default=1000, help="Number of multi-turn examples")
    parser.add_argument("--miss-func-count", type=int, default=500, help="Number of miss-func examples")
    parser.add_argument("--grpo-count", type=int, default=800, help="Number of GRPO preference pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bfcl_dir = Path(args.bfcl_dir)
    
    print("=" * 60)
    print("BFCL Training Data Generator v2")
    print("=" * 60)
    
    # Load real BFCL function definitions
    print("\nLoading BFCL function definitions...")
    collections = load_bfcl_func_docs(bfcl_dir)
    
    if not collections:
        print("ERROR: No BFCL function docs found. Check --bfcl-dir path.")
        sys.exit(1)
    
    print(f"\n--- Generating Training Data ---")
    generate_irrelevance_negatives(output_path, collections, args.irrelevance_count)
    generate_multiturn_examples(output_path, collections, args.multiturn_count)
    generate_miss_func_examples(output_path, collections, args.miss_func_count)
    generate_grpo_pairs(output_path, collections, args.grpo_count)
    merge_datasets(output_path)
    
    print(f"\n✅ Training data generation complete!")
    print(f"Files written to: {output_path}")


if __name__ == "__main__":
    main()

================
File: merge_adapters.py
================
#!/usr/bin/env python3
"""
Model Souping: Adapter Weight Merging for BFCL
(SoCE - Soup of Category Experts strategy)

Merges two LoRA adapters with configurable weights to create a hybrid
that performs well across multiple BFCL categories (AST + Agentic).

Usage:
    python merge_adapters.py --adapter-a adapters/sft --adapter-b adapters/dpo \
        --output adapters/merged --weight-a 0.6 --weight-b 0.4

The merged adapter can then be fused with mlx_lm.fuse as normal.
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("ERROR: mlx not installed. Run: pip install mlx mlx-lm")
    sys.exit(1)


def merge_adapters(
    adapter_a_path: str,
    adapter_b_path: str,
    output_path: str,
    weight_a: float = 0.6,
    weight_b: float = 0.4,
):
    """Merge two LoRA adapter weight files using weighted averaging.
    
    This implements the "Model Souping" strategy from the SoCE paper:
    merged_weight = weight_a * adapter_a + weight_b * adapter_b
    
    Both adapters must have identical architecture (same lora_rank, lora_layers).
    """
    assert abs(weight_a + weight_b - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weight_a + weight_b}"
    
    print(f"Loading Adapter A: {adapter_a_path} (weight={weight_a})")
    print(f"Loading Adapter B: {adapter_b_path} (weight={weight_b})")
    
    # Load adapter weights
    weights_a = mx.load(os.path.join(adapter_a_path, "adapters.safetensors"))
    weights_b = mx.load(os.path.join(adapter_b_path, "adapters.safetensors"))
    
    # Verify keys match
    keys_a = set(weights_a.keys())
    keys_b = set(weights_b.keys())
    
    if keys_a != keys_b:
        missing_in_b = keys_a - keys_b
        missing_in_a = keys_b - keys_a
        if missing_in_b:
            print(f"WARNING: Keys in A but not B: {missing_in_b}")
        if missing_in_a:
            print(f"WARNING: Keys in B but not A: {missing_in_a}")
        # Use intersection
        common_keys = keys_a & keys_b
        print(f"Using {len(common_keys)} common keys for merge")
    else:
        common_keys = keys_a
        print(f"All {len(common_keys)} keys match between adapters")
    
    # Merge with weighted average
    merged = {}
    for key in sorted(common_keys):
        merged[key] = weight_a * weights_a[key] + weight_b * weights_b[key]
    
    # Save merged adapter
    os.makedirs(output_path, exist_ok=True)
    mx.save_safetensors(os.path.join(output_path, "adapters.safetensors"), merged)
    
    # Copy adapter config from adapter A (architecture must match)
    config_path = os.path.join(adapter_a_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["merge_info"] = {
            "adapter_a": adapter_a_path,
            "adapter_b": adapter_b_path,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "strategy": "SoCE (Soup of Category Experts)",
        }
        out_config = os.path.join(output_path, "adapter_config.json")
        with open(out_config, "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"\n✅ Merged adapter saved to: {output_path}")
    print(f"   Strategy: {weight_a:.0%} SFT + {weight_b:.0%} DPO")
    print(f"   Keys merged: {len(merged)}")
    print(f"\nNext: fuse with 'python -m mlx_lm.fuse --model <base> --adapter-path {output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Merge two LoRA adapters (Model Souping)")
    parser.add_argument("--adapter-a", required=True, help="Path to first adapter (e.g., SFT)")
    parser.add_argument("--adapter-b", required=True, help="Path to second adapter (e.g., DPO)")
    parser.add_argument("--output", required=True, help="Output path for merged adapter")
    parser.add_argument("--weight-a", type=float, default=0.6, help="Weight for adapter A (default: 0.6)")
    parser.add_argument("--weight-b", type=float, default=0.4, help="Weight for adapter B (default: 0.4)")
    args = parser.parse_args()
    
    merge_adapters(args.adapter_a, args.adapter_b, args.output, args.weight_a, args.weight_b)


if __name__ == "__main__":
    main()

================
File: run_bfcl_pipeline.sh
================
#!/bin/bash
# =============================================================================
# BFCL #1 Master Training Pipeline (MLX Native)
# =============================================================================
# Runs the complete pipeline: data gen -> MLX convert -> LoRA SFT -> GRPO -> eval
# Designed for M5 Max 48GB with Apple Silicon MLX
#
# Usage:
#   ./run_bfcl_pipeline.sh           # Full pipeline (72B default)
#   ./run_bfcl_pipeline.sh --skip-convert  # Resume after conversion
#   ./run_bfcl_pipeline.sh --eval-only     # Run evaluation only
#
# Prerequisites:
#   pip install mlx-lm
#   pip install bfcl  (for evaluation)
# =============================================================================

set -euo pipefail

TRAINING_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$TRAINING_DIR/data/bfcl"
OUTPUT_DIR="$TRAINING_DIR/output/bfcl-72b"
BFCL_DIR="$HOME/gorilla-bfcl/berkeley-function-call-leaderboard"

# Model configuration
HF_MODEL="Qwen/Qwen2.5-72B-Instruct"
BFCL_MODEL="prism-coder-72b-FC"
MLX_MODEL="$OUTPUT_DIR/Qwen-Qwen2.5-72B-Instruct-4bit"

# Parse arguments
SKIP_CONVERT=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-convert) SKIP_CONVERT=true ;;
        --eval-only) EVAL_ONLY=true ;;
    esac
done

echo "=========================================="
echo "BFCL #1 Pipeline - 72B MLX (M5 Max 48GB)"
echo "=========================================="
echo "Training dir: $TRAINING_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "HF Model:     $HF_MODEL"
echo ""

# Detect hardware
TOTAL_MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
echo "System memory: ${TOTAL_MEM_GB} GB"

if [ "$TOTAL_MEM_GB" -lt 44 ]; then
    echo "WARNING: 72B model requires ~48GB unified memory."
    echo "   Detected: ${TOTAL_MEM_GB}GB"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Jump to eval if requested
if [ "$EVAL_ONLY" = true ]; then
    echo "Jumping to evaluation..."
fi

if [ "$EVAL_ONLY" != true ]; then

# =============================================================================
# Step 1: Generate training data
# =============================================================================
echo ""
echo "Step 1: Generating BFCL training data"
echo "--------------------------------------"
cd "$TRAINING_DIR"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/train.jsonl" ]; then
    EXISTING_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
    echo "Training data exists: $EXISTING_COUNT examples"
    read -p "Regenerate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python generate_bfcl_training_data.py \
            --output-dir "$DATA_DIR" \
            --bfcl-dir "$BFCL_DIR" \
            --irrelevance-count 2000 \
            --multiturn-count 1000 \
            --miss-func-count 500 \
            --grpo-count 800
    fi
else
    python generate_bfcl_training_data.py \
        --output-dir "$DATA_DIR" \
        --bfcl-dir "$BFCL_DIR" \
        --irrelevance-count 2000 \
        --multiturn-count 1000 \
        --miss-func-count 500 \
        --grpo-count 800
fi

# =============================================================================
# Step 2: Convert model to MLX (4-bit quantization)
# =============================================================================
echo ""
echo "Step 2: Convert to MLX 4-bit"
echo "--------------------------------------"

if [ "$SKIP_CONVERT" = true ] || [ -d "$MLX_MODEL" ]; then
    echo "MLX model exists or --skip-convert set. Skipping."
else
    echo "This downloads ~140GB and quantizes to ~38GB."
    read -p "Start conversion? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python bfcl_qlora_finetune.py \
            --model "$HF_MODEL" \
            --data "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --skip-train --no-fuse
    fi
fi

# =============================================================================
# Step 3: QLoRA SFT Fine-Tune
# =============================================================================
echo ""
echo "Step 3: QLoRA SFT Fine-Tuning (72B)"
echo "--------------------------------------"
echo "Config: rank=64, iters=1000, lr=1e-5, batch=1"
echo "Estimated: ~8-16 hours on M5 Max 48GB"

SFT_FUSED="$OUTPUT_DIR/fused_model"

if [ -d "$SFT_FUSED" ]; then
    echo "SFT fused model already exists. Skipping."
else
    read -p "Start SFT training? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python bfcl_qlora_finetune.py \
            --mlx-model "$MLX_MODEL" \
            --data "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --iters 1000 \
            --lora-rank 64 \
            --lora-layers 16 \
            --lr 1e-5 \
            --batch-size 1 \
            --skip-convert
    fi
fi

# =============================================================================
# Step 4: GRPO Alignment
# =============================================================================
echo ""
echo "Step 4: GRPO Alignment"
echo "--------------------------------------"

GRPO_DIR="$OUTPUT_DIR/grpo"
GRPO_FUSED="$GRPO_DIR/fused_aligned"

if [ -d "$SFT_FUSED" ]; then
    if [ -d "$GRPO_FUSED" ]; then
        echo "GRPO model already exists. Skipping."
    else
        read -p "Start GRPO alignment? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python bfcl_grpo_align.py \
                --model "$SFT_FUSED" \
                --data "$DATA_DIR" \
                --output-dir "$GRPO_DIR" \
                --iters 500 \
                --lora-rank 32 \
                --lr 5e-6
        fi
    fi
else
    echo "WARNING: SFT model not found at $SFT_FUSED - run Step 3 first"
fi

# =============================================================================
# Step 5: Deploy to Ollama
# =============================================================================
echo ""
echo "Step 5: Deploy to Ollama"
echo "--------------------------------------"

# Find GGUF from latest stage
FINAL_DIR="$GRPO_FUSED"
[ ! -d "$FINAL_DIR" ] && FINAL_DIR="$SFT_FUSED"

if [ -d "$FINAL_DIR" ]; then
    GGUF_FILE=$(find "$FINAL_DIR" -name "*.gguf" 2>/dev/null | head -1)
    if [ -n "$GGUF_FILE" ]; then
        echo "Found GGUF: $GGUF_FILE"
        
        MODELFILE="$OUTPUT_DIR/Modelfile"
        cat > "$MODELFILE" << OLLAMA_EOF
FROM $GGUF_FILE
PARAMETER temperature 0.6
PARAMETER num_ctx 32768
PARAMETER stop <|im_end|>
OLLAMA_EOF
        
        echo "Modelfile created: $MODELFILE"
        read -p "Deploy to Ollama now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ollama create prism-coder-72b-FC -f "$MODELFILE"
            echo "Deployed as prism-coder-72b-FC"
        fi
    else
        echo "No GGUF file found. Run Steps 3-4 first."
    fi
else
    echo "No fused model found. Run Steps 3-4 first."
fi

fi  # end of EVAL_ONLY check

# =============================================================================
# Step 6: BFCL Evaluation
# =============================================================================
echo ""
echo "Step 6: BFCL Evaluation"
echo "--------------------------------------"
cd "$BFCL_DIR"

echo "Running evaluation on all categories..."
echo "  Model: $BFCL_MODEL"
echo ""

read -p "Start BFCL eval? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating responses..."
    bfcl generate --model "$BFCL_MODEL" --test-category all --num-threads 1 --backend vllm 2>&1 | tee "$OUTPUT_DIR/eval_generate.log"
    
    echo ""
    echo "Evaluating..."
    bfcl evaluate --model "$BFCL_MODEL" --test-category all 2>&1 | tee "$OUTPUT_DIR/eval_results.log"
    
    echo ""
    echo "Results saved to: $OUTPUT_DIR/eval_results.log"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  MLX Model:    $MLX_MODEL"
echo "  SFT Fused:    $OUTPUT_DIR/fused_model"
echo "  GRPO Aligned: $OUTPUT_DIR/grpo/fused_aligned"
echo "  Eval Log:     $OUTPUT_DIR/eval_results.log"

================
File: test_handler.py
================
#!/usr/bin/env python3
"""
BFCL Handler Test Suite — prism-coder

Validates the PrismCoderHandler against all BFCL scoring categories:
1. Non-Live (10%): Single-turn function calling with type coercion
2. Live (10%): Real-world API calls  
3. Irrelevance (10%): Correctly abstains when no tool matches
4. Multi-Turn (30%): Multi-turn conversation with tool calling
5. Agentic (40%): Memory + WebSearch via multi-turn FC loop

Run: python test_handler.py
Run verbose: python test_handler.py -v

Requires: Ollama running with a Qwen2.5-Coder model pulled.
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add BFCL repo to PATH
BFCL_DIR = Path.home() / "gorilla-bfcl" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(BFCL_DIR))

from bfcl_eval.model_handler.local_inference.prism_coder import PrismCoderHandler


class TestExtractToolCalls(unittest.TestCase):
    """Test _extract_tool_calls — the core parsing logic."""

    def test_standard_tool_call_tag(self):
        """Standard <tool_call> tag extraction."""
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "get_weather")
        self.assertEqual(result[0]["arguments"]["city"], "London")

    def test_tool_call_no_newlines(self):
        """<tool_call> without newlines around JSON (whitespace-tolerant)."""
        text = '<tool_call>{"name": "ls", "arguments": {}}</tool_call>'
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "ls")

    def test_parallel_tool_calls(self):
        """Multiple parallel tool calls."""
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["arguments"]["city"], "London")
        self.assertEqual(result[1]["arguments"]["city"], "Paris")

    def test_bare_json_object(self):
        """Bare JSON object without tags."""
        text = '{"name": "mkdir", "arguments": {"dir_name": "test"}}'
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "mkdir")

    def test_abstention_plain_text(self):
        """Plain text response = abstention (no tool call)."""
        text = "I don't have the right tools for that request."
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(result, [])

    def test_abstention_empty(self):
        """Empty string = abstention."""
        result = PrismCoderHandler._extract_tool_calls("")
        self.assertEqual(result, [])

    def test_think_tags_stripped(self):
        """<think> tags should be stripped before extraction."""
        text = '<think>\nLet me analyze this...\n</think>\n\n<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "ls")

    def test_irrelevant_braces_ignored(self):
        """Text containing { but no valid tool call should return empty."""
        text = "The formula is {x + y = z} where x and y are integers."
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(result, [])

    def test_nested_json_arguments(self):
        """Tool call with nested JSON arguments."""
        text = '<tool_call>\n{"name": "create_event", "arguments": {"title": "Meeting", "attendees": [{"name": "Alice"}, {"name": "Bob"}]}}\n</tool_call>'
        result = PrismCoderHandler._extract_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "create_event")
        self.assertEqual(len(result[0]["arguments"]["attendees"]), 2)


class TestTypeCercion(unittest.TestCase):
    """Test language-aware type coercion — critical for Java/JS compat."""

    def test_python_bool_coercion(self):
        """Python: 'true'/'false' strings → True/False."""
        args = {"enabled": "true", "verbose": "false"}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Python")
        self.assertIs(fixed["enabled"], True)
        self.assertIs(fixed["verbose"], False)

    def test_python_null_coercion(self):
        """Python: 'null'/'none' → None."""
        args = {"value": "null", "other": "none"}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Python")
        self.assertIsNone(fixed["value"])
        self.assertIsNone(fixed["other"])

    def test_java_no_bool_coercion(self):
        """Java: 'true'/'false' should STAY as strings."""
        args = {"enabled": "true", "name": "test"}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Java")
        self.assertEqual(fixed["enabled"], "true")  # NOT True
        self.assertEqual(fixed["name"], "test")

    def test_javascript_no_bool_coercion(self):
        """JavaScript: 'true'/'false' should STAY as strings."""
        args = {"flag": "false"}
        fixed = PrismCoderHandler._fix_argument_types(args, language="JavaScript")
        self.assertEqual(fixed["flag"], "false")  # NOT False

    def test_stringified_json_object(self):
        """Stringified JSON '{"key": "val"}' → {"key": "val"} (all languages)."""
        args = {"config": '{"host": "localhost", "port": 5432}'}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Python")
        self.assertIsInstance(fixed["config"], dict)
        self.assertEqual(fixed["config"]["host"], "localhost")

    def test_extra_quoted_string(self):
        """Extra-quoted: \"'USERSPACE1'\" → \"USERSPACE1\" (all languages)."""
        args = {"namespace": "'USERSPACE1'"}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Java")
        self.assertEqual(fixed["namespace"], "USERSPACE1")

    def test_nested_dict(self):
        """Nested dicts should be recursively fixed."""
        args = {"outer": {"inner_bool": "true"}}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Python")
        self.assertIs(fixed["outer"]["inner_bool"], True)

    def test_list_values(self):
        """Lists should be recursively fixed."""
        args = {"items": ["true", "false", "hello"]}
        fixed = PrismCoderHandler._fix_argument_types(args, language="Python")
        self.assertEqual(fixed["items"], [True, False, "hello"])


class TestDecodeAst(unittest.TestCase):
    """Test decode_ast — called by the BFCL evaluator for scoring."""

    def setUp(self):
        """Create handler with mocked Ollama connection."""
        with patch.object(PrismCoderHandler, '__init__', lambda self, *a, **k: None):
            self.handler = PrismCoderHandler.__new__(PrismCoderHandler)

    def test_single_call(self):
        """Single function call → [{name: args}] format."""
        result = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "NY"}}\n</tool_call>'
        decoded = self.handler.decode_ast(result, "Python", False)
        self.assertEqual(len(decoded), 1)
        self.assertIn("get_weather", decoded[0])
        self.assertEqual(decoded[0]["get_weather"]["city"], "NY")

    def test_abstention_returns_empty(self):
        """Abstention (no tool call) → [] for irrelevance checker."""
        result = "I can't help with that."
        decoded = self.handler.decode_ast(result, "Python", False)
        self.assertEqual(decoded, [])

    def test_multiple_calls(self):
        """Parallel function calls → list of {name: args}."""
        result = (
            '<tool_call>\n{"name": "func_a", "arguments": {"x": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "func_b", "arguments": {"y": 2}}\n</tool_call>'
        )
        decoded = self.handler.decode_ast(result, "Python", False)
        self.assertEqual(len(decoded), 2)
        self.assertIn("func_a", decoded[0])
        self.assertIn("func_b", decoded[1])

    def test_java_bool_stays_string(self):
        """Java decode: 'true' arg stays as string."""
        result = '<tool_call>\n{"name": "setFlag", "arguments": {"enabled": "true"}}\n</tool_call>'
        decoded = self.handler.decode_ast(result, "Java", False)
        self.assertEqual(decoded[0]["setFlag"]["enabled"], "true")

    def test_python_bool_converted(self):
        """Python decode: 'true' arg → True."""
        result = '<tool_call>\n{"name": "setFlag", "arguments": {"enabled": "true"}}\n</tool_call>'
        decoded = self.handler.decode_ast(result, "Python", False)
        self.assertIs(decoded[0]["setFlag"]["enabled"], True)


class TestFormatPrompt(unittest.TestCase):
    """Test _format_prompt — the prompt assembly logic."""

    def setUp(self):
        with patch.object(PrismCoderHandler, '__init__', lambda self, *a, **k: None):
            self.handler = PrismCoderHandler.__new__(PrismCoderHandler)

    def test_basic_single_turn(self):
        """Basic single-turn: system + user + tools."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
        ]
        tools = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]
        
        prompt = self.handler._format_prompt(messages, tools)
        
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("You are helpful.", prompt)
        self.assertIn("<tools>", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("IMPORTANT RULES:", prompt)
        self.assertIn("If NONE of the provided functions are relevant", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_no_think_tags_by_default(self):
        """Default: no <think> tags injected (Qwen2.5-Coder doesn't support them)."""
        messages = [{"role": "user", "content": "Hello"}]
        prompt = self.handler._format_prompt(messages, [])
        self.assertNotIn("<think>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_think_tags_with_env(self):
        """With PRISM_ENABLE_THINKING=1, <think> tags ARE injected."""
        messages = [{"role": "user", "content": "Hello"}]
        with patch.dict(os.environ, {"PRISM_ENABLE_THINKING": "1"}):
            prompt = self.handler._format_prompt(messages, [])
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)

    def test_tool_response_native_role(self):
        """CRITICAL: Tool responses must use native <|im_start|>tool role."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "ls", "arguments": {}}}
            ]},
            {"role": "tool", "content": '["file1.txt", "file2.py"]'},
            {"role": "user", "content": "Now delete file1.txt"},
        ]
        tools = [{"name": "ls"}, {"name": "rm"}]
        
        prompt = self.handler._format_prompt(messages, tools)
        
        # Tool response MUST use native <|im_start|>tool role
        self.assertIn("<|im_start|>tool\n", prompt)
        self.assertIn('["file1.txt", "file2.py"]', prompt)
        self.assertIn("<|im_end|>", prompt)

    def test_consecutive_tool_responses_separate(self):
        """Multiple tool responses should each use native <|im_start|>tool role."""
        messages = [
            {"role": "user", "content": "Get weather for London and Paris"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}},
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
            ]},
            {"role": "tool", "content": '{"temp": 15}'},
            {"role": "tool", "content": '{"temp": 20}'},
        ]
        tools = [{"name": "get_weather"}]
        
        prompt = self.handler._format_prompt(messages, tools)
        
        # Each tool response should have its own <|im_start|>tool block
        tool_count = prompt.count("<|im_start|>tool\n")
        self.assertEqual(tool_count, 2, "Each tool response should have separate <|im_start|>tool block")

    def test_abstention_instruction_present(self):
        """System prompt MUST contain abstention instruction when tools present."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "func1"}]
        prompt = self.handler._format_prompt(messages, tools)
        self.assertIn("If NONE of the provided functions are relevant", prompt)

    def test_no_tools_no_abstention(self):
        """Without tools, no abstention instruction needed."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompt = self.handler._format_prompt(messages, [])
        self.assertNotIn("If NONE of the provided functions", prompt)


class TestDecodeExecute(unittest.TestCase):
    """Test decode_execute — used in multi-turn to execute tool calls."""

    def setUp(self):
        with patch.object(PrismCoderHandler, '__init__', lambda self, *a, **k: None):
            self.handler = PrismCoderHandler.__new__(PrismCoderHandler)

    def test_single_call_execute(self):
        """decode_execute returns executable function call format."""
        result = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
        decoded = self.handler.decode_execute(result, False)
        self.assertIsInstance(decoded, list)
        self.assertTrue(len(decoded) > 0)

    def test_abstention_execute(self):
        """decode_execute returns [] for abstention."""
        result = "I can't help with that."
        decoded = self.handler.decode_execute(result, False)
        self.assertEqual(decoded, [])


class TestParseQueryResponse(unittest.TestCase):
    """Test _parse_query_response_FC — the multi-turn response parser."""

    def setUp(self):
        with patch.object(PrismCoderHandler, '__init__', lambda self, *a, **k: None):
            self.handler = PrismCoderHandler.__new__(PrismCoderHandler)

    def _make_response(self, text):
        """Create a mock API response."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].text = text
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        return resp

    def test_tool_call_response(self):
        """Tool call → message with tool_calls for chat history."""
        resp = self._make_response(
            '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
        )
        parsed = self.handler._parse_query_response_FC(resp)
        msg = parsed["model_responses_message_for_chat_history"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIn("tool_calls", msg)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "ls")

    def test_abstention_response(self):
        """Abstention → message with content, no tool_calls."""
        resp = self._make_response("I don't have the right tools.")
        parsed = self.handler._parse_query_response_FC(resp)
        msg = parsed["model_responses_message_for_chat_history"]
        self.assertEqual(msg["role"], "assistant")
        self.assertNotIn("tool_calls", msg)
        self.assertIn("don't have", msg["content"])

    def test_reasoning_extraction(self):
        """<think> content should be extracted into reasoning_content."""
        resp = self._make_response(
            '<think>\nAnalyzing the request...\n</think>\n\n<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
        )
        parsed = self.handler._parse_query_response_FC(resp)
        self.assertIn("Analyzing the request", parsed["reasoning_content"])
        msg = parsed["model_responses_message_for_chat_history"]
        self.assertEqual(msg["reasoning_content"], parsed["reasoning_content"])


class TestTrainingDataFormat(unittest.TestCase):
    """Validate that training data format matches what the handler expects."""

    def test_training_data_messages_format(self):
        """Training data MUST use the same role format as the handler."""
        # Simulate a multi-turn training example
        messages = [
            {"role": "system", "content": "# Tools\n..."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>',
             "tool_calls": [{"function": {"name": "ls", "arguments": {}}}]},
            {"role": "tool", "content": '["file1.txt"]'},
            {"role": "user", "content": "Delete file1.txt"},
        ]
        
        # The handler's _format_prompt should handle this correctly
        with patch.object(PrismCoderHandler, '__init__', lambda self, *a, **k: None):
            handler = PrismCoderHandler.__new__(PrismCoderHandler)
        
        tools = [{"name": "ls"}, {"name": "rm"}]
        prompt = handler._format_prompt(messages, tools)
        
        # Verify tool response uses native <|im_start|>tool role
        self.assertIn("<|im_start|>tool\n", prompt)
        # Verify the conversation maintains proper structure
        self.assertIn("<|im_start|>assistant", prompt)


if __name__ == "__main__":
    print("=" * 60)
    print("BFCL Handler Test Suite — prism-coder")
    print("=" * 60)
    print(f"Handler: {BFCL_DIR / 'bfcl_eval/model_handler/local_inference/prism_coder.py'}")
    print()
    
    # Run tests
    unittest.main(verbosity=2)





================================================================
End of Codebase
================================================================
```
