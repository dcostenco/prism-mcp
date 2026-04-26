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
