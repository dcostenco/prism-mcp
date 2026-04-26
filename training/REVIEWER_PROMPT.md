# External Code Review Prompt — BFCL Training Pipeline

Paste the following prompt alongside the contents of `repomix-output.txt`.

---

## Prompt for External Reviewer

```
You are an expert ML engineer specializing in function-calling LLM fine-tuning, 
BFCL (Berkeley Function Calling Leaderboard) evaluation methodology, and Apple 
Silicon MLX training pipelines. Perform a deep adversarial code review.

## CONTEXT

Goal: Train Salesforce/xLAM-2-32b-fc-r to achieve #1 on BFCL V4 (current #1 
is Claude Opus 4.5 at 77.47%). The model outputs in a proprietary format called
"Synalux" with these special tokens:

- <|synalux_think|> / </|synalux_think|>  — internal reasoning (CoT)
- <|tool_call|> / </|tool_call|>  — tool invocations (JSON body)
- <|tool_response|> / </|tool_response|>  — tool results
- <|synalux_answer|> / </|synalux_answer|>  — final user-facing answer
- <|memory_query|> / </|memory_query|>  — memory retrieval

The training pipeline runs on Apple Silicon M5 Max 48GB using MLX-LM (not PyTorch/CUDA).

## BFCL V4 SCORING (critical — weights determine training data mix)

Overall = Agentic×40% + Multi-Turn×30% + Live×10% + Non-Live×10% + Hallucination×10%

## RECENT BUG FIXES (verify these were done correctly)

14 bugs were just fixed in this codebase:

CRITICAL (6):
- C1: All <tool_call> tokens replaced with Synalux <|tool_call|> / </|tool_call|>
- C2: DPO rejected samples updated to use Synalux tokens
- C3: Double <|im_start|>assistant prefix removed from GRPO prompts
- C4: Non-existent mlx_lm.dpo replaced with mlx_lm.lora
- C5: Generic system prompt replaced with Synalux SYSTEM_PROMPT
- C6: GRPO pairs format aligned (was raw text, now should use ChatML)

MAJOR (5):
- M1: Eval tool registry expanded from 17 Prism to 30 Synalux tools
- M2: Eval model changed from prism-coder:7b to synalux-coder-32b-FC
- M3: Added <|synalux_think|> reasoning traces to training data
- M4: Training data mix adjusted (was missing agentic-specific data)
- M5: Vehicle scenario wrong function mappings fixed

MINOR (3):
- m1: Tool role changed to use <|tool_response|> wrapper
- m2: Eval/train token consistency (resolved by C1)
- m3: Interactive read -p replaced with non-interactive logic

## YOUR REVIEW TASKS

1. **Token Consistency Audit**: Verify EVERY occurrence of tool-related tokens 
   across ALL files uses the correct Synalux format. Flag any remaining <tool_call> 
   or </tool_call> without the pipe delimiters.

2. **Train-Inference Format Alignment**: Does the training data format 
   (system prompt + user + assistant + tool responses) match EXACTLY what the 
   Synalux inference handler would produce? Flag any gaps.

3. **MLX-LM Compatibility**: The C4 fix changed mlx_lm.dpo to mlx_lm.lora. 
   Is this correct? Does mlx_lm.lora support DPO/preference data format? 
   What data format changes are needed?

4. **BFCL V4 Coverage**: Does the training data adequately cover all 22 BFCL V4 
   test tasks? (web_search_base, web_search_no_snippet, memory_kv, memory_vector, 
   memory_rec_sum, multi_turn_base, miss_func, miss_param, long_context, 
   live_simple, live_multiple, live_parallel, live_parallel_multiple, 
   simple_python, simple_java, simple_js, multiple, parallel, parallel_multiple, 
   irrelevance, live_irrelevance). Flag any categories with zero coverage.

5. **Memory Budget**: Verify that the training configs (batch_size=4, seq_len=16384, 
   lora_rank=64, lora_layers=24) fit within 48GB - 17GB model - 3GB OS = 28GB 
   activation headroom. Flag any OOM risk.

6. **Reward Function**: The R2IF composite reward 
   (R_format + R_correct + R_CER + R_SMV) is described but not yet implemented.
   Is the current mlx_lm.lora-based preference training an adequate substitute? 
   What's the expected performance gap?

7. **Data Quality Traps**: Check for:
   - Hardcoded examples that could cause overfitting
   - Duplicate training pairs
   - Category class imbalance
   - Synthetic data that contradicts real BFCL tool schemas

8. **Model Souping**: Phase 5 uses SLERP merge of checkpoints. Is this 
   compatible with LoRA adapters / GGUF export in MLX? Flag implementation gaps.

## OUTPUT FORMAT

For each finding, use this format:

### [SEVERITY: CRITICAL/MAJOR/MINOR/INFO] Finding Title
- **File**: filename:line_range
- **Bug**: What's wrong
- **Impact**: How it affects BFCL score or training
- **Fix**: Exact code change needed

End with a summary table of all findings sorted by severity.
```

---

## How to Use

```bash
# 1. Generate the repomix (already done)
cd /Users/admin/prism
npx repomix --include "training/**/*.py,training/**/*.sh" \
  --output training/repomix-output.txt

# 2. Copy this prompt + repomix output into your preferred LLM reviewer:
#    - Claude Opus 4.5 (recommended for ML review depth)
#    - GPT-4o (good alternative)
#    - Gemini 2.5 Pro (strong at code review)

# 3. Paste: this prompt FIRST, then the full repomix-output.txt content
```
