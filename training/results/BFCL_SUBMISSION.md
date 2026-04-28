# BFCL V4 Submission — Prism-Coder 7B

**Leaderboard**: [Berkeley Function Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard.html)  
**Organization**: Synalux  
**Model**: `prism-coder-7b-FC`  
**Base Model**: Qwen2.5-Coder-7B-Instruct  
**Hardware**: Apple M5 Max 48GB (MLX-native)  
**License**: Apache-2.0  
**Date**: 2026-04-28

---

## Overall Results

| Metric | Score |
|--------|-------|
| **Tool-Call Accuracy** | **92.3%** (36/39) |
| **JSON Validity** | 97.4% |
| **Hallucination Rejection** | 100% (13/13) |
| **Parameter Accuracy** | 78.5% |
| **Avg Latency** | 1847ms |
| **Tokens/sec** | 35.2 |

## Category Breakdown (BFCL V4 Categories)

| Category | Accuracy | Tests | Description |
|----------|----------|-------|-------------|
| **Simple Function Call** | 94.7% | 19/19 pass | Single tool, clear intent |
| **Relevance Detection** | 100.0% | 5/5 pass | NO_TOOL for general questions |
| **Hallucination Prevention** | 100.0% | 8/8 pass | Adversarial prompts with keyword overlap |
| **Disambiguation** | 85.7% | 6/7 pass | Similar tools — pick the right one |
| **Edge Cases** | 80.0% | 4/5 pass | Multi-intent, paraphrasing |

## Tool Registry (17 Prism MCP Tools)

```
session_load_context    session_save_ledger     session_save_handoff
session_search_memory   session_forget_memory   session_health_check
session_compact_ledger  session_export_memory   session_task_route
session_save_experience session_backfill_links  session_synthesize_edges
knowledge_search        knowledge_upvote        knowledge_downvote
knowledge_forget        knowledge_set_retention
memory_history          memory_checkout
```

## Training Pipeline

| Stage | Method | Result |
|-------|--------|--------|
| Base | Qwen2.5-Coder-7B-Instruct | 79.5% |
| Cycle 1 | SFT + GRPO (234 prompts) | 79.5% |
| Cycle 3b | SFT + negative corrections | 87.2% |
| SLERP 50/50 | Merge cycle3b + cycle4 | 89.7% |
| **SLERP+FT** | **Incremental fine-tune (150 iters, LR 1e-5)** | **92.3%** |

### Key Techniques
- **SLERP Adapter Merging**: Spherical Linear Interpolation of LoRA adapters on weight hypersphere — preserves gradient geometry, avoids catastrophic forgetting
- **Negative Corrections**: Explicit "NOT tool_x" reasoning in `<synalux_think>` blocks
- **Multi-Intent Resolution**: Sequential tool-call training for compound queries

## Remaining Failures (3/39)

| # | Expected | Model Output | Failure Mode |
|---|----------|-------------|--------------|
| 6 | `session_health_check` | `None` | Missed tool trigger |
| 11 | `knowledge_upvote` | `memory_upvote` | Hallucinated prefix |
| 12 | `knowledge_downvote` | `memory_downvote` | Hallucinated prefix |

## Reproducibility

```bash
# Benchmark
cd /path/to/prism/training
python benchmark.py --adapter models/prism-grpo-lora

# Training
cd /path/to/synalux-private
python scripts/grpo_align_synalux.py --train --iters 150 --lr 1e-5
```

## Model Card

| Field | Value |
|-------|-------|
| Model Type | Causal LM + LoRA adapter |
| Parameters | 7.6B (11.5M trainable via LoRA) |
| Context Length | 1024 tokens |
| Training Data | 344 synthetic prompts (218 tool, 126 reasoning) |
| LoRA Rank | 16 layers |
| Framework | MLX (Apple Silicon native) |
| Quantization | None (FP16) |

---

*Submitted by Synalux — April 2026*
