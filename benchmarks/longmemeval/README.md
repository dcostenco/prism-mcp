# LongMemEval Benchmark — Prism MCP

Runs the [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) benchmark (ICLR 2025) against Prism's hybrid retrieval system.

## Results

| Category | R@5 | Score |
|----------|-----|-------|
| **Overall** | **92.3%** | **434/470** |
| single-session-assistant | 98.2% | 55/56 |
| multi-session | 95.9% | 116/121 |
| single-session-preference | 93.3% | 28/30 |
| knowledge-update | 91.7% | 66/72 |
| temporal-reasoning | 89.0% | 113/127 |
| single-session-user | 87.5% | 56/64 |

## Setup

```bash
# Download the dataset (~264MB)
mkdir -p data && cd data
curl -L -o longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ..

# Ensure Ollama is running with nomic-embed-text
ollama pull nomic-embed-text
```

## Running

```bash
# Full benchmark (500 questions, ~10 min)
npx tsx benchmarks/longmemeval/run_benchmark.ts

# Quick test (10 questions)
npx tsx benchmarks/longmemeval/run_benchmark.ts --limit 10 --skip-qa

# With QA generation (requires Ollama or OPENAI_API_KEY)
npx tsx benchmarks/longmemeval/run_benchmark.ts --model gpt-4o-mini
```

## Methodology

1. For each of the 500 questions, a fresh libSQL database is created
2. All haystack sessions (~40-50 per question) are ingested with FTS5 indexing and vector embeddings
3. The question is used to retrieve top-5 sessions via hybrid search (FTS5 + cosine similarity)
4. Session-level recall (R@K) measures whether any answer session appears in the top-K retrieved set
5. 30 abstention questions are excluded from retrieval metrics (per LongMemEval convention)

**Embedding model:** `nomic-embed-text` via Ollama (768-dim, local, free)
**Retrieval:** Hybrid FTS5 keyword search + vector cosine similarity with FTS boost
