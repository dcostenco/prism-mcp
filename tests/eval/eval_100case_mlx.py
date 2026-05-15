#!/usr/bin/env python3
"""Run the 100-case Prism eval against an MLX model directly.

Mirrors tests/benchmarks/prism-routing-100/benchmark.py but uses
mlx_lm.generate instead of Ollama HTTP, so we can eval local fused
LoRA adapters before publishing to Ollama Hub.
"""
import argparse
import importlib.util
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from mlx_lm import generate, load

# Load TEST_POOL + SYSTEM_PROMPT from the benchmark file directly
BENCH = Path.home() / "prism/tests/benchmarks/prism-routing-100/benchmark.py"
spec = importlib.util.spec_from_file_location("bench", BENCH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def extract_tool_name(text):
    m = re.search(r'<\|tool_call\|>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: look for {"name": "...} anywhere
    m = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def score(model, tokenizer, test_pool, n=100, seed=2027):
    random.seed(seed)
    pool = list(test_pool)
    random.shuffle(pool)
    sample = pool[:n]

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_all = 0
    invented = 0
    latencies = []

    KNOWN_TOOLS = {t["name"] for t in bench.TOOLS_SCHEMA}

    for cat, prompt, expected in sample:
        # CRITICAL: Qwen3 chat template defaults to thinking-mode ON. Without
        # `enable_thinking=False` the model emits a long `<think>...</think>`
        # reasoning block that eats all 160 max_tokens before it can emit the
        # `<|tool_call|>` block — driving false-negative tool-call extraction
        # on tool-routing categories. Ollama's published Modelfile bakes the
        # nothink prefix in; MLX-LM does not. Without this, MLX scored 70%
        # vs Ollama's 87% on the SAME base 14B model.
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": bench.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        t0 = time.time()
        out = generate(model, tokenizer, prompt=prompt_text, max_tokens=160, verbose=False)
        latencies.append(time.time() - t0)

        called_tool = extract_tool_name(out)

        is_correct = False
        if expected is None:
            is_correct = called_tool is None
        else:
            is_correct = called_tool == expected

        if called_tool and called_tool not in KNOWN_TOOLS:
            invented += 1

        by_cat[cat]["total"] += 1
        if is_correct:
            by_cat[cat]["correct"] += 1
            correct_all += 1

    return {
        "overall": correct_all,
        "total": n,
        "pct": correct_all * 100 / n,
        "invented": invented,
        "avg_lat": sum(latencies) / len(latencies),
        "p50_lat": sorted(latencies)[len(latencies) // 2],
        "cats": {c: by_cat[c]["correct"] * 100 / by_cat[c]["total"]
                 if by_cat[c]["total"] > 0 else 0
                 for c in by_cat},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)
    print("Loaded.\n", flush=True)

    r = score(model, tokenizer, bench.TEST_POOL, n=args.n, seed=args.seed)
    print(f"\n{'='*60}")
    print(f"RESULT: {r['overall']}/{r['total']} = {r['pct']:.1f}%")
    print(f"  avg lat: {r['avg_lat']:.2f}s")
    print(f"  invented tools: {r['invented']}")
    print(f"\nPer-category:")
    for cat, pct in sorted(r["cats"].items()):
        print(f"  {cat:>8s}: {pct:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
