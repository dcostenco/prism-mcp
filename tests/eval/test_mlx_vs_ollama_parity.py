#!/usr/bin/env python3
"""
MLX vs Ollama parity test.

Prevents the regression where the MLX direct-eval harness scored 17 points
LOWER than the Ollama-based 100-case benchmark on the SAME base model
(Qwen3-14B). The root cause was Qwen3's default chat template enabling
"thinking mode" — the model emits a long `<think>...</think>` block that
eats all `max_tokens` before it can emit the `<|tool_call|>` block. Ollama
ships a Modelfile that disables thinking; MLX's `apply_chat_template` does
NOT by default. Fix: pass `enable_thinking=False`.

This test pins the parity behavior:

  1. The chat template MUST emit `<think>\n\n</think>` when called with
     `enable_thinking=False`. If a future Qwen tokenizer version drops or
     renames this kwarg silently, the test catches it.
  2. A 20-prompt golden set is routed through BOTH MLX and Ollama. The
     two routes must agree on ≥18/20 tool-call decisions (90% agreement).

If parity drifts >2 points, the test fails and forces investigation
BEFORE someone burns cloud GPU on a model whose offline eval is wrong.

Run:
    pytest tests/eval/test_mlx_vs_ollama_parity.py -v
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "benchmarks" / "prism-routing-100"))


def _load_bench_constants():
    import importlib.util
    p = REPO_ROOT / "tests" / "benchmarks" / "prism-routing-100" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("bench", p)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


bench = _load_bench_constants()


GOLDEN_PROMPTS = [
    # Categories where MLX previously failed catastrophically (thinking-mode bug)
    ("hand", "Pass this to the next agent: routing is done, focus on iOS next", "session_save_handoff"),
    ("hand", "Save a handoff for prism-coder — training complete, deploy next", "session_save_handoff"),
    ("smem", "What did we discuss about BFCL last time?", "session_search_memory"),
    ("smem", "Find in my sessions anything about RunPod configuration", "session_search_memory"),
    ("load", "Load context for project synalux-health", "session_load_context"),
    ("load", "Resume context for bcba-private", "session_load_context"),
    ("save", "Note: finished migrating the auth service to JWT", "session_save_ledger"),
    ("save", "Save a ledger for prism-mcp — completed BFCL eval", "session_save_ledger"),
    # Categories that worked even without the fix (sanity checks)
    ("aac", "Suggest phrases for expressing pain", None),
    ("aac", "Give me AAC phrases for asking for help", None),
    ("tran", "Translate 'hello' into Spanish", None),
    ("tran", "How do you say 'thank you' in French?", None),
    ("irrel", "I feel tired and want to rest", None),
    ("irrel", "What's the weather like today?", None),
    ("web", "Google: latest LLM benchmarks 2026", "brave_web_search"),
    ("web", "Search the internet for current vast.ai pricing", "brave_web_search"),
    ("know", "What do I know about HIPAA compliance?", "knowledge_search"),
    ("cmpct", "Compact the ledger for synalux-private project", "session_compact_ledger"),
    ("pred", "What is the capital of France?", None),
    ("pred", "Explain what a LoRA adapter is", None),
]


def _extract_tool(text: str):
    """Match the regex used by benchmark.py — keep this in sync."""
    m = re.search(r'<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|tool_call_end\|>|$)', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("name")
    except Exception:
        return None


@pytest.fixture(scope="module")
def mlx_model():
    """Load the MLX 14B base model once."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import load
    path = REPO_ROOT / "training" / "mlx_model_qwen3_14b"
    if not path.exists():
        pytest.skip(f"MLX 14B model not found at {path}")
    return load(str(path))


def test_chat_template_supports_enable_thinking(mlx_model):
    """Pin the kwarg name — guards against future Qwen tokenizer breaking changes."""
    _, tokenizer = mlx_model
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # The signature emitted when thinking is disabled MUST be:
    #   ...<|im_start|>assistant\n<think>\n\n</think>\n\n
    assert "<think>\n\n</think>" in text, (
        f"enable_thinking=False should emit empty <think></think> block. "
        f"Got:\n{text[-200:]}"
    )


def test_chat_template_thinking_on_emits_open_think_only(mlx_model):
    """Without enable_thinking=False, Qwen3 emits an OPEN <think> tag and
    expects the model to fill in reasoning. Pin this to detect when the
    default changes (which would be a silent behavior shift)."""
    _, tokenizer = mlx_model
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
    ]
    # Default (no enable_thinking)
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    # Should NOT have the closed <think></think> pair — thinking is open
    has_closed = "<think>\n\n</think>" in text
    assert not has_closed, (
        "Default chat template should leave thinking OPEN, not pre-close it. "
        "If this fails, Qwen tokenizer flipped its default — re-evaluate "
        "MLX harness to confirm tool-call extraction still works."
    )


def _ollama_route(prompt: str, model: str = "dcostenco/prism-coder:14b") -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": bench.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 160},
            },
            timeout=120,
        )
        return r.json().get("message", {}).get("content", "")
    except Exception:
        return ""


def _mlx_route(model_pair, prompt: str) -> str:
    from mlx_lm import generate
    model, tokenizer = model_pair
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": bench.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=160, verbose=False)


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running on localhost:11434")
def test_mlx_vs_ollama_parity_on_golden_set(mlx_model):
    """The two paths must agree on ≥18/20 routing decisions (90%)."""
    matches = 0
    disagreements = []
    for cat, prompt, expected in GOLDEN_PROMPTS:
        mlx_out = _mlx_route(mlx_model, prompt)
        ollama_out = _ollama_route(prompt)
        mlx_tool = _extract_tool(mlx_out)
        ollama_tool = _extract_tool(ollama_out)
        if mlx_tool == ollama_tool:
            matches += 1
        else:
            disagreements.append((cat, prompt[:40], expected, mlx_tool, ollama_tool))
    n = len(GOLDEN_PROMPTS)
    assert matches >= n - 2, (
        f"MLX↔Ollama agreement {matches}/{n} (must be ≥{n-2}). "
        f"Disagreements:\n" + "\n".join(
            f"  [{c}] {p!r}  expect={e}  mlx={m}  ollama={o}"
            for c, p, e, m, o in disagreements
        )
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
