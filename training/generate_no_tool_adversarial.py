#!/usr/bin/env python3
"""Generate NO_TOOL adversarial SFT examples.

Counterpart to generate_contrastive_sft.py — these are prompts that LOOK
like Prism tool requests (use words like 'session', 'memory', 'forget',
'search', 'load') but are actually general programming/CS questions.

The <think> trace explicitly debates the tempting tool, then commits to
abstaining and answering directly.

Output: data/no_tool_adversarial.jsonl in {"text": "..."} chatml format.
"""
import os, sys, json, time, argparse, urllib.request
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "no_tool_adversarial.jsonl"
TEACHER_MODEL = "qwen2.5-coder:32b"
OLLAMA_API = "http://localhost:11434/api/generate"
EXAMPLES_PER_TRAP = 12   # 5 traps * 12 = 60 examples

# (tempting Prism tool, real programming/CS topic)
TRAPS = [
    ("session_load_context", "loading sessions in web frameworks (PHP session_start, Express middleware, Django sessions)"),
    ("session_search_memory", "searching algorithms (BFS, DFS, binary search) and search libraries (Elasticsearch, Algolia)"),
    ("session_forget_memory", "ML forget gates (LSTM, GRU), catastrophic forgetting, elastic weight consolidation"),
    ("session_export_memory", "exporting data from databases (PostgreSQL pg_dump, MySQL mysqldump), CSV/JSON serialization"),
    ("knowledge_search", "knowledge graphs, semantic search, RAG retrieval implementations in code"),
]

SYSTEM_PROMPT = """You generate adversarial NO-TOOL SFT examples for a tool-calling LLM.

Each example is a USER prompt that uses Prism-tool keywords ('session', 'memory', 'forget', 'search', 'load', 'export', 'knowledge') in a GENERAL PROGRAMMING context — NOT a request to use Prism's tools.

The ASSISTANT response must:
1. Have a <think> trace that names the tempting Prism tool, explains what made it tempting (the keyword), and identifies the discriminating signal that this is a general programming question.
2. Commit to abstaining (no tool call).
3. Provide a brief direct answer to the programming question (1-2 sentences).

Output ONLY a JSON array of {prompt, think, answer} objects. No prose, no markdown, no code fences."""

USER_TEMPLATE = """Generate {n} diverse NO-TOOL adversarial examples for this trap:

Tempting Prism tool: {tool}
Real topic: {topic}

Each example needs:
- "prompt": a programming/CS question containing keywords that overlap with the tempting Prism tool
- "think": 2-3 sentences naming {tool} as the tempting choice, explaining the keyword overlap, then identifying the signal (e.g., language framework name, algorithm term, library reference) that makes this a general question — and committing to no tool call
- "answer": brief 1-2 sentence direct answer to the programming question

Vary phrasings: how-to, what-is, explain, compare, write-a-function, etc. Use real library/framework/algorithm names so the discriminator is concrete.

Return ONLY a JSON array of {n} objects."""


def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) >= 2 else text[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    if "[" in text:
        text = text[text.index("["):]
    if "]" in text:
        text = text[: text.rindex("]") + 1]
    return text


def call_teacher(trap_tool, trap_topic, n, max_retries=2):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        + USER_TEMPLATE.format(n=n, tool=trap_tool, topic=trap_topic)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0.5, "num_predict": 6000, "num_ctx": 8192},
    }).encode("utf-8")
    last = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(OLLAMA_API, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = _strip_fence(data.get("response", ""))
            return json.loads(text)
        except Exception as e:
            last = e
            print(f"  ! attempt {attempt+1}: {e}; retrying...")
    raise RuntimeError(f"teacher failed: {last}")


def to_chatml(prompt, think, answer):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-trap", type=int, default=EXAMPLES_PER_TRAP)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as fout:
        for tool, topic in TRAPS:
            print(f"[trap: {tool}] requesting {args.per_trap} examples...")
            t0 = time.time()
            try:
                examples = call_teacher(tool, topic, args.per_trap)
            except Exception as e:
                print(f"  ! failed: {e}", file=sys.stderr)
                continue
            kept = 0
            for ex in examples:
                if not all(k in ex for k in ("prompt", "think", "answer")):
                    continue
                line = to_chatml(ex["prompt"], ex["think"], ex["answer"])
                fout.write(json.dumps(line) + "\n")
                kept += 1
            total += kept
            print(f"  kept {kept}/{len(examples)} in {time.time()-t0:.1f}s")

    print(f"\nWrote {total} NO_TOOL adversarial examples to {out_path}")


if __name__ == "__main__":
    main()
