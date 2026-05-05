"""Convert train_v17_1.jsonl from Qwen2.5-Coder rendered format to Qwen3-rendered format.

The original file contains pre-rendered text with <|im_start|>...<|im_end|> tokens
matching Qwen2.5-Coder's chat template. Qwen3-8B uses the same base tokens but a
different tools-wrapping convention and supports an optional thinking block.

We parse each row back into structured [{role, content}] messages, then re-render
using Qwen3-8B's tokenizer.apply_chat_template so polishing on Qwen3-8B sees the
format it was pretrained on.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TURN_RE = re.compile(
    re.escape(IM_START)
    + r"(system|user|assistant|tool)\n(.*?)"
    + re.escape(IM_END),
    re.DOTALL,
)


def parse_rendered(text: str) -> list[dict]:
    """Parse a rendered Qwen-style chat string into [{role, content}, ...]."""
    messages = []
    for m in TURN_RE.finditer(text):
        role, content = m.group(1), m.group(2)
        messages.append({"role": role, "content": content.strip()})
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl with rendered text")
    ap.add_argument("--out", required=True, help="output jsonl with Qwen3-rendered text")
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="HF model id for tokenizer")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"loading tokenizer for {args.base}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    n_in = n_out = n_skip_no_messages = n_skip_no_assistant = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")
            if not isinstance(text, str):
                continue

            messages = parse_rendered(text)
            if not messages:
                n_skip_no_messages += 1
                continue
            if not any(m["role"] == "assistant" for m in messages):
                n_skip_no_assistant += 1
                continue

            try:
                rendered = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

            fout.write(json.dumps({"text": rendered}) + "\n")
            n_out += 1
            if args.limit and n_out >= args.limit:
                break

    print(
        f"done: {n_out}/{n_in} rows written. "
        f"skipped: no_messages={n_skip_no_messages}, no_assistant={n_skip_no_assistant}",
        flush=True,
    )


if __name__ == "__main__":
    main()
