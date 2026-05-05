"""Build Phase 1.5 polish dataset for 32B/14B/7B top-tier BFCL push.

Targets the 3 weak categories:
  1. Multi-Turn (30% of Overall) — 1,600 rows from data/bfcl/multiturn_train.jsonl
  2. Live Irrelevance (10% Overall, currently 49% on 7B) — 2,000 rows from
     xlam-irrelevance + bfcl/irrelevance_train (handler SYSTEM prompt fix
     covers 50% of failures; this trains the model itself for the other half)
  3. Polyglot Java/JS — 1,053 rows from build_polyglot_polish.py output

Total: ~4,653 rows.

Conversion: Strip Synalux custom tokens (<|synalux_think|>, etc.) — render
plain Qwen <|im_start|>...<|im_end|> ChatML compatible with Qwen2.5-Coder
tokenizer (no special-token addition needed).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v18coder_polish_v1_5.jsonl"

# Synalux custom tokens to strip in conversion
SYNALUX_TOKENS = [
    "<|synalux_think|>", "</|synalux_think|>",
    "<|synalux_answer|>", "</|synalux_answer|>",
    "<|memory_query|>", "</|memory_query|>",
    "<|tool_response|>", "</|tool_response|>",
]


def strip_synalux(text: str) -> str:
    """Convert Synalux custom-token content to plain Qwen-compatible text.

    Two passes:
      1. Pair-replace: unwrap content of paired tokens (synalux_think drops,
         synalux_answer / memory_query unwrap to inner content).
      2. Tail-clean: any remaining bare `<|synalux_*|>` / `<|memory_*|>`
         tokens that appear in literal system-prompt text (e.g. "Use
         <|synalux_think|> for reasoning") get dropped entirely.

    <|tool_call|> / <|tool_response|> are renamed to <tool_call> / <tool_response>
    (Qwen-recognized BFCL form).
    """
    # 1. Pair-replace
    text = re.sub(r"<\|synalux_think\|>.*?</\|synalux_think\|>",
                  "", text, flags=re.DOTALL)
    text = re.sub(r"<\|synalux_answer\|>(.*?)</\|synalux_answer\|>",
                  r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<\|memory_query\|>(.*?)</\|memory_query\|>",
                  r"\1", text, flags=re.DOTALL)

    # 2. Rename tool tokens to standard form (Qwen2.5-Coder recognizes <tool_call>)
    text = text.replace("<|tool_call|>", "<tool_call>")
    text = text.replace("</|tool_call|>", "</tool_call>")
    text = text.replace("<|tool_response|>", "<tool_response>")
    text = text.replace("</|tool_response|>", "</tool_response>")

    # 3. Tail-clean — any leftover bare custom tokens (literal references in
    #    system-prompt instructional text). Drop them so the model isn't
    #    trained to emit them.
    text = re.sub(r"</?\|synalux_[a-z_]+\|>", "", text)
    text = re.sub(r"</?\|memory_[a-z_]+\|>", "", text)

    # 4. Tidy whitespace introduced by stripping
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_messages(messages: list[dict]) -> str:
    """Render a list of {role, content} messages as Qwen <|im_start|>...<|im_end|>."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = strip_synalux(m.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def load_messages_jsonl(path: Path, limit: int | None = None) -> list[list[dict]]:
    """Load JSONL where each line has {"messages": [...]}."""
    out = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages")
                if isinstance(messages, list) and messages:
                    out.append(messages)
                    if limit and len(out) >= limit:
                        break
            except json.JSONDecodeError:
                continue
    return out


def load_xlam_irrelevance(path: Path, limit: int) -> list[list[dict]]:
    """xLAM irrelevance set — convert to messages format."""
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    out = []
    for entry in data[:limit] if limit else data:
        # xLAM format: {query, tools, answers}
        query = entry.get("query") or entry.get("instruction") or ""
        tools = entry.get("tools") or []
        # For irrelevance, the correct response is to abstain
        if not query:
            continue
        # Build a system prompt with the tools
        sys_content = (
            "You are an expert in composing functions. You are given a question and a set of "
            "possible functions.\n\n"
            "<tools>\n" + json.dumps(tools, ensure_ascii=False) + "\n</tools>\n\n"
            "Rules:\n"
            "1. ABSOLUTE TOOL CONSTRAINT: Only call functions whose name appears verbatim "
            "in <tools>. NEVER invent function names.\n"
            "2. If no tool applies to the user's query, respond in plain text — do NOT emit "
            "a <tool_call> tag.\n"
        )
        # Assistant abstains
        abstain = (
            "None of the listed functions match this request. I cannot help with this query "
            "using the available tools."
        )
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": abstain},
        ]
        out.append(messages)
    return out


def main():
    random.seed(42)
    rendered: list[str] = []

    # 1. Multi-Turn (1,600 rows from bfcl)
    mt = load_messages_jsonl(DATA / "bfcl" / "multiturn_train.jsonl")
    print(f"multi-turn loaded: {len(mt)}")
    for messages in mt:
        text = render_messages(messages)
        if 200 < len(text) < 12000:
            rendered.append(text)

    # 2. Irrelevance — bfcl source (1000) + xLAM (sample 1000 from 7500)
    irr_a = load_messages_jsonl(DATA / "bfcl" / "irrelevance_train.jsonl")
    print(f"bfcl irrelevance loaded: {len(irr_a)}")
    irr_b = load_xlam_irrelevance(DATA / "external" / "xlam-irrelevance" /
                                   "xlam-7.5k-irrelevancek.json", limit=1000)
    print(f"xlam irrelevance loaded: {len(irr_b)}")
    for messages in irr_a + irr_b:
        text = render_messages(messages)
        if 100 < len(text) < 8000:
            rendered.append(text)

    # 3. Polyglot polish — already-built Qwen-formatted dataset
    polyglot_path = DATA / "train_polyglot_polish.jsonl"
    if polyglot_path.exists():
        with polyglot_path.open() as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if 200 < len(text) < 8000:
                        rendered.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"polyglot loaded: see prior step (~1053)")

    random.shuffle(rendered)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for text in rendered:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    n = len(rendered)
    avg_chars = sum(len(t) for t in rendered) // max(1, n)
    print(f"\ntotal: {n} rows -> {OUT}")
    print(f"avg chars: {avg_chars:,}")
    print(f"approx tokens: {sum(len(t) for t in rendered) // 4:,}")


if __name__ == "__main__":
    main()
