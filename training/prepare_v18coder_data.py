"""Build train_v18coder.jsonl — commercial-safe BFCL-only mix for prism-coder:7b-coder.

Strategy: Hammer-style SFT recipe with permissive licenses (Apache 2.0 / CC-BY-4.0)
since prism-coder has paid tiers — NC datasets are off-limits.

Composition:
  glaive-v2 (112,960 single-turn, Apache 2.0) — base BFCL volume
  ToolACE (11,300 multi-turn, Apache 2.0) — multi-turn quality signal
  v17_1_bfcl (12,712, internal) — Prism tool coverage (canonicalized)
  TOTAL: ~136K commercial-safe examples

NO AAC data — this is the coder model. AAC stays separate (prism-coder:7b-aac).
NO NC data (xlam-irrelevance, APIGen-MT skipped due to license).

Output: data/train_v18coder.jsonl (canonical Qwen2 chat-format text field).

Function-masking augmentation: 30% of rows have function names replaced with
generic placeholders (func_001, func_002, ...) — Hammer's recipe to force the
model to bind by description rather than name patterns.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

random.seed(18000)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v18coder.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def messages_to_canonical(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role") or m.get("from") or "user"
        # Map ShareGPT-style roles to canonical
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        elif role == "function-response":
            role = "tool"
        content = m.get("content") or m.get("value") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Hammer-style function masking (arXiv:2410.04587)
# ──────────────────────────────────────────────────────────────────────────
def mask_function_names(text: str, mask_prob: float = 0.30) -> str:
    """If we hit mask_prob, replace function names with generic placeholders.

    Forces the model to bind on `description` and parameter shape, not on
    function name pattern recognition. Original Hammer paper achieves +1-2pp
    on Non-Live AST and +5pp on Irrelevance with 20-40% masking rate.
    """
    if random.random() > mask_prob:
        return text

    # Find all unique function names from <tools>...</tools> declarations
    # Schema: {"type": "function", "function": {"name": "<NAME>", ...}}
    name_matches = re.findall(r'"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]+)"', text)
    if not name_matches:
        return text  # no functions to mask

    # Build mapping
    unique = list(dict.fromkeys(name_matches))  # preserve order, dedup
    if len(unique) > 50:
        return text  # very long tools block — skip to avoid pathological cases
    mapping = {name: f"func_{i:03d}" for i, name in enumerate(unique, start=1)}

    # Apply mapping wherever the name appears as identifier (avoid partial matches in english)
    for orig, new in mapping.items():
        text = re.sub(rf'\b{re.escape(orig)}\b', new, text)
    return text


# ──────────────────────────────────────────────────────────────────────────
# 1) glaive-function-calling-v2 — single-turn, ~112K examples
#    Format: {"system": "SYSTEM: ...", "chat": "USER: ...\n\n\nASSISTANT: ..."}
# ──────────────────────────────────────────────────────────────────────────
def parse_glaive_chat(chat: str) -> list[dict]:
    """Parse glaive-v2 chat format into role-content message list.

    glaive-v2 separators (verified by inspection):
      USER:           — user message starts
      ASSISTANT:      — assistant text response
      FUNCTION CALL:  — assistant function call (usually JSON)
      FUNCTION RESPONSE:  — tool response
    """
    messages = []
    # Split on common markers, keeping markers
    pattern = r'(USER:|ASSISTANT:|FUNCTION CALL:|FUNCTION RESPONSE:)'
    parts = re.split(pattern, chat)
    # parts will be: [pre, marker, content, marker, content, ...]
    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not content:
            i += 2
            continue
        if marker == "USER:":
            messages.append({"role": "user", "content": content})
        elif marker == "ASSISTANT:":
            # Strip trailing <|endoftext|> etc.
            content = re.sub(r'<\|endoftext\|>\s*$', '', content).strip()
            messages.append({"role": "assistant", "content": content})
        elif marker == "FUNCTION CALL:":
            # Wrap as canonical <tool_call>
            content = re.sub(r'<\|endoftext\|>\s*$', '', content).strip()
            wrapped = f"<tool_call>\n{content}\n</tool_call>"
            messages.append({"role": "assistant", "content": wrapped})
        elif marker == "FUNCTION RESPONSE:":
            wrapped = f"<tool_response>\n{content}\n</tool_response>"
            messages.append({"role": "tool", "content": wrapped})
        i += 2
    return messages


def glaive_system_to_canonical(system: str) -> str:
    """Convert glaive system prompt to canonical Qwen2 system + tools format."""
    # glaive format: "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{...function spec...}"
    # Extract function specs (everything after the SYSTEM: header text)
    m = re.search(r'SYSTEM:\s*(.+?)Use them if required\s*-?\s*(.+)', system, re.DOTALL)
    if m:
        intro = m.group(1).strip()
        funcs_raw = m.group(2).strip()
    else:
        # fallback: assume entire content is functions
        intro = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        funcs_raw = re.sub(r'^SYSTEM:\s*', '', system).strip()

    # Try to parse functions as JSON (may be a single object or list)
    funcs = []
    funcs_clean = re.sub(r',\s*}\s*}', '}}', funcs_raw)  # trailing commas
    # Find all top-level JSON objects via brace-matching
    depth = 0
    start = -1
    for i, c in enumerate(funcs_clean):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                blob = funcs_clean[start:i + 1]
                try:
                    obj = json.loads(blob)
                    funcs.append(obj)
                except Exception:
                    pass
                start = -1

    # Wrap each function in canonical Qwen2 tools format
    wrapped_funcs = [{"type": "function", "function": f} for f in funcs] if funcs else []
    tools_json = json.dumps(wrapped_funcs, ensure_ascii=False) if wrapped_funcs else "[]"

    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags."
    )


def load_glaive() -> Iterable[dict]:
    src = DATA / "external" / "glaive-v2" / "glaive-function-calling-v2.json"
    if not src.exists():
        print(f"WARN: {src} not found")
        return
    with src.open() as f:
        data = json.load(f)
    print(f"  glaive-v2: {len(data)} raw entries — parsing...")
    parsed = 0
    skipped = 0
    for entry in data:
        sys_canonical = glaive_system_to_canonical(entry.get("system", ""))
        msgs = parse_glaive_chat(entry.get("chat", ""))
        if not msgs:
            skipped += 1
            continue
        # Build single text field
        text = f"<|im_start|>system\n{sys_canonical}<|im_end|>\n" + messages_to_canonical(msgs)
        # Apply function masking probabilistically
        text = mask_function_names(text, mask_prob=0.30)
        if len(text) >= 50:
            yield {"text": text, "_src": "glaive_v2"}
            parsed += 1
    print(f"  glaive-v2: parsed {parsed}, skipped {skipped}")


# ──────────────────────────────────────────────────────────────────────────
# 2) ToolACE — multi-turn ShareGPT-style
# ──────────────────────────────────────────────────────────────────────────
def load_toolace() -> Iterable[dict]:
    src = DATA / "external" / "toolace" / "data.json"
    if not src.exists():
        print(f"WARN: {src} not found")
        return
    with src.open() as f:
        data = json.load(f)
    print(f"  ToolACE: {len(data)} raw entries — parsing...")
    parsed = 0
    skipped = 0
    for entry in data:
        sys_text = entry.get("system", "")
        convs = entry.get("conversations", [])
        if not sys_text or not convs:
            skipped += 1
            continue
        # ToolACE system prompt already includes function specs in JSON form
        # Wrap in canonical Qwen2 system block
        sys_canonical = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
            "# Tools\n\nYou may call one or more functions.\n\n"
            f"{sys_text}"
        )
        # Convert conversations to canonical messages
        # ToolACE roles: "user", "assistant", "function-response" (sometimes "tool")
        text = f"<|im_start|>system\n{sys_canonical}<|im_end|>\n" + messages_to_canonical(convs)
        text = mask_function_names(text, mask_prob=0.30)
        if len(text) >= 50:
            yield {"text": text, "_src": "toolace"}
            parsed += 1
    print(f"  ToolACE: parsed {parsed}, skipped {skipped}")


# ──────────────────────────────────────────────────────────────────────────
# 3) Our v17_1_bfcl — already canonicalized in prepare_v17_1_data.py logic
#    Reuse the conversion from prepare_v17_1_data.py
# ──────────────────────────────────────────────────────────────────────────
def load_internal_bfcl() -> Iterable[dict]:
    """Load our internal BFCL gen output (12.7K examples), convert format."""
    src = DATA / "v17_1_bfcl" / "train.jsonl"
    if not src.exists():
        print(f"WARN: {src} not found")
        return
    parsed = 0
    with src.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            # Convert custom <|tool_call|> to canonical <tool_call>
            text_parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                # Replace legacy custom format with canonical
                content = re.sub(r"<\|tool_call\|>", "<tool_call>", content)
                content = re.sub(r"</\|tool_call\|>", "</tool_call>", content)
                content = re.sub(r"<\|tool_call_end\|>", "</tool_call>", content)
                content = re.sub(r"<\|synalux_think\|>", "<think>", content)
                content = re.sub(r"</\|synalux_think\|>", "</think>", content)
                content = re.sub(r"<\|synalux_answer\|>", "", content)
                content = re.sub(r"</\|synalux_answer\|>", "", content)
                text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(text_parts)
            text = mask_function_names(text, mask_prob=0.30)
            if len(text) >= 50:
                yield {"text": text, "_src": "internal_bfcl"}
                parsed += 1
    print(f"  internal_bfcl: parsed {parsed}")


# ──────────────────────────────────────────────────────────────────────────
def load_xlam_60k() -> Iterable[dict]:
    """Load Salesforce xlam-function-calling-60k (CC-BY-4.0, commercial-safe).
    Schema: {id, query, answers, tools} where answers/tools are JSON strings."""
    src = DATA / "external" / "xlam-60k" / "xlam_function_calling_60k.json"
    if not src.exists():
        print(f"WARN: {src} not found")
        return
    with src.open() as f:
        data = json.load(f)
    print(f"  xlam-60k: {len(data)} raw entries — parsing...")
    parsed = 0
    skipped = 0
    for entry in data:
        query = entry.get("query", "")
        try:
            tools = json.loads(entry.get("tools", "[]"))
            answers = json.loads(entry.get("answers", "[]"))
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not query or not tools or not answers:
            skipped += 1
            continue

        # Wrap tools in canonical Qwen2 format
        wrapped_funcs = [{"type": "function", "function": t} for t in tools]
        sys_msg = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            f"<tools>\n{json.dumps(wrapped_funcs, ensure_ascii=False)}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags."
        )
        # Build assistant response — multiple tool calls per answer
        tool_calls = []
        for a in answers:
            tool_calls.append(f'<tool_call>\n{json.dumps(a, ensure_ascii=False)}\n</tool_call>')
        assistant = "\n".join(tool_calls)

        text = wrap_canonical(sys_msg, query, assistant)
        text = mask_function_names(text, mask_prob=0.30)
        if len(text) >= 50:
            yield {"text": text, "_src": "xlam_60k"}
            parsed += 1
    print(f"  xlam-60k: parsed {parsed}, skipped {skipped}")


def main() -> None:
    print("=== Building train_v18coder.jsonl (commercial-safe BFCL mix) ===")
    rows: list[dict] = []
    rows.extend(load_glaive())
    rows.extend(load_toolace())
    rows.extend(load_xlam_60k())
    rows.extend(load_internal_bfcl())

    random.shuffle(rows)
    print(f"\n=== Total: {len(rows)} rows ===")
    src_counts: dict[str, int] = {}
    masked_count = sum(1 for r in rows if "func_001" in r["text"])
    for r in rows:
        src_counts[r["_src"]] = src_counts.get(r["_src"], 0) + 1
    for k, v in sorted(src_counts.items()):
        print(f"  {k}: {v}")
    print(f"  function-masked rows: ~{masked_count} (~{100*masked_count/max(len(rows),1):.0f}%)")

    print("\nWriting...")
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    size_mb = OUT.stat().st_size // (1024 * 1024)
    print(f"\nwrote {OUT} ({size_mb} MB, {len(rows)} rows)")


if __name__ == "__main__":
    main()
