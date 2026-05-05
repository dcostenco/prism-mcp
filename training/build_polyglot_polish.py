"""Build a polyglot polish dataset to fix Java/JS BFCL weakness.

Sources:
  - data/v16_gen_72b/bfcl_simple_java.jsonl       (434 rows)
  - data/v16_gen_72b/bfcl_simple_javascript.jsonl (419 rows)
  - data/v16_gen_72b/bfcl_simple_python.jsonl     (if exists, for balance)

Output: data/train_polyglot_polish.jsonl in Qwen ChatML format ready for SFT.

Format target — each row:
  {
    "text": "<|im_start|>system\\n{system}\\n<|im_end|>\\n<|im_start|>user\\n{u}<|im_end|>\\n<|im_start|>assistant\\n<tool_call>\\n{json}\\n</tool_call><|im_end|>"
  }
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC_JAVA = Path("/Users/admin/prism/training/data/v16_gen_72b/bfcl_simple_java.jsonl")
SRC_JS   = Path("/Users/admin/prism/training/data/v16_gen_72b/bfcl_simple_javascript.jsonl")
SRC_PY   = Path("/Users/admin/prism/training/data/v16_gen_72b/bfcl_simple_python.jsonl")
OUT      = Path("/Users/admin/prism/training/data/train_polyglot_polish.jsonl")

SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
    "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, "
    "also point it out. You should only return the function call in tools call sections.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "<tool_call>{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}</tool_call>\n"
    "You SHOULD NOT include any other text in the response."
)


def render_tools_block(tools: list[dict]) -> str:
    """Render the tool list inside <tools>...</tools> XML."""
    lines = ["# Tools\n\nYou may call one or more functions to assist with the user query."]
    lines.append("\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
    for t in tools:
        f = t.get("function", t)
        lines.append("\n" + json.dumps(f, ensure_ascii=False))
    lines.append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
    return "\n".join(lines)


def row_to_chatml(row: dict) -> str | None:
    user = row.get("user")
    call = row.get("call")
    tools = row.get("tools") or []
    if not user or not call or not tools:
        return None

    system = SYSTEM_PROMPT + "\n\n" + render_tools_block(tools)
    assistant = (
        "<tool_call>\n"
        + json.dumps({"name": call.get("name", ""), "arguments": call.get("arguments", {})}, ensure_ascii=False)
        + "\n</tool_call>"
    )
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return text


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    random.seed(42)

    java = load_jsonl(SRC_JAVA)
    js = load_jsonl(SRC_JS)
    py = load_jsonl(SRC_PY)

    print(f"java: {len(java)}, js: {len(js)}, python: {len(py)}")

    # Polish mix: ALL Java/JS + ~200 Python for balance (catastrophic-forgetting guard)
    py_subset = random.sample(py, min(200, len(py))) if py else []

    rows: list[str] = []
    for src, name in [(java, "java"), (js, "js"), (py_subset, "python")]:
        kept = 0
        for r in src:
            text = row_to_chatml(r)
            if text and 200 < len(text) < 8000:
                rows.append(text)
                kept += 1
        print(f"  {name}: {kept} rendered")

    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for text in rows:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"\ntotal: {len(rows)} -> {OUT}")
    print(f"avg chars: {sum(len(t) for t in rows) // max(1, len(rows)):,}")


if __name__ == "__main__":
    main()
