"""Build train_v17_1.jsonl for the v17.1 surgical SFT pass.

All output rows are canonical Qwen2 chat format with single `text` field:
  <|im_start|>system\n...<|im_end|>
  <|im_start|>user\n...<|im_end|>
  <|im_start|>assistant\n<tool_call>{"name":"...","arguments":{...}}</tool_call><|im_end|>

Composition:
  ~2500 rows  v18_1_multiturn.jsonl       (already canonical multi-turn)
  ~3500 rows  v17_1_bfcl/train.jsonl      (BFCL gen — converted from custom <|tool_call|>)
  ~1000 rows  caregiver_parse (existing) + boost_word fixes (NEW)
  ~500 rows   regression-prevention from train_v17.jsonl (canonical only)

Total: ~7500 rows.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(17)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v17_1.jsonl"

# ──────────────────────────────────────────────────────────────────────────
# 1) Canonical Qwen2 system prompt with tools block
# ──────────────────────────────────────────────────────────────────────────
QWEN2_SYS_TPL = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    """Produce canonical Qwen2 chat-format text with system+user+assistant turns."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def convert_custom_to_canonical(text: str) -> str:
    """Replace <|tool_call|>...</|tool_call|> with <tool_call>...</tool_call>.
    Also normalize <|synalux_think|> → <think> and <|synalux_answer|> wrapper away."""
    text = re.sub(r"<\|tool_call\|>", "<tool_call>", text)
    text = re.sub(r"</\|tool_call\|>", "</tool_call>", text)
    text = re.sub(r"<\|tool_call_end\|>", "</tool_call>", text)
    text = re.sub(r"<\|synalux_think\|>", "<think>", text)
    text = re.sub(r"</\|synalux_think\|>", "</think>", text)
    text = re.sub(r"<\|synalux_answer\|>", "", text)
    text = re.sub(r"</\|synalux_answer\|>", "", text)
    return text


# ──────────────────────────────────────────────────────────────────────────
# 2) Load v18_1_multiturn.jsonl (already canonical)
# ──────────────────────────────────────────────────────────────────────────
def load_v18_1_multiturn() -> list[dict]:
    rows = []
    src = DATA / "v18_1_multiturn.jsonl"
    if not src.exists():
        print(f"WARN: {src} not found")
        return rows
    with src.open() as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "")
            if "<tool_call>" in text or "<|im_start|>" in text:
                rows.append({"text": text, "_src": "v18_1_multiturn"})
    print(f"  v18_1_multiturn: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 3) Convert BFCL gen output (messages format, custom <|tool_call|>) → canonical
# ──────────────────────────────────────────────────────────────────────────
def messages_to_canonical_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = convert_custom_to_canonical(m.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def load_bfcl_gen() -> list[dict]:
    rows = []
    src = DATA / "v17_1_bfcl" / "train.jsonl"
    if not src.exists():
        print(f"WARN: {src} not found")
        return rows
    with src.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages")
            if not msgs:
                continue
            text = messages_to_canonical_text(msgs)
            cat = rec.get("category", "bfcl")
            rows.append({"text": text, "_src": f"bfcl_{cat}"})
    print(f"  bfcl_gen: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 4) Caregiver — convert existing v16_gen_72b/caregiver_parse_extra.jsonl
#    AAC eval expects bare JSON array (NOT tool_call wrapped) — match that.
# ──────────────────────────────────────────────────────────────────────────
CAREGIVER_SYS = (
    "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
    "Available categories: help: Help/Needs, food: Food & Drink, feelings: Feelings, "
    "school: School/Work, quick: Quick Talk\n\n"
    "Parse the following caregiver instruction into structured JSON actions.\n"
    "Return ONLY a JSON array of action objects. No explanation.\n\n"
    "Action types:\n"
    "  add_phrase: { categoryId, text }\n"
    "  remove_phrase: { phraseText, categoryId }\n"
    "  add_category: { name, icon }\n"
    "  add_sequence: { name, categoryId, steps }\n"
    "  reorder_phrase: { phraseId, newSortOrder, categoryId }\n"
    "  boost_word: { word, boostCount }\n"
    "  note_only: {} (for observations with no configuration change)"
)


def load_caregiver_existing() -> list[dict]:
    rows = []
    for src_name in ("v16_gen_72b/caregiver_parse_extra.jsonl",
                     "v16_gen_72b/caregiver_parse.jsonl"):
        src = DATA / src_name
        if not src.exists():
            continue
        with src.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                note = rec.get("note", "")
                actions = rec.get("actions", [])
                if not note or not actions:
                    continue
                user = f'Caregiver says: "{note}"'
                # Eval scoring expects each action to have "type" field for must_action match
                assistant = json.dumps(actions, separators=(",", ":"))
                text = wrap_canonical(CAREGIVER_SYS, user, assistant)
                rows.append({"text": text, "_src": "caregiver_existing"})
    print(f"  caregiver_existing: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 5) NEW boost_word patterns — the v17 weak spot
#    Failure case: "He's using 'because' a lot now" → emitted note_only, expected boost_word
# ──────────────────────────────────────────────────────────────────────────
WORDS = ["because", "please", "help", "more", "stop", "want", "need", "thank you",
         "no", "yes", "all done", "again", "tired", "happy", "sad", "hungry",
         "thirsty", "cold", "hot", "play", "go", "come", "look", "open"]
PHRASING_TEMPLATES = [
    "He's using '{w}' a lot now",
    "She's using '{w}' a lot now",
    "They're saying '{w}' frequently",
    "Started saying '{w}' a lot this week",
    "Really likes the word '{w}' lately",
    "Uses '{w}' often during sessions",
    "We're hearing '{w}' a lot today",
    "Has been saying '{w}' a lot",
    "Keeps saying '{w}'",
    "Says '{w}' all the time now",
    "Loves the word '{w}'",
    "Is using '{w}' more often",
    "Repeats '{w}' often",
    "{caps} is one of his favorite words now",
    "Their favorite word right now is '{w}'",
    "Spontaneously using '{w}' more",
    "Hearing more '{w}' from her",
    "Boost '{w}' please",
    "Can you boost '{w}'",
    "Let's prioritize '{w}'",
]


def gen_caregiver_boost_word(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        w = random.choice(WORDS)
        tpl = random.choice(PHRASING_TEMPLATES)
        note = tpl.format(w=w, caps=w.capitalize())
        actions = [{"type": "boost_word", "word": w, "boostCount": random.choice([1, 2, 3])}]
        user = f'Caregiver says: "{note}"'
        assistant = json.dumps(actions, separators=(",", ":"))
        text = wrap_canonical(CAREGIVER_SYS, user, assistant)
        rows.append({"text": text, "_src": "caregiver_boost_NEW"})
    print(f"  caregiver_boost_NEW: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 6) NEW reorder_phrase patterns
# ──────────────────────────────────────────────────────────────────────────
REORDER_PHRASES = [
    ("bathroom", "help"), ("water", "food"), ("hungry", "feelings"),
    ("more", "quick"), ("help", "help"), ("please", "quick"),
    ("milk", "food"), ("tired", "feelings"), ("hello", "quick"),
    ("home", "school"), ("teacher", "school"),
]
REORDER_TEMPLATES = [
    "Move '{p}' to top of {c} category",
    "Make '{p}' the first phrase in {c}",
    "Reorder '{p}' to position 1 in {c}",
    "Put '{p}' at the top of {c}",
    "Promote '{p}' to first in {c} category",
]


def gen_caregiver_reorder(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        phrase, cat = random.choice(REORDER_PHRASES)
        tpl = random.choice(REORDER_TEMPLATES)
        note = tpl.format(p=phrase, c=cat.capitalize())
        actions = [{"type": "reorder_phrase", "phraseId": phrase, "newSortOrder": 1, "categoryId": cat}]
        user = f'Caregiver says: "{note}"'
        assistant = json.dumps(actions, separators=(",", ":"))
        text = wrap_canonical(CAREGIVER_SYS, user, assistant)
        rows.append({"text": text, "_src": "caregiver_reorder_NEW"})
    print(f"  caregiver_reorder_NEW: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 7) Regression-prevention sample from train_v17.jsonl — canonical-format only
# ──────────────────────────────────────────────────────────────────────────
def load_v17_canonical_sample(n: int) -> list[dict]:
    src = DATA / "train_v17.jsonl"
    candidates = []
    with src.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            # Skip rows that use the deprecated custom format
            if "<|tool_call|>" in text or "<|synalux_" in text:
                continue
            # Keep only canonical Qwen2 chat format
            if "<|im_start|>" not in text:
                continue
            candidates.append({"text": text, "_src": "v17_regress"})
    random.shuffle(candidates)
    rows = candidates[:n]
    print(f"  v17_regress_sample: {len(rows)} rows (from {len(candidates)} canonical candidates)")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Build and write
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Building train_v17_1.jsonl ===")
    rows: list[dict] = []
    rows.extend(load_v18_1_multiturn())
    rows.extend(load_bfcl_gen())
    rows.extend(load_caregiver_existing())
    rows.extend(gen_caregiver_boost_word(400))
    rows.extend(gen_caregiver_reorder(150))
    rows.extend(load_v17_canonical_sample(500))

    random.shuffle(rows)
    print(f"\n=== Total: {len(rows)} rows ===")
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r["_src"]] = src_counts.get(r["_src"], 0) + 1
    for k, v in sorted(src_counts.items()):
        print(f"  {k}: {v}")

    with OUT.open("w") as f:
        for r in rows:
            out = {"text": r["text"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
