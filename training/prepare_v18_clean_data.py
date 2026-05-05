"""Build train_v18_clean.jsonl — full SFT from clean Qwen base, dual-objective.

Goal: BFCL ≥95% AND all AAC gates pass, in a SINGLE model.

Strategy: do NOT continue from v17.x adapter chain (carries legacy format
patterns that resist scrubbing). Instead train Qwen2.5-Coder-7B-Instruct
fresh on a balanced dataset. The 7B base is strong; the question is whether
balanced data + full SFT (not surgical) breaks past v17.2's 82.2% Pareto.

Composition (~7900 rows):
  5450 BFCL gen (canonical Qwen2 <tool_call>) — restore v12-class coding
  1500 caregiver (from v17.2 winning subset) — preserve BCBA win
   300 text_correct (from v17.2, the perfect 15/15)
   200 emergency Q&A (from v17.4 generated)
   100 ask_ai (from v17.2)
   100 translate (variations)
   200 format anchor (canonical wrapper bias)
    50 abstention (joke / greeting → text response)

NO LEGACY <|tool_call|> rows. NO v18_1_multiturn (poisoned multi-turn).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(180)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v18_clean.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def convert_custom_to_canonical(text: str) -> str:
    text = re.sub(r"<\|tool_call\|>", "<tool_call>", text)
    text = re.sub(r"</\|tool_call\|>", "</tool_call>", text)
    text = re.sub(r"<\|tool_call_end\|>", "</tool_call>", text)
    text = re.sub(r"<\|synalux_think\|>", "<think>", text)
    text = re.sub(r"</\|synalux_think\|>", "</think>", text)
    text = re.sub(r"<\|synalux_answer\|>", "", text)
    text = re.sub(r"</\|synalux_answer\|>", "", text)
    return text


# ──────────────────────────────────────────────────────────────────────────
# 1) BFCL gen — convert custom format to canonical
# ──────────────────────────────────────────────────────────────────────────
def load_bfcl_gen() -> list[dict]:
    rows = []
    src = DATA / "v17_1_bfcl" / "train.jsonl"
    if not src.exists():
        print(f"WARN: {src} missing")
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
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = convert_custom_to_canonical(m.get("content", ""))
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(parts)
            rows.append({"text": text, "_src": "bfcl_gen"})
    print(f"  bfcl_gen: {len(rows)} rows")
    return rows


def sample_v17_2_subset(predicate, n: int, label: str, src_name: str = "train_v17_2.jsonl") -> list[dict]:
    src = DATA / src_name
    cands = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if predicate(t):
                cands.append({"text": t, "_src": label})
    random.shuffle(cands)
    rows = cands[:n]
    print(f"  {label}: {len(rows)} rows (from {len(cands)} candidates in {src_name})")
    return rows


def main() -> None:
    print("=== Building train_v18_clean.jsonl ===")
    rows: list[dict] = []
    rows.extend(load_bfcl_gen())
    rows.extend(sample_v17_2_subset(lambda t: "AAC app configuration assistant" in t, 1500, "caregiver"))
    rows.extend(sample_v17_2_subset(lambda t: "fast text-cleanup engine" in t, 300, "text_correct"))
    # Emergency from v17.4 if exists
    if (DATA / "train_v17_4.jsonl").exists():
        rows.extend(sample_v17_2_subset(lambda t: "emergency-response AI" in t, 250, "emergency", "train_v17_4.jsonl"))
    rows.extend(sample_v17_2_subset(lambda t: "friendly helper for a child" in t, 100, "ask_ai"))
    rows.extend(sample_v17_2_subset(lambda t: "translator" in t.lower() and "translate the input" in t.lower(), 100, "translate"))
    # Format anchor + abstention from v17.3 if exists
    if (DATA / "train_v17_3.jsonl").exists():
        rows.extend(sample_v17_2_subset(lambda t: "format_anchor" in t or ("get_weather" in t and "<tool_call>" in t), 200, "format_anchor", "train_v17_3.jsonl"))

    random.shuffle(rows)
    print(f"\n=== Total: {len(rows)} rows ===")
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r["_src"]] = src_counts.get(r["_src"], 0) + 1
    for k, v in sorted(src_counts.items()):
        print(f"  {k}: {v}")

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
