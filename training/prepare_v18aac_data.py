"""Build train_v18aac.jsonl — AAC-optimized dataset for Qwen3-8B base.

Goal: BEAT v17.4's emergency 13/13 + caregiver 6/7 + text_correct 15/15 + ask_ai 5/5
+ translate 7/8 from a fresh Qwen3-8B base (not the v17 chain).

Composition (~7000 rows):
  All v17_2 caregiver examples (held the win)
  All v17_4 emergency examples (the 13/13 dataset)
  All text_correct examples from v16_gen_72b + v17_2 (preserves 15/15)
  All ask_ai examples from v17_2 + v16_gen_72b
  Translate examples (small)
  Word predict examples (preserves AAC core capability)
  Format anchor examples (canonical wrapper bias)

Output: data/train_v18aac.jsonl
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(18001)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v18aac.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def sample_v17_2_subset(predicate, n: int, label: str, src_name: str = "train_v17_2.jsonl") -> list[dict]:
    src = DATA / src_name
    if not src.exists():
        print(f"WARN: {src} missing")
        return []
    cands = []
    with src.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("text", "")
            if predicate(t):
                cands.append({"text": t, "_src": label})
    random.shuffle(cands)
    rows = cands[:n]
    print(f"  {label}: {len(rows)} rows (from {len(cands)} in {src_name})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Raw v16_gen_72b loaders — bare JSON to canonical Qwen2 SFT
# ──────────────────────────────────────────────────────────────────────────
CAREGIVER_SYS = (
    "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
    "Available categories: help: Help/Needs, food: Food & Drink, feelings: Feelings, school: School/Work, quick: Quick Talk\n\n"
    "Parse the following caregiver instruction into structured JSON actions.\n"
    "Return ONLY a JSON array of action objects. No explanation.\n\n"
    "Action types:\n"
    "  add_phrase: { categoryId, text }\n"
    "  remove_phrase: { phraseText, categoryId }\n"
    "  reorder_phrase: { phraseId, newSortOrder, categoryId }\n"
    "  add_category: { name, icon }\n"
    "  add_sequence: { name, categoryId, steps }\n"
    "  boost_word: { word, boostCount }\n"
    "  note_only: {} (for observations with no configuration change)"
)


def load_caregiver_v16_raw() -> list[dict]:
    rows = []
    for fname in ("v16_gen_72b/caregiver_parse_extra.jsonl",
                  "v16_gen_72b/caregiver_parse.jsonl",
                  "v16_corrective/caregiver_parse_extra.jsonl"):
        src = DATA / fname
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
                assistant = json.dumps(actions, separators=(",", ":"))
                text = wrap_canonical(CAREGIVER_SYS, user, assistant)
                rows.append({"text": text, "_src": "caregiver_v16_raw"})
    print(f"  caregiver_v16_raw: {len(rows)} rows")
    return rows


def load_v16_gen_jsonl_as_chat(fname: str, label: str) -> list[dict]:
    """Load v16_gen_72b/* JSONL files that already have system+user+assistant structure."""
    src = DATA / fname
    if not src.exists():
        print(f"WARN: {src} missing")
        return []
    rows = []
    with src.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # v16_gen_72b files use varying schemas — try common patterns
            sys_text = rec.get("system", "")
            user = rec.get("user", "") or rec.get("question", "") or rec.get("input", "")
            assistant = rec.get("assistant", "") or rec.get("answer", "") or rec.get("response", "") or rec.get("output", "")
            if not user or not assistant:
                continue
            if not sys_text:
                # Build minimal system prompt per task type
                sys_text = "You are a helpful AAC assistant."
            text = wrap_canonical(sys_text, user, str(assistant))
            rows.append({"text": text, "_src": label})
    print(f"  {label}: {len(rows)} rows (from {fname})")
    return rows


def main() -> None:
    print("=== Building train_v18aac.jsonl (AAC-only for Qwen3-8B base) ===")
    rows: list[dict] = []

    # Caregiver — v17.2 winning subset + raw v16_gen_72b for full coverage
    rows.extend(sample_v17_2_subset(lambda t: "AAC app configuration assistant" in t, 9999, "caregiver_v17_2"))
    rows.extend(load_caregiver_v16_raw())

    # Emergency — v17.4's 13/13 dataset
    rows.extend(sample_v17_2_subset(lambda t: "emergency-response AI" in t, 9999, "emergency_v17_4", "train_v17_4.jsonl"))

    # Text correct — v17.2's perfect 15/15 dataset
    rows.extend(sample_v17_2_subset(lambda t: "fast text-cleanup engine" in t, 9999, "text_correct_v17_2"))

    # Pull more raw AAC sources from v16_gen_72b for breadth
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/text_correct.jsonl", "text_correct_v16_raw"))
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/emergency_qa.jsonl", "emergency_v16_raw"))
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/emergency_qa_extra.jsonl", "emergency_extra_v16"))
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/ask_ai_aac.jsonl", "ask_ai_v16"))
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/translate.jsonl", "translate_v16"))
    rows.extend(load_v16_gen_jsonl_as_chat("v16_gen_72b/word_predict_aac.jsonl", "word_predict_v16"))

    # Held over from v17_2/v17_3
    rows.extend(sample_v17_2_subset(lambda t: "friendly helper for a child" in t, 9999, "ask_ai_v17_2"))
    rows.extend(sample_v17_2_subset(lambda t: "translator" in t.lower() and "translate the input" in t.lower(), 9999, "translate_v17_2"))
    rows.extend(sample_v17_2_subset(lambda t: "format_anchor" in t or ("get_weather" in t and "<tool_call>" in t and "<tools>" in t), 200, "format_anchor", "train_v17_3.jsonl"))

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
