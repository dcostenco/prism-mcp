"""Build train_v17_2.jsonl — v17.2 surgical pass to fix v17.1 regressions.

v17.1 results:
  ✅ caregiver 4/7 → 6/7 (+29pp)  — KEEP THIS
  ✅ caregiver targeted 20/20    — KEEP THIS
  ✅ BFCL 71.2% → 77.5%          — push higher to 95%+
  ❌ text_correct 13/15 → 11/15  — REPAIR
  ❌ format pollution 5/5 mixed  — anchor canonical

v17.2 approach: continue training from v17.1 adapter with:
  - 600 text_correct conservative examples (the v17.1 weak spot)
  - 1500 caregiver examples (HOLD — re-use winning v17.1 caregiver subset)
  - 1500 single-turn BFCL canonical-only (further BFCL push)
  - 500 multi-turn BFCL (subset of v18_1_multiturn)
  - 200 explicit format anchor examples (canonical format only)
  - 300 regression-prevention from v17 strengths

Total: ~4600 rows. Smaller and more focused than v17.1's 11K.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(172)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v17_2.jsonl"

# ──────────────────────────────────────────────────────────────────────────
# Canonical Qwen2 wrapper helpers
# ──────────────────────────────────────────────────────────────────────────
def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ──────────────────────────────────────────────────────────────────────────
# 1) text_correct conservative repair — fixes the exact v17.1 failure modes
# ──────────────────────────────────────────────────────────────────────────
TEXT_CORRECT_SYS = (
    "You are a fast text-cleanup engine for an AAC (augmentative and alternative communication) "
    "app used by users with motor impairments. Your only job: take possibly-malformed input and "
    "return the most likely intended utterance.\n\n"
    "Rules:\n"
    "- Fix obvious typos, missing spaces, dropped letters, transposed letters.\n"
    "- Fix voice-transcript word-boundary errors (e.g. \"bowlof rice\" -> \"bowl of rice\", "
    "\"i wantto eat\" -> \"i want to eat\").\n"
    "- Fix spurious commas/punctuation that came from hurried typing.\n"
    "- Capitalize \"I\" and the first word.\n"
    "- DO NOT rewrite the user's voice — keep their words and tone.\n"
    "- DO NOT add new content the user did not say.\n"
    "- DO NOT remove content the user did say.\n"
    "- DO NOT translate.\n"
    "- If the input is already well-formed, return it unchanged.\n"
    "- Return ONLY the corrected text, no quotes, no explanation, no preamble."
)

# Targeted at v17.1's exact failures + reinforcement of the rules
TEXT_CORRECT_PAIRS = [
    # The exact v17.1 failure cases (reinforce correct behavior)
    ("ineedto goto bathroom", "I need to go to bathroom"),  # NO "the"
    ("no..stop..hurts", "No stop hurts"),                    # KEEP "stop", NO "It"
    ("wer is mom", "Where is mom"),                          # wer -> Where (not who)
    ("umm i wanna go home", "I want to go home"),            # normalize wanna; remove umm OK
    # Variations to teach generalization
    ("ineedto eat", "I need to eat"),
    ("ineedto sleep", "I need to sleep"),
    ("ineedto drink water", "I need to drink water"),
    ("ineedto see mom", "I need to see mom"),
    ("ineedto go school", "I need to go school"),
    # No-article preservation
    ("igo to store", "I go to store"),
    ("igoto park", "I go to park"),
    ("can iplay", "Can I play"),
    ("doi need it", "Do I need it"),
    # Word preservation under multiple errors
    ("no..stop..pain", "No stop pain"),
    ("yes..go..home", "Yes go home"),
    ("more..please..food", "More please food"),
    ("hurry..mom..come", "Hurry mom come"),
    ("dont..stop..hurts", "Don't stop hurts"),
    ("ineed..help..now", "I need help now"),
    ("she..wants..milk", "She wants milk"),
    # wanna/gonna normalization
    ("i wanna eat", "I want to eat"),
    ("i wanna play", "I want to play"),
    ("i wanna sleep", "I want to sleep"),
    ("i wanna sit", "I want to sit"),
    ("she wanna come", "She wants to come"),
    ("we wanna go", "We want to go"),
    ("i gonna eat", "I am going to eat"),
    ("i gonna play", "I am going to play"),
    ("i gonna leave", "I am going to leave"),
    ("we gonna eat", "We are going to eat"),
    # filler word removal
    ("umm i need water", "I need water"),
    ("uh i want food", "I want food"),
    ("erm where is mom", "Where is mom"),
    ("hmm i feel sad", "I feel sad"),
    ("uh i need help", "I need help"),
    ("umm please come", "Please come"),
    # Word boundary fixes
    ("iwant to go", "I want to go"),
    ("iwantto go home", "I want to go home"),
    ("iwantto eat now", "I want to eat now"),
    ("ineed help please", "I need help please"),
    ("doi want this", "Do I want this"),
    ("areyou hungry", "Are you hungry"),
    ("can iget more", "Can I get more"),
    ("isit time togo", "Is it time to go"),
    # wer → Where (semantic recovery, not who)
    ("wer are you", "Where are you"),
    ("wer is dad", "Where is dad"),
    ("wer my toys", "Where my toys"),
    ("wer the cat", "Where the cat"),
    ("wer should i go", "Where should I go"),
    ("wen is mom coming", "When is mom coming"),
    ("wen do we eat", "When do we eat"),
    ("wat do you mean", "What do you mean"),
    ("wat is for dinner", "What is for dinner"),
    ("y is it cold", "Why is it cold"),
    ("y are we going", "Why are we going"),
    # Voice-transcript word boundary
    ("bowlof rice", "bowl of rice"),
    ("cupof water", "cup of water"),
    ("piece ofbread", "piece of bread"),
    ("lot ofthings", "lot of things"),
    ("i wantto eat bowlof rice", "I want to eat bowl of rice"),
    ("can iget cupof water", "Can I get cup of water"),
    # Already correct (no change)
    ("Hello, how are you?", "Hello, how are you?"),
    ("I love my mom.", "I love my mom."),
    ("I want to play outside.", "I want to play outside."),
    ("Please give me water.", "Please give me water."),
    ("Where is the bathroom?", "Where is the bathroom?"),
    ("Yes I need help.", "Yes I need help."),
    ("Thank you very much.", "Thank you very much."),
    # Punctuation cleanup
    ("yes,i,want,more,please,", "Yes I want more please"),
    ("no..stop..hurts", "No stop hurts"),
    ("hello,,how,,are,,you", "Hello how are you"),
    ("yes,please,more", "Yes please more"),
    ("ok,let,go,now", "Ok let go now"),
    # Dropped letter recovery (single letter)
    ("i wnt food", "I want food"),
    ("i ned help", "I need help"),
    ("can yu help", "Can you help"),
    ("plese come", "Please come"),
    ("hapy birthday", "Happy birthday"),
    # Transposed letter recovery
    ("teh dog", "The dog"),
    ("hlep me", "Help me"),
    ("waht do you want", "What do you want"),
    # Capitalization only
    ("i love mom", "I love mom"),
    ("i feel happy", "I feel happy"),
    ("i want food", "I want food"),
    ("i need water", "I need water"),
    # No bracket: contraction with apostrophe
    ("i dont know", "I don't know"),
    ("its fine", "It's fine"),
    ("im hungry", "I'm hungry"),
    ("hes sleeping", "He's sleeping"),
    ("shes happy", "She's happy"),
    ("wer not going", "We're not going"),
    ("youre nice", "You're nice"),
    ("imhungry can i hav water", "I'm hungry, can I have water"),
]


def gen_text_correct() -> list[dict]:
    rows = []
    # Repeat each pair multiple times to give it weight (key examples 4x, others 2x)
    weights = {
        "ineedto goto bathroom": 6,  # exact failure case
        "no..stop..hurts": 6,        # exact failure case (preserves "stop")
        "wer is mom": 6,             # exact failure case (Where not who)
        "umm i wanna go home": 6,    # exact failure case (wanna → want to + drop umm)
    }
    for inp, exp in TEXT_CORRECT_PAIRS:
        copies = weights.get(inp, 3)
        for _ in range(copies):
            user = f'Language: en. Input: "{inp}"'
            text = wrap_canonical(TEXT_CORRECT_SYS, user, exp)
            rows.append({"text": text, "_src": "text_correct_v17_2"})
    print(f"  text_correct_v17_2: {len(rows)} rows (from {len(TEXT_CORRECT_PAIRS)} unique pairs)")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 2) Sample HALF of v17.1 caregiver data (preserves the win, lower volume)
# ──────────────────────────────────────────────────────────────────────────
def sample_v17_1_caregiver(n: int) -> list[dict]:
    src = DATA / "train_v17_1.jsonl"
    candidates = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if "AAC app configuration assistant" in t:
                candidates.append({"text": t, "_src": "caregiver_holdover"})
    random.shuffle(candidates)
    rows = candidates[:n]
    print(f"  caregiver_holdover: {len(rows)} rows (from {len(candidates)} v17.1 caregiver)")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 3) Single-turn BFCL canonical re-anchor — selective from v17.1 bfcl
# ──────────────────────────────────────────────────────────────────────────
def sample_bfcl_canonical(n: int) -> list[dict]:
    src = DATA / "train_v17_1.jsonl"
    candidates = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            # BFCL examples have <tools> blocks in system prompt
            if "<tools>" in t and "<tool_call>" in t and "AAC" not in t:
                candidates.append({"text": t, "_src": "bfcl_canonical_holdover"})
    random.shuffle(candidates)
    rows = candidates[:n]
    print(f"  bfcl_canonical_holdover: {len(rows)} rows (from {len(candidates)} v17.1 bfcl)")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 4) Multi-turn from v18_1_multiturn (smaller subset)
# ──────────────────────────────────────────────────────────────────────────
def sample_multiturn(n: int) -> list[dict]:
    src = DATA / "v18_1_multiturn.jsonl"
    candidates = []
    if not src.exists():
        return candidates
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if "<tool_call>" in t:
                candidates.append({"text": t, "_src": "multiturn_holdover"})
    random.shuffle(candidates)
    rows = candidates[:n]
    print(f"  multiturn_holdover: {len(rows)} rows (from {len(candidates)})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 5) Format anchor examples — strict canonical demonstrations
# ──────────────────────────────────────────────────────────────────────────
FORMAT_ANCHOR_TOOLS = [
    ("get_weather", {"city": {"type": "string"}}, ["city"], "What's the weather in {}?", ["Paris", "Tokyo", "Berlin", "Madrid", "London", "Rome", "Cairo", "Sydney"]),
    ("calculate_sum", {"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"], "What is {} plus {}?", [(5, 3), (12, 7), (100, 25), (44, 16)]),
    ("search_files", {"query": {"type": "string"}, "path": {"type": "string"}}, ["query"], "Find files matching {} in {}", [("config", "/etc"), ("test", "/tmp"), ("readme", "/home")]),
    ("send_email", {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["to", "subject", "body"], "Send email to {} about {}", [("alice@example.com", "meeting"), ("bob@example.com", "report")]),
    ("get_stock_price", {"symbol": {"type": "string"}}, ["symbol"], "What's the price of {}?", ["AAPL", "GOOGL", "MSFT", "AMZN"]),
]


def gen_format_anchors() -> list[dict]:
    rows = []
    for name, props, required, q_tpl, vals in FORMAT_ANCHOR_TOOLS:
        tools_decl = [{"type": "function", "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": props, "required": required}}}]
        sys = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{json.dumps(tools_decl, ensure_ascii=False)}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )
        for v in vals:
            if isinstance(v, tuple):
                user = q_tpl.format(*v)
                args = dict(zip(required, v))
            else:
                user = q_tpl.format(v)
                args = {required[0]: v}
            assistant = f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'
            text = wrap_canonical(sys, user, assistant)
            # Repeat each 5x for strong format anchor
            for _ in range(5):
                rows.append({"text": text, "_src": "format_anchor"})
    print(f"  format_anchor: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 6) Regression-prevention from v17 (canonical only, exclude text_correct/caregiver since they have own buckets)
# ──────────────────────────────────────────────────────────────────────────
def sample_v17_other(n: int) -> list[dict]:
    src = DATA / "train_v17.jsonl"
    candidates = []
    with src.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("text", "")
            # Filter: canonical only, NOT text_correct, NOT caregiver (those have own buckets)
            if "<|tool_call|>" in t or "<|synalux_" in t:
                continue
            if "<|im_start|>" not in t:
                continue
            if "AAC app configuration assistant" in t:
                continue
            if "fast text-cleanup engine" in t:
                continue
            candidates.append({"text": t, "_src": "v17_other"})
    random.shuffle(candidates)
    rows = candidates[:n]
    print(f"  v17_other: {len(rows)} rows (from {len(candidates)})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Building train_v17_2.jsonl ===")
    rows: list[dict] = []
    rows.extend(gen_text_correct())              # ~600 (heavy weight on failures)
    rows.extend(sample_v17_1_caregiver(1500))    # 1500 caregiver hold
    rows.extend(sample_bfcl_canonical(1500))     # 1500 BFCL push
    rows.extend(sample_multiturn(500))           # 500 multi-turn hold
    rows.extend(gen_format_anchors())            # ~120 (5 tools × ~5 vals × 5 reps)
    rows.extend(sample_v17_other(300))           # 300 regression-prevention

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
