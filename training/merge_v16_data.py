#!/usr/bin/env python3
"""Merge v16 generated data into the canonical chatml train.jsonl.

Reads every category jsonl from data/v16_gen/, converts each to the
<|im_start|>...<|synalux_think|>...<|tool_call|>...<|im_end|> chatml
format the existing trainer expects, optionally concatenates the
existing data/train.jsonl (the ~2,288 v12 examples), shuffles, and
writes data/train_v16.jsonl.

Usage:
  source venv/bin/activate
  python3 merge_v16_data.py
  python3 merge_v16_data.py --no-legacy     # exclude existing train.jsonl
  python3 merge_v16_data.py --out data/train_v16_aac_only.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
GEN_DIR_72B = ROOT / "data" / "v16_gen_72b"          # primary, higher quality
GEN_DIR_7B  = ROOT / "data" / "v16_gen_7b_backup"    # fallback for category coverage
GEN_DIR_CORRECTIVE = ROOT / "data" / "v16_corrective" # v16.1 fixes for the 3 regressions
LEGACY = ROOT / "data" / "train.jsonl"
DEFAULT_OUT = ROOT / "data" / "train_v16_1.jsonl"

# Composite keys per category for cross-source dedup (72B+7B union, prefer 72B)
DEDUP_KEYS = {
    "text_correct": lambda r: (r.get("input","").lower().strip(), r.get("expected","").lower().strip()),
    "emergency_qa": lambda r: (r.get("script","")[:200], r.get("question","").lower().strip()),
    "caregiver_parse": lambda r: (r.get("note","").lower().strip(),),
    "translate": lambda r: (r.get("text","").lower().strip(), r.get("toLang","")),
    "ask_ai_aac": lambda r: (r.get("question","").lower().strip(),),
    "word_predict_aac": lambda r: (r.get("prefix","").lower().strip(),),
    "memory_checkout": lambda r: (r.get("prompt","").lower().strip(),),
    "hipaa": lambda r: (r.get("prompt","").lower().strip(),),
    "contrastive_extra": lambda r: (r.get("prompt","").lower().strip(),),
    "no_tool_extra": lambda r: (r.get("prompt","").lower().strip(),),
}


def chatml(system: str, user: str, assistant: str) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


# ── Per-category renderers — convert generator output to canonical chatml ───

def render_text_correct(o: dict) -> str | None:
    inp, lang, exp = o.get("input"), o.get("lang", "en"), o.get("expected")
    if not (inp and exp):
        return None
    sys_p = (
        "You are a fast text-cleanup engine for an AAC (augmentative and alternative communication) "
        "app used by users with motor impairments. Fix obvious typos, missing spaces, dropped letters, "
        "transposed letters. Fix voice-transcript word-boundary errors. Capitalize 'I' and the first word. "
        "DO NOT rewrite the user's voice. DO NOT add or remove content. DO NOT translate. "
        "Return ONLY the corrected text, no quotes, no explanation, no preamble."
    )
    u = f'Language: {lang}. Input: "{inp}"'
    return chatml(sys_p, u, exp)


def render_emergency_qa(o: dict) -> str | None:
    script, q, a = o.get("script"), o.get("question"), o.get("answer")
    if not (script and q and a):
        return None
    sys_p = (
        "You are an emergency-response AI on a phone call. Answer questions from a 911 operator using "
        "ONLY the facts in the script. If a fact is not in the script, say \"I don't have that "
        "information.\" Be concise. One sentence per answer. Stay calm and professional.\n\n"
        f"Emergency script:\n{script}"
    )
    return chatml(sys_p, q, a)


def render_caregiver_parse(o: dict) -> str | None:
    note, actions = o.get("note"), o.get("actions")
    if not (note and actions is not None):
        return None
    if not isinstance(actions, list):
        return None
    u = (
        "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
        "Available categories: help, food, feelings, school, quick, animals, colors, body, time\n\n"
        "Action types: add_phrase, remove_phrase, reorder_phrase, add_category, add_sequence, "
        "remove_sequence, boost_word, note_only.\n\n"
        f'Caregiver says: "{note}"\n\n'
        "Return ONLY a JSON array of action objects."
    )
    a = json.dumps(actions, ensure_ascii=False)
    return chatml("", u, a)


def render_translate(o: dict) -> str | None:
    text, src, tgt, t = o.get("text"), o.get("fromLang"), o.get("toLang"), o.get("translation")
    if not (text and src and tgt and t):
        return None
    sys_p = f"You are a translator. Translate from {src} to {tgt}. Return ONLY the translation — no explanations, no quotes, no extra text."
    return chatml(sys_p, text, t)


def render_ask_ai_aac(o: dict) -> str | None:
    q, a = o.get("question"), o.get("answer")
    if not (q and a):
        return None
    sys_p = (
        "You are a friendly helper for a child who uses an AAC (communication) device. "
        "Keep responses to 2-3 short sentences. Use simple words. Be encouraging and patient. "
        "If it is a math or science question, include relevant symbols."
    )
    return chatml(sys_p, q, a)


def render_word_predict_aac(o: dict) -> str | None:
    prefix, top5 = o.get("prefix"), o.get("top5")
    if not (prefix and isinstance(top5, list) and top5):
        return None
    words = [str(w).strip() for w in top5[:5] if w]
    if not words:
        return None
    sys_p = (
        "You predict the most likely next 1-3 words for an AAC user given a partial sentence. "
        "Return ONLY a comma-separated list of up to 5 single-word predictions, ranked from most "
        "to least likely. No explanation, no preamble."
    )
    return chatml(sys_p, prefix, ", ".join(words))


def render_tone_switch(o: dict) -> str | None:
    """AAC tone-classified response. Assistant output starts with a <tone:*> tag
    that synalux strips server-side before delivering to the user."""
    inp = o.get("input")
    resp = o.get("response")
    lang = o.get("lang", "en")
    if not (inp and resp):
        return None
    resp = resp.strip()
    # Reject rows that didn't include a tone tag — they aren't useful for the
    # tone-classification objective.
    if not resp.startswith("<tone:"):
        return None
    sys_p = (
        "You are an AAC response engine. Pick the appropriate tone register and prefix your response with "
        "a tone tag in <tone:gentle|urgent|clinical|playful|neutral> form. Pick urgent for emergencies, "
        "gentle for distress without emergency, clinical for medical/factual, playful for fun/social, "
        "neutral for everything else. The synalux portal strips the tone tag before delivery. "
        "After the tag, give a brief, AAC-appropriate response."
    )
    return chatml(sys_p, f"Language: {lang}. Input: {inp}", resp)


def render_tool_call(o: dict, tool_name: str | None = None) -> str | None:
    """Used for memory_checkout, hipaa, contrastive_extra."""
    prompt = o.get("prompt")
    if not prompt:
        return None
    think = o.get("think", "")
    args = o.get("args", {}) or {}
    name = o.get("right_tool") or tool_name
    if not name:
        return None
    a = (
        f"<|synalux_think|>\n{think}\n</|synalux_think|>\n\n"
        f"<|tool_call|>\n"
        f"{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n"
        f"</|tool_call|>"
    )
    return chatml("", prompt, a)


def render_no_tool(o: dict) -> str | None:
    prompt, think, ans = o.get("prompt"), o.get("think", ""), o.get("answer")
    if not (prompt and ans):
        return None
    a = (
        f"<|synalux_think|>\n{think}\n</|synalux_think|>\n\n"
        f"<|synalux_answer|>{ans}</|synalux_answer|>"
    )
    return chatml("", prompt, a)


# ── BFCL renderers — emit STANDARD Qwen FC format (<tool_call>...</tool_call>),
# NOT the synalux variant. The system prompt mirrors what bfcl-eval's QwenFCHandler
# produces so the model learns to switch formats based on the tools block style.
_BFCL_SYSTEM_TEMPLATE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)


def _bfcl_tools_json(tools) -> str | None:
    if not isinstance(tools, list) or not tools:
        return None
    try:
        return json.dumps(tools, ensure_ascii=False)
    except Exception:
        return None


def _coerce_call(call) -> dict | None:
    """Normalize teacher output of `call` to {name, arguments}.

    72B sometimes emits Python-call syntax instead of structured JSON, e.g.
    `"calculate_area(length=10, width=5)"`. We parse those into the canonical
    shape so they're usable as training data instead of dropping ~half the
    BFCL corpus to validator failures.
    """
    import re
    if isinstance(call, dict):
        if call.get("name"):
            return {"name": call["name"], "arguments": call.get("arguments", {}) or {}}
        return None
    if not isinstance(call, str):
        return None
    s = call.strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)\s*$", s, re.DOTALL)
    if not m:
        return None
    name = m.group(1).split(".")[-1]  # drop module prefix like java.io.file
    args_str = m.group(2).strip()
    args: dict = {}
    if args_str:
        # Split top-level commas (avoid splitting inside strings/brackets).
        depth = 0
        in_str = False
        quote = ""
        parts: list[str] = []
        cur: list[str] = []
        for ch in args_str:
            if in_str:
                cur.append(ch)
                if ch == quote and (len(cur) < 2 or cur[-2] != "\\"):
                    in_str = False
            elif ch in ('"', "'"):
                in_str = True
                quote = ch
                cur.append(ch)
            elif ch in "([{":
                depth += 1
                cur.append(ch)
            elif ch in ")]}":
                depth -= 1
                cur.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur).strip())

        for i, part in enumerate(parts):
            if "=" in part:
                k, _, v = part.partition("=")
                k = k.strip()
                v = v.strip()
            else:
                k = f"arg{i}"
                v = part.strip()
            # Try to parse the value as JSON, else fall back to the raw string
            # with surrounding quotes stripped.
            try:
                args[k] = json.loads(v)
            except Exception:
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    args[k] = v[1:-1]
                elif v.lower() in ("true", "false"):
                    args[k] = v.lower() == "true"
                elif v.lower() == "null":
                    args[k] = None
                else:
                    try:
                        args[k] = int(v)
                    except ValueError:
                        try:
                            args[k] = float(v)
                        except ValueError:
                            args[k] = v
    return {"name": name, "arguments": args}


def _bfcl_assistant_single(call: dict) -> str:
    return (
        "<tool_call>\n"
        f"{json.dumps({'name': call.get('name'), 'arguments': call.get('arguments', {})}, ensure_ascii=False)}\n"
        "</tool_call>"
    )


def _bfcl_assistant_multi(calls: list) -> str:
    parts = []
    for c in calls:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        parts.append(
            "<tool_call>\n"
            f"{json.dumps({'name': c['name'], 'arguments': c.get('arguments', {})}, ensure_ascii=False)}\n"
            "</tool_call>"
        )
    return "\n".join(parts) if parts else ""


def render_bfcl_simple(o: dict) -> str | None:
    """One tool, one call — also reused for simple_java/simple_javascript."""
    tools = o.get("tools")
    user = o.get("user")
    call = _coerce_call(o.get("call"))
    tools_json = _bfcl_tools_json(tools)
    if not (tools_json and user and call and call.get("name")):
        return None
    sys_p = _BFCL_SYSTEM_TEMPLATE.format(tools_json=tools_json)
    return chatml(sys_p, user, _bfcl_assistant_single(call))


def render_bfcl_parallel(o: dict) -> str | None:
    """One tool definition, multiple parallel calls in the response."""
    tools = o.get("tools")
    user = o.get("user")
    raw_calls = o.get("calls")
    tools_json = _bfcl_tools_json(tools)
    if not (tools_json and user and isinstance(raw_calls, list) and raw_calls):
        return None
    coerced = [_coerce_call(c) for c in raw_calls]
    coerced = [c for c in coerced if c and c.get("name")]
    if not coerced:
        return None
    a = _bfcl_assistant_multi(coerced)
    if not a:
        return None
    sys_p = _BFCL_SYSTEM_TEMPLATE.format(tools_json=tools_json)
    return chatml(sys_p, user, a)


def render_bfcl_irrelevance(o: dict) -> str | None:
    """1-3 tools but no call — model answers in natural language."""
    tools = o.get("tools")
    user = o.get("user")
    answer = o.get("answer")
    tools_json = _bfcl_tools_json(tools)
    if not (tools_json and user and isinstance(answer, str) and answer.strip()):
        return None
    sys_p = _BFCL_SYSTEM_TEMPLATE.format(tools_json=tools_json)
    return chatml(sys_p, user, answer.strip())


CATEGORY_RENDERERS = {
    "text_correct": render_text_correct,
    "emergency_qa": render_emergency_qa,
    "caregiver_parse": render_caregiver_parse,
    "translate": render_translate,
    "ask_ai_aac": render_ask_ai_aac,
    "word_predict_aac": render_word_predict_aac,
    "memory_checkout": lambda o: render_tool_call(o, "memory_checkout"),
    "hipaa": lambda o: render_tool_call(o, "hipaa"),
    "contrastive_extra": lambda o: render_tool_call(o, None),  # uses right_tool from object
    "no_tool_extra": render_no_tool,
    # v16.1 corrective categories
    "session_load_context_extra": lambda o: render_tool_call(o, "session_load_context"),
    "caregiver_parse_extra": render_caregiver_parse,
    "emergency_qa_extra": render_emergency_qa,
    "checkout_vs_loadcontext": lambda o: render_tool_call(o, None),  # uses right_tool
    # v16.2 BFCL-format categories — standard Qwen FC tool-call shape
    "bfcl_simple_python": render_bfcl_simple,
    "bfcl_multiple": render_bfcl_simple,
    "bfcl_parallel": render_bfcl_parallel,
    "bfcl_parallel_multiple": render_bfcl_parallel,
    "bfcl_simple_java": render_bfcl_simple,
    "bfcl_simple_javascript": render_bfcl_simple,
    "bfcl_irrelevance": render_bfcl_irrelevance,
    # v17 — tone-classification training
    "tone_switch": render_tone_switch,
}

# Add the same composite-key dedup rules for the new categories
DEDUP_KEYS["session_load_context_extra"] = lambda r: (r.get("prompt","").lower().strip(),)
DEDUP_KEYS["caregiver_parse_extra"] = lambda r: (r.get("note","").lower().strip(),)
DEDUP_KEYS["emergency_qa_extra"] = lambda r: (r.get("script","")[:200], r.get("question","").lower().strip())
DEDUP_KEYS["checkout_vs_loadcontext"] = lambda r: (r.get("prompt","").lower().strip(),)
# BFCL composite keys — dedup on the user request text only. call shape varies
# (string vs dict) per teacher output; the user prompt is the stable signal.
DEDUP_KEYS["bfcl_simple_python"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_multiple"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_parallel"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_parallel_multiple"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_simple_java"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_simple_javascript"] = lambda r: (r.get("user","").lower().strip(),)
DEDUP_KEYS["bfcl_irrelevance"] = lambda r: (r.get("user","").lower().strip(),)
# v17 — tone-classification dedup on input + tone (allow same input across tones)
DEDUP_KEYS["tone_switch"] = lambda r: (r.get("input","").lower().strip(), r.get("tone",""))


def load_category(path: Path):
    """Load raw records from a category jsonl."""
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
            except Exception:
                pass
    return out


def union_dedup(cat: str, primary_records, fallback_records):
    """Union primary (72B) + fallback (7B) with composite-key dedup,
    prefer primary on conflict. Returns unique ordered list of records."""
    keyfn = DEDUP_KEYS.get(cat, lambda r: (json.dumps(r, sort_keys=True),))
    seen = set()
    out = []
    # primary first → wins for any duplicate key
    for r in primary_records + fallback_records:
        try:
            k = keyfn(r)
        except Exception:
            continue
        if not k or not any(k):
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def fix_contrastive_args(records):
    """Synthesize empty args dict for any contrastive_extra record missing it."""
    for r in records:
        if not r.get("args"):
            r["args"] = {}
    return records


def process_category(cat: str, render):
    """Load 72B + 7B + corrective for this category, dedupe, render, yield {text}."""
    primary = load_category(GEN_DIR_72B / f"{cat}.jsonl")
    fallback = load_category(GEN_DIR_7B / f"{cat}.jsonl")
    corrective = load_category(GEN_DIR_CORRECTIVE / f"{cat}.jsonl")
    # Corrective examples take precedence (added FIRST), then 72B, then 7B
    deduped = union_dedup(cat, corrective + primary, fallback)
    if cat in ("contrastive_extra", "checkout_vs_loadcontext"):
        deduped = fix_contrastive_args(deduped)
    n_render = 0
    for obj in deduped:
        text = render(obj)
        if not text or len(text) < 50:
            continue
        n_render += 1
        yield {"text": text}
    print(f"  {cat:25s}  corr={len(corrective):4d}  72b={len(primary):5d}  7b={len(fallback):5d}  unique={len(deduped):5d}  rendered={n_render:5d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-legacy", action="store_true", help="exclude existing train.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    args = ap.parse_args()

    rows: list[dict] = []

    # Legacy v12 train.jsonl (already in canonical format)
    if not args.no_legacy and LEGACY.exists():
        with LEGACY.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj.get("text"), str) and len(obj["text"]) >= 50:
                    rows.append({"text": obj["text"]})
        print(f"  legacy train.jsonl                in={len(rows):5d}  rendered={len(rows):5d}")

    legacy_count = len(rows)

    # Generated v16 categories: union of 72B (primary) + 7B (fallback), deduped
    if not GEN_DIR_72B.exists() and not GEN_DIR_7B.exists():
        print(f"ERROR: neither {GEN_DIR_72B} nor {GEN_DIR_7B} found", file=sys.stderr)
        sys.exit(1)

    for cat, render in CATEGORY_RENDERERS.items():
        for r in process_category(cat, render):
            rows.append(r)

    new_count = len(rows) - legacy_count
    print(f"\nTotals: legacy={legacy_count}  new={new_count}  combined={len(rows)}")

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
