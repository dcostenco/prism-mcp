#!/usr/bin/env python3
"""Realigned AAC validation — runs prism-aac's REAL production paths.

Replaces the generic SCERTS test with the actual system prompts pulled from:
  - prism-aac/services/textCorrectService.ts  (LOCAL_SYSTEM)
  - prism-aac/services/aiService.ts           (askAI, translateAI, parseCaregiverNote)
  - prism-aac/README.md                       (Live emergency Q&A — 13 Twilio test questions)

Each category uses prism-aac's exact system prompt + 80-token cap (matching
production temperature=0.0 / num_predict=80 for text-correct, etc.).

Output: results/aac_realigned_<model>.json

Pass criterion (per category):
  text_correct  — normalized exact match against expected, OR token-overlap >= 0.85
  emergency_qa  — substring match of any acceptable answer (factual extraction)
  caregiver     — parses to JSON array AND contains expected action types
  translate     — normalized exact match OR overlap >= 0.7 (translation flex)
  ask_ai        — overlap >= 0.5 (open-ended; just measure non-degenerate)
"""
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ.get("MODEL", "prism-coder:7b")
OUT_PATH = Path(__file__).parent / "results" / f"aac_realigned_{MODEL.replace(':','_').replace('/','_')}.json"

# ── Production system prompts (verbatim from prism-aac source) ──────────────

TEXT_CORRECT_SYSTEM = """You are a fast text-cleanup engine for an AAC (augmentative and alternative communication) app used by users with motor impairments. Your only job: take possibly-malformed input and return the most likely intended utterance.

Rules:
- Fix obvious typos, missing spaces, dropped letters, transposed letters.
- Fix voice-transcript word-boundary errors (e.g. "bowlof rice" -> "bowl of rice", "i wantto eat" -> "i want to eat").
- Fix spurious commas/punctuation that came from hurried typing.
- Capitalize "I" and the first word.
- DO NOT rewrite the user's voice — keep their words and tone.
- DO NOT add new content the user did not say.
- DO NOT remove content the user did say.
- DO NOT translate.
- If the input is already well-formed, return it unchanged.
- Return ONLY the corrected text, no quotes, no explanation, no preamble."""

ASK_AI_SYSTEM = """You are a friendly helper for a child who uses an AAC (communication) device.
The child may have limited vocabulary. Keep responses to 2-3 short sentences.
Use simple words. Be encouraging and patient.
If it is a math or science question, include relevant symbols.
If the child asks about a real-world topic, use web search results if available."""

TRANSLATE_SYSTEM_TPL = "You are a translator. Translate the input from {fromLang} to {toLang}. Return ONLY the translation — no explanations, no quotes, no extra text."

# Emergency AI runs server-side via Twilio in production, but the same model
# answers. We test with the documented emergency script as system context.
EMERGENCY_SCRIPT = """This is an automated emergency call from PrismAAC. An 8-year-old nonverbal individual named Alex needs help. They communicated: I cannot breathe. Location: 123 Oak Street, Room 4, Springfield. Medical conditions: epilepsy. Allergies: penicillin. Medications: Keppra 250 milligrams. Callback number: 555-0123. Alert time: 4/30/2026, 12:35:37 PM."""

EMERGENCY_SYSTEM = f"""You are an emergency-response AI on a phone call. You answer questions from a 911 operator using ONLY the facts below. If a fact is not in the script, say "I don't have that information."

Emergency script:
{EMERGENCY_SCRIPT}

Be concise. One sentence per answer. Stay calm and professional."""

CAREGIVER_PROMPT_TPL = """You are an AAC app configuration assistant for a BCBA/caregiver.
Available categories: help: Help/Needs, food: Food & Drink, feelings: Feelings, school: School/Work, quick: Quick Talk

Parse the following caregiver instruction into structured JSON actions.
Return ONLY a JSON array of action objects. No explanation.

Action types:
  add_phrase: {{ categoryId, text }}
  remove_phrase: {{ phraseText, categoryId }}
  add_category: {{ name, icon }}
  add_sequence: {{ name, categoryId, steps }}
  boost_word: {{ word, boostCount }}
  note_only: {{}} (for observations with no configuration change)

Caregiver says: "{note}"

Return JSON array like: [{{"type":"add_phrase","description":"...","payload":{{...}}}}]"""

# ── Test cases ──────────────────────────────────────────────────────────────

TEXT_CORRECT_CASES = [
    # Word-boundary fixes (the prime use case)
    {"input": "i wantto eat bowlof rice", "lang": "en", "expected": "I want to eat bowl of rice"},
    {"input": "imhungry can i hav water", "lang": "en", "expected": "I'm hungry, can I have water"},
    {"input": "ineedto goto bathroom", "lang": "en", "expected": "I need to go to bathroom"},
    {"input": "can yu help me please", "lang": "en", "expected": "Can you help me please"},
    {"input": "iwant to playoutside", "lang": "en", "expected": "I want to play outside"},
    # Punctuation cleanup
    {"input": "yes,i,want,more,please,", "lang": "en", "expected": "Yes I want more please"},
    {"input": "no..stop..hurts", "lang": "en", "expected": "No stop hurts"},
    # Already correct (must pass through)
    {"input": "Hello, how are you?", "lang": "en", "expected": "Hello, how are you?"},
    {"input": "I love my mom.", "lang": "en", "expected": "I love my mom."},
    # Dropped letters
    {"input": "i ned hlp now", "lang": "en", "expected": "I need help now"},
    {"input": "wer is mom", "lang": "en", "expected": "Where is mom"},
    # Voice transcript artifacts
    {"input": "umm i wanna go home", "lang": "en", "expected": "I want to go home"},
    # Emergency phrases (CRITICAL — must not corrupt)
    {"input": "ican't breathe", "lang": "en", "expected": "I can't breathe"},
    {"input": "someone hurt me", "lang": "en", "expected": "Someone hurt me"},
    {"input": "i dont feel safe", "lang": "en", "expected": "I don't feel safe"},
]

EMERGENCY_QA_CASES = [
    # The 13 documented Twilio test questions (from README live test)
    {"q": "Where is the patient?", "accept": ["123 Oak Street", "Room 4", "Springfield"]},
    {"q": "How old is the patient?", "accept": ["8", "eight", "8-year-old", "eight years old"]},
    {"q": "What happened?", "accept": ["cannot breathe", "can't breathe", "breathing", "breath"]},
    {"q": "What is the callback number?", "accept": ["555-0123", "5550123"]},
    {"q": "Does the patient have allergies?", "accept": ["penicillin"]},
    {"q": "What medications is the patient taking?", "accept": ["Keppra", "keppra", "250 milligrams", "250mg"]},
    {"q": "What medical conditions does the patient have?", "accept": ["epilepsy"]},
    {"q": "What is the patient's name?", "accept": ["Alex", "alex"]},
    {"q": "When did this happen?", "accept": ["12:35", "4/30/2026", "April 30", "today"]},
    {"q": "Is anyone with the patient?", "accept": ["alone", "by themselves", "no one", "don't have", "do not have"]},
    {"q": "Is the patient verbal?", "accept": ["nonverbal", "non-verbal", "cannot speak", "can't speak", "AAC"]},
    {"q": "What is the address?", "accept": ["123 Oak Street", "Oak Street", "Springfield"]},
    {"q": "Who is calling?", "accept": ["PrismAAC", "automated", "emergency call"]},
]

TRANSLATE_CASES = [
    {"text": "I am hungry", "fromLang": "English", "toLang": "Spanish", "accept": ["tengo hambre", "estoy hambriento", "yo tengo hambre"]},
    {"text": "Help me please", "fromLang": "English", "toLang": "Spanish", "accept": ["ayúdame por favor", "ayudame por favor", "por favor ayúdame"]},
    {"text": "Where is the bathroom", "fromLang": "English", "toLang": "Spanish", "accept": ["dónde está el baño", "donde esta el baño", "dónde queda el baño"]},
    {"text": "I love you", "fromLang": "English", "toLang": "French", "accept": ["je t'aime", "je vous aime"]},
    {"text": "I need water", "fromLang": "English", "toLang": "French", "accept": ["j'ai besoin d'eau", "il me faut de l'eau"]},
    {"text": "Estoy cansado", "fromLang": "Spanish", "toLang": "English", "accept": ["i am tired", "i'm tired"]},
    {"text": "Help", "fromLang": "English", "toLang": "Russian", "accept": ["помогите", "помоги", "помощь"]},
    {"text": "Thank you", "fromLang": "English", "toLang": "Japanese", "accept": ["ありがとう", "ありがとうございます", "arigatou", "arigato"]},
]

CAREGIVER_CASES = [
    {"note": "Add 'I feel sick' to Help", "must_have_action": "add_phrase"},
    {"note": "Remove the phrase 'I want candy' from Food", "must_have_action": "remove_phrase"},
    {"note": "He's using 'because' a lot now", "must_have_action": "boost_word"},
    {"note": "Good session today, 15 phrases independently", "must_have_action": "note_only"},
    {"note": "Add a McDonald's ordering flow with: drink (water, milk), food (burger, nuggets), size (small, medium)", "must_have_action": "add_sequence"},
    {"note": "Add 'I'm tired' to Feelings", "must_have_action": "add_phrase"},
    {"note": "Move 'bathroom' to top of Help category", "must_have_action": "reorder_phrase"},
]

ASK_AI_CASES = [
    {"q": "What is 5 + 3?", "must_contain": ["8", "eight"]},
    {"q": "What color is the sky?", "must_contain": ["blue"]},
    {"q": "How many legs does a dog have?", "must_contain": ["4", "four"]},
    {"q": "What sound does a cat make?", "must_contain": ["meow", "purr"]},
    {"q": "Is fire hot or cold?", "must_contain": ["hot"]},
]

# ── Scoring ─────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"^[\"'\s]+|[\"'\s]+$", "", s)
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


def overlap(a: str, b: str) -> float:
    A = set(normalize(a).split())
    B = set(normalize(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


def score_text_correct(resp: str, expected: str) -> tuple[bool, float]:
    if not resp:
        return False, 0.0
    n_resp, n_exp = normalize(resp), normalize(expected)
    if n_resp == n_exp:
        return True, 1.0
    o = overlap(resp, expected)
    return o >= 0.85, o


def score_emergency(resp: str, accept: list[str]) -> tuple[bool, float]:
    if not resp:
        return False, 0.0
    nr = normalize(resp)
    for a in accept:
        if normalize(a) in nr:
            return True, 1.0
    best = max((overlap(resp, a) for a in accept), default=0.0)
    return False, best


def score_translate(resp: str, accept: list[str]) -> tuple[bool, float]:
    if not resp:
        return False, 0.0
    nr = normalize(resp)
    for a in accept:
        na = normalize(a)
        if na in nr or nr in na:
            return True, 1.0
    best = max((overlap(resp, a) for a in accept), default=0.0)
    return best >= 0.7, best


def score_caregiver(resp: str, must_action: str) -> tuple[bool, float, str]:
    if not resp:
        return False, 0.0, ""
    cleaned = re.sub(r"```(?:json)?\s*", "", resp).rstrip("`").strip()
    # Find first JSON array
    m = re.search(r"\[[\s\S]*\]", cleaned)
    if not m:
        return False, 0.0, "no_json"
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return False, 0.0, "json_parse_fail"
    if not isinstance(arr, list) or not arr:
        return False, 0.0, "not_array"
    types = [a.get("type", "") for a in arr if isinstance(a, dict)]
    if must_action in types:
        return True, 1.0, "match"
    return False, 0.5, f"got={types}"


def score_ask_ai(resp: str, must_contain: list[str]) -> tuple[bool, float]:
    if not resp:
        return False, 0.0
    nr = normalize(resp)
    for c in must_contain:
        if normalize(c) in nr:
            return True, 1.0
    return False, 0.0


# ── Ollama call ─────────────────────────────────────────────────────────────

def call(system: str, prompt: str, num_predict: int = 80, temperature: float = 0.0, timeout: int = 60) -> tuple[str, int]:
    payload = json.dumps({
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("response", "").strip(), int((time.time() - t0) * 1000)


# ── Runners ─────────────────────────────────────────────────────────────────

def run_text_correct(case):
    try:
        resp, lat = call(TEXT_CORRECT_SYSTEM, f"Language: {case['lang']}. Input: \"{case['input']}\"", num_predict=80)
        # strip surrounding quotes the model often adds
        clean = resp.strip().strip('"').strip("'").split("\n")[0].strip()
        ok, sim = score_text_correct(clean, case["expected"])
        return {"task": "text_correct", "input": case["input"], "expected": case["expected"], "response": clean, "ok": ok, "score": round(sim, 3), "latency_ms": lat}
    except Exception as e:
        return {"task": "text_correct", "input": case["input"], "ok": False, "score": 0.0, "error": str(e)}


def run_emergency(case):
    try:
        resp, lat = call(EMERGENCY_SYSTEM, case["q"], num_predict=120, temperature=0.0)
        ok, sim = score_emergency(resp, case["accept"])
        return {"task": "emergency_qa", "q": case["q"], "accept": case["accept"], "response": resp, "ok": ok, "score": round(sim, 3), "latency_ms": lat}
    except Exception as e:
        return {"task": "emergency_qa", "q": case["q"], "ok": False, "score": 0.0, "error": str(e)}


def run_translate(case):
    try:
        sys_p = TRANSLATE_SYSTEM_TPL.format(fromLang=case["fromLang"], toLang=case["toLang"])
        resp, lat = call(sys_p, case["text"], num_predict=80, temperature=0.0)
        clean = resp.strip().strip('"').strip("'").split("\n")[0].strip()
        ok, sim = score_translate(clean, case["accept"])
        return {"task": "translate", "text": case["text"], "to": case["toLang"], "accept": case["accept"], "response": clean, "ok": ok, "score": round(sim, 3), "latency_ms": lat}
    except Exception as e:
        return {"task": "translate", "text": case["text"], "ok": False, "score": 0.0, "error": str(e)}


def run_caregiver(case):
    try:
        prompt = CAREGIVER_PROMPT_TPL.format(note=case["note"])
        resp, lat = call("", prompt, num_predict=350, temperature=0.0)
        ok, sim, why = score_caregiver(resp, case["must_have_action"])
        return {"task": "caregiver", "note": case["note"], "must_have": case["must_have_action"], "response": resp[:500], "ok": ok, "score": round(sim, 3), "diag": why, "latency_ms": lat}
    except Exception as e:
        return {"task": "caregiver", "note": case["note"], "ok": False, "score": 0.0, "error": str(e)}


def run_ask_ai(case):
    try:
        resp, lat = call(ASK_AI_SYSTEM, case["q"], num_predict=120, temperature=0.3)
        ok, sim = score_ask_ai(resp, case["must_contain"])
        return {"task": "ask_ai", "q": case["q"], "must_contain": case["must_contain"], "response": resp, "ok": ok, "score": round(sim, 3), "latency_ms": lat}
    except Exception as e:
        return {"task": "ask_ai", "q": case["q"], "ok": False, "score": 0.0, "error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== Realigned AAC eval against {MODEL} ===\n")

    jobs = []
    for c in TEXT_CORRECT_CASES:
        jobs.append((run_text_correct, c))
    for c in EMERGENCY_QA_CASES:
        jobs.append((run_emergency, c))
    for c in TRANSLATE_CASES:
        jobs.append((run_translate, c))
    for c in CAREGIVER_CASES:
        jobs.append((run_caregiver, c))
    for c in ASK_AI_CASES:
        jobs.append((run_ask_ai, c))

    print(f"running {len(jobs)} cases (parallel=4)...\n")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(fn, c) for fn, c in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            tag = "OK " if r.get("ok") else "FAIL"
            ident = r.get("input") or r.get("q") or r.get("text") or r.get("note") or ""
            ident = (ident or "")[:60]
            print(f"  [{i:3d}/{len(jobs)}] {tag} {r['task']:13s} {ident}")

    # Group + summarize
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    summary = {"model": MODEL, "wall_time_s": round(time.time() - t0, 1), "by_task": {}, "results": results}
    print(f"\n=== Summary ({MODEL}) ===")
    overall_ok = 0
    overall_total = 0
    for task, rs in sorted(by_task.items()):
        ok = sum(1 for r in rs if r.get("ok"))
        tot = len(rs)
        avg_lat = sum(r.get("latency_ms", 0) for r in rs) / tot if tot else 0
        summary["by_task"][task] = {"passed": ok, "total": tot, "pct": round(ok / tot, 4), "avg_latency_ms": int(avg_lat)}
        overall_ok += ok
        overall_total += tot
        print(f"  {task:14s} {ok:2d}/{tot:2d} = {ok/tot*100:5.1f}%   lat={int(avg_lat)}ms")
    summary["overall"] = {"passed": overall_ok, "total": overall_total, "pct": round(overall_ok / overall_total, 4)}
    print(f"  {'OVERALL':14s} {overall_ok:2d}/{overall_total:2d} = {overall_ok/overall_total*100:5.1f}%")

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nReport: {OUT_PATH}")


if __name__ == "__main__":
    main()
