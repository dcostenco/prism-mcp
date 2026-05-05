#!/usr/bin/env python3
"""Local SFT data generator for v16 — uses Ollama qwen2.5-coder:32b as teacher.

Generates the same 10 categories as modal_sft_generator.py but runs locally
against the already-installed qwen2.5-coder:32b. Parallel via ThreadPool.

Output: data/v16_gen/<category>.jsonl

Run:
  source venv/bin/activate
  python3 generate_v16_data.py [--cats text_correct,emergency_qa]
  python3 generate_v16_data.py --per-batch 30 --workers 2
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = os.environ.get("TEACHER_MODEL", "qwen2.5-coder:32b")
OUT_DIR = Path(__file__).parent / "data" / "v16_gen"

# Category specs — total ~5060 examples
SPECS = {
    "text_correct": {
        "target": 1500, "batch": 20,
        "system": (
            "You generate AAC text-correction training examples. Each example: a malformed AAC user "
            "input (typos, missing spaces, voice-transcript word boundaries, hurried punctuation) and "
            "the most-likely intended utterance. Cover: word-boundary fixes (\"bowlof rice\"->\"bowl of rice\"), "
            "dropped letters, transposed letters, spurious commas, voice 'umm'/'uhh' fillers, missing apostrophes. "
            "Include emergency phrases too — those MUST be preserved verbatim, just with correct spelling. "
            "Mix English and a sprinkle of Spanish/French. "
            "Return ONLY a JSON array of {input, lang, expected} objects — no fences, no preamble."
        ),
        "user": "Generate {n} diverse AAC text-correction examples. ~70% English, 15% Spanish, 10% French, 5% emergency phrases (preserved exactly).",
    },
    "emergency_qa": {
        "target": 500, "batch": 15,
        "system": (
            "You generate emergency-phone-call Q&A training data for an AAC emergency-response AI. "
            "Each example: a 911 operator's question + a CONTEXT BLOCK (emergency script with name, "
            "age, location, conditions, allergies, medications, callback). The answer extracts ONLY "
            "facts present in the script. If the question asks something not in the script, the answer "
            "must be \"I don't have that information.\" "
            "Vary scripts: child/teen/adult, conditions (epilepsy, asthma, diabetes, autism, cardiac, "
            "anaphylaxis), locations (home, school, hospital, public). "
            "Vary questions: location, age, what happened, allergies, meds, conditions, callback, "
            "who's with the patient, time, who is calling, AAC vs verbal, etc. "
            "Return ONLY a JSON array of {script, question, answer} objects."
        ),
        "user": "Generate {n} emergency Q&A examples with full script + operator question + extracted answer.",
    },
    "caregiver_parse": {
        "target": 500, "batch": 15,
        "system": (
            "You generate caregiver-note parsing training examples for an AAC app. Each: a natural "
            "caregiver/BCBA instruction + the structured JSON action array it should parse to. "
            "Action types: add_phrase {categoryId, text}, remove_phrase {phraseText, categoryId}, "
            "reorder_phrase {phraseId, newSortOrder, categoryId}, add_category {name, icon}, "
            "add_sequence {name, categoryId, steps:[{label, options:[string]}]}, remove_sequence "
            "{sequenceName}, boost_word {word, boostCount}, note_only {} (clinical observation). "
            "categoryIds: help, food, feelings, school, quick, animals, colors, body, time. "
            "Return ONLY a JSON array of {note, actions} objects."
        ),
        "user": "Generate {n} diverse caregiver notes covering all 8 action types. Mix simple + complex multi-action notes.",
    },
    "translate": {
        "target": 500, "batch": 20,
        "system": (
            "You generate translation training examples for an AAC app. Each: a short AAC utterance "
            "(2-10 words) + source language + target language + ONE acceptable translation. "
            "Targets: Spanish, French, Portuguese, Romanian, Ukrainian, Russian, German, Japanese, "
            "Korean, Chinese, Arabic. Also include reverse-direction. AAC content: requests, feelings, "
            "basic needs, social greetings, emergency phrases. "
            "Return ONLY a JSON array of {text, fromLang, toLang, translation} objects."
        ),
        "user": "Generate {n} translation examples covering all 11 target languages roughly evenly.",
    },
    "ask_ai_aac": {
        "target": 500, "batch": 20,
        "system": (
            "You generate AAC-helper Q&A training examples. The AI is a friendly helper for a child "
            "who uses an AAC device. Responses MUST be 2-3 short sentences, simple words, encouraging tone. "
            "Topics: math (single-digit + and -), science basics, social, school (counting/letters/shapes), "
            "safety. Use vocabulary appropriate for ages 4-10. "
            "Return ONLY a JSON array of {question, answer} objects."
        ),
        "user": "Generate {n} kid-AAC questions and 2-3 sentence simple answers.",
    },
    "word_predict_aac": {
        "target": 1000, "batch": 25,
        "system": (
            "You generate word-prediction training data for AAC. Each: a partial sentence (with '___' "
            "marking the blank) + the top 5 most-likely next words for an AAC user. Use AAC core "
            "vocabulary (I, want, need, like, more, please, help, mom, dad, eat, drink, go, play, "
            "stop, no, yes). Order most likely first. "
            "Return ONLY a JSON array of {prefix, top5} objects — top5 is a list of 5 single-word strings."
        ),
        "user": "Generate {n} word-prediction examples across diverse AAC contexts.",
    },
    "memory_checkout": {
        "target": 80, "batch": 10,
        "system": (
            "You generate tool-call training examples for the SYNALUX memory_checkout tool. "
            "memory_checkout(project, version) — restores project memory to a specific version. "
            "Each: USER prompt requesting memory restoration to a version, plus the correct tool call. "
            "<think> reasoning EXPLICITLY DEBATES tempting alternatives (session_restore_handoff, "
            "memory_history, session_load_context, session_checkout) and explains why memory_checkout "
            "is correct because the user named a SPECIFIC version number. "
            "Return ONLY JSON: [{prompt, think, args}]. args = {project, version}."
        ),
        "user": "Generate {n} memory_checkout examples. Vary phrasings and project names.",
    },
    "hipaa": {
        "target": 80, "batch": 10,
        "system": (
            "You generate tool-call training examples for the SYNALUX hipaa tool. "
            "hipaa(action, text, data, encrypted, event, user, client_id) — verifies HIPAA compliance, "
            "encrypts/decrypts PHI, audits PHI access. action in {verify, encrypt, decrypt, audit, log_access}. "
            "Each: USER prompt about HIPAA/PHI/compliance + correct tool call. <think> reasoning debates "
            "session_health_check (tempting because both involve 'check') and explains why hipaa is "
            "correct because the topic is PHI/compliance. "
            "Return ONLY JSON: [{prompt, think, args}]."
        ),
        "user": "Generate {n} hipaa tool examples with varied actions and PHI subjects.",
    },
    "contrastive_extra": {
        "target": 200, "batch": 15,
        "system": (
            "You generate contrastive SFT examples to fix bias in two confused tool pairs:\n"
            "(A) session_save_ledger vs session_save_experience — for 'record this work' phrasing. "
            "session_save_ledger = end-of-session work logs. session_save_experience = structured "
            "learning events (correction/success/failure with context+action+outcome).\n"
            "(B) session_forget_memory vs knowledge_forget — for 'remove memory about X' phrasing. "
            "session_forget_memory = delete a SPECIFIC memory entry. knowledge_forget = purge "
            "KNOWLEDGE entries by project/age.\n"
            "Mix BOTH directions evenly. Each example has <think> naming the wrong tool and the "
            "discriminating signal. "
            "Return JSON: [{prompt, think, right_tool, args}]."
        ),
        "user": "Generate {n} contrastive examples ~50/50 split between pairs A and B.",
    },
    "no_tool_extra": {
        "target": 200, "batch": 15,
        "system": (
            "You generate NO-TOOL adversarial training examples. USER prompt USES Prism tool keywords "
            "('session', 'memory', 'forget', 'search', 'load', 'export', 'knowledge', 'compact', "
            "'health') in a GENERAL PROGRAMMING context — NOT a request to use Prism tools. "
            "ASSISTANT response: <think> trace names the tempting Prism tool, explains the keyword "
            "trap, identifies the discriminating signal (general programming question). Then a brief "
            "1-2 sentence direct answer wrapped in <|synalux_answer|>. "
            "Cover: PHP session_start, BFS/DFS search, LSTM forget gates, pg_dump, RAG retrieval, GC, "
            "system health checks, etc. "
            "Return JSON: [{prompt, think, answer}]."
        ),
        "user": "Generate {n} NO-TOOL adversarial examples each tricking on a different keyword.",
    },
}


def call_teacher(spec: dict, n: int, retries: int = 2):
    payload = json.dumps({
        "model": TEACHER,
        "system": spec["system"],
        "prompt": spec["user"].format(n=n),
        "stream": False,
        "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 8000, "num_ctx": 8192},
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = (data.get("response") or "").strip()
            text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
            if "[" in text and "]" in text:
                text = text[text.index("["): text.rindex("]") + 1]
            arr = json.loads(text)
            if isinstance(arr, list):
                return arr
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
                continue
    raise RuntimeError(f"teacher failed: {last_err}")


def run_one_batch(category: str, batch_n: int) -> list:
    spec = SPECS[category]
    try:
        return call_teacher(spec, min(batch_n, spec["batch"]))
    except Exception as e:
        print(f"  [{category}] batch failed: {e}", file=sys.stderr)
        return []


def run_category(category: str, target: int, batch_size: int, workers: int = 2) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{category}.jsonl"
    n_batches = (target + batch_size - 1) // batch_size
    print(f"[{category}] target={target} batches={n_batches} batch_size={batch_size}", flush=True)

    kept = 0
    t0 = time.time()
    with out_path.open("w") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one_batch, category, batch_size) for _ in range(n_batches)]
            for i, fut in enumerate(as_completed(futs), 1):
                arr = fut.result()
                for ex in arr:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    kept += 1
                if i % 5 == 0 or i == n_batches:
                    print(f"  [{category}] batch {i}/{n_batches} kept_total={kept} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{category}] done: {kept} examples in {time.time()-t0:.0f}s -> {out_path}", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="", help="comma-separated category names; default all")
    ap.add_argument("--per-batch", type=int, default=0, help="override batch size")
    ap.add_argument("--workers", type=int, default=2, help="parallel batches per category")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply targets (use <1 for testing)")
    args = ap.parse_args()

    cats = [c.strip() for c in args.cats.split(",") if c.strip()] if args.cats else list(SPECS)
    cats = [c for c in cats if c in SPECS]
    if not cats:
        print("no valid categories", file=sys.stderr); sys.exit(1)

    print(f"=== Generating with {TEACHER} ===")
    print(f"categories: {cats}")
    print()

    total = 0
    for cat in cats:
        spec = SPECS[cat]
        target = max(int(spec["target"] * args.scale), 1)
        batch = args.per_batch or spec["batch"]
        kept = run_category(cat, target, batch, workers=args.workers)
        total += kept
        print()

    print(f"=== TOTAL: {total} examples across {len(cats)} categories ===")


if __name__ == "__main__":
    main()
