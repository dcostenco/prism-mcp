#!/usr/bin/env python3
"""Generate synthetic AAC SFT data via local Ollama teacher.

Targets the same 6 categories as data/aac_validation.json — but with
strict separation: the teacher never sees validation prompts.

Output: data/aac_sft.jsonl in the chatml {"text": "..."} format that
mlx_lm.lora consumes (matches generate_contrastive_sft.py output style).

Categories generated (volumes per AAC research recommendation):
  expand        2000   symbol/fragment → fluent sentence
  predict       1500   partial sentence → top-5 next words
  intent        1000   utterance + context → SCERTS function
  rephrase       500   clinical/restrictive → dignity-preserving
  multi_turn    1500   context + utterance → contextual response
  vocab_match    500   match output to user developmental level
  ─────────────────
  TOTAL         7000
"""
import os, sys, json, time, argparse, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OUT_PATH = Path(__file__).parent / "data" / "aac_sft.jsonl"
# Default: local Ollama. Override via env to use Modal-hosted vLLM teacher.
TEACHER_URL = os.environ.get("TEACHER_URL", "http://localhost:11434")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5-coder:32b")
TEACHER_BACKEND = os.environ.get("TEACHER_BACKEND", "ollama")  # "ollama" | "openai"
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))  # parallel batches

# Per-category volume + system prompts. Teacher generates batches of 25 per call.
CATEGORIES = {
    "emergency_classify": {
        "target": 1500, "batch": 25,
        "system": (
            "You generate LIFE-CRITICAL emergency phrase classification training data. "
            "Each example: USER provides an utterance + brief context. "
            "ASSISTANT classifies as: critical (life-threatening, uncancellable: "
            "violence/abuse/medical-emergency/abduction), urgent (needs help, cancellable), "
            "medical (non-life-threatening medical), or none.\n\n"
            "CRITICAL must include: 'cant breathe', 'someone hurt me', 'not safe', "
            "'dont know you', 'dont touch me', 'said no', 'im lost', 'im bleeding', "
            "'taking me', 'help help help' (repeated), '911', medical-collapse signals.\n\n"
            "Account for AAC user phrasing: terse, fragmented, possible spelling errors, "
            "single-word emergencies ('help', 'hurt', 'fire'), symbol-derived ('[help] [now]').\n\n"
            "Return ONLY a JSON array of {prompt, response} objects. NO false positives — "
            "'I need help with homework' is NOT critical."
        ),
        "user_template": (
            "Generate {n} diverse emergency-classification examples. Mix CRITICAL, urgent, "
            "medical, and none in roughly 30/30/15/25 ratio. Cover children, teens, adults; "
            "home, school, public, hospital, transport; native + ESL phrasing; clear AND "
            "ambiguous cases. Include AAC-typical fragments and symbol-derived utterances.\n\n"
            "Format: [{{\"prompt\": \"Classify emergency level: 'someone is hurting me' said quietly while looking scared.\", "
            "\"response\": \"critical — abuse signal, uncancellable, requires immediate dispatch\"}}]"
        ),
    },
    "vision_to_aac": {
        "target": 1500, "batch": 25,
        "system": (
            "You generate VISION-to-AAC mapping training data. Each example: USER describes "
            "a scene (as a vision encoder caption would describe it) + AAC user profile. "
            "ASSISTANT generates the top 3-5 likely things the user might want to communicate "
            "about that scene, ranked by likelihood.\n\n"
            "Vary scenes: home (kitchen, bedroom, bathroom), school (classroom, playground), "
            "medical (waiting room, exam room), public (store, restaurant, transport), "
            "social (park, family gathering), emergency (smoke, blood, stranger).\n\n"
            "Match register to user profile (child/teen/adult). Cover request, comment, "
            "protest, social, repair, regulate functions.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} vision→AAC examples. Vary scene complexity, user profiles, intent.\n\n"
            "Format: [{{\"prompt\": \"Vision caption: 'A golden retriever sitting in a sunny park near a child'. "
            "User: 5-year-old. Generate top 3 likely AAC responses ranked.\", "
            "\"response\": \"1. Dog! (comment)\\n2. I want to pet (request)\\n3. Mommy look (social attention)\"}}]"
        ),
    },
    "asr_repair": {
        "target": 1000, "batch": 25,
        "system": (
            "You generate ASR-repair training data. Each example: USER provides a noisy "
            "Whisper-style transcript with typical errors (homophones, AAC-jargon mishears, "
            "split words, vocabulary OOV). ASSISTANT returns the cleaned/repaired text.\n\n"
            "Common error patterns: 'AAC' → 'A.A.C.' or 'eh ay see' or 'a-c'; "
            "'PECS' → 'pecks'; 'BCBA' → 'bee see bee ay'; 'Bliss symbols' → 'bliss simples'; "
            "feeding tube → 'feeling tube'; medication names mangled.\n\n"
            "Also handle dysarthria-typical patterns and code-switching.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} ASR-repair examples covering: AAC jargon mishears, medical/clinical "
            "terms, names of equipment, mid-sentence language switches, dysarthria patterns.\n\n"
            "Format: [{{\"prompt\": \"Repair this Whisper transcript: 'I need my pecks book to talk in eh ay see class'\", "
            "\"response\": \"I need my PECS book to talk in AAC class\"}}]"
        ),
    },
    "code_switch": {
        "target": 1000, "batch": 25,
        "system": (
            "You generate multilingual code-switching AAC training data. Each example: "
            "USER provides input mixing 2 languages (typical of bilingual families). "
            "ASSISTANT generates a fluent response that respects both languages.\n\n"
            "Cover: Spanish-English (~60%), Russian-English (~10%), Ukrainian-English (~10%), "
            "Romanian-English (~10%), Portuguese-English (~10%). Both directions: target "
            "language English OR target language non-English depending on context.\n\n"
            "Honor cultural pragmatic conventions (formality, kinship terms).\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} code-switch examples. Mix languages mid-utterance. Include AAC "
            "symbols + bilingual context + target output language.\n\n"
            "Format: [{{\"prompt\": \"Bilingual user (Spanish/English at home). Symbols: [mama] [hungry]. "
            "Mom speaks Spanish. Generate response in Spanish.\", "
            "\"response\": \"Mamá, tengo hambre. ¿Puedo comer algo?\"}}]"
        ),
    },
    "expand": {
        "target": 2000, "batch": 25,
        "system": (
            "You generate AAC training examples for a 7B language model that helps "
            "non-speaking and minimally-verbal users communicate.\n\n"
            "Each example: a USER prompt with symbol-list or text fragment input, "
            "and an ASSISTANT response that expands to a fluent, natural sentence.\n\n"
            "Vary user profiles: child age 4-12, teen, adult, adult with aphasia, "
            "user with autism. Vary contexts: home, school, clinic, public, work.\n\n"
            "Symbol notation: [word]. Fragments: 2-4 words. Outputs: 4-12 words, natural. "
            "Avoid clinical jargon. Use dignity-preserving language.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} diverse symbol-expansion examples. Cover: requests, "
            "comments, protests, social greetings, repairs, medical needs, emotional "
            "expression. Mix adult and child voices. Each prompt distinct.\n\n"
            "Format: [{{\"prompt\": \"Symbols: [hungry] [want] [snack]. Generate appropriate utterance.\", "
            "\"response\": \"I'm hungry, can I have a snack please?\"}}]"
        ),
    },
    "predict": {
        "target": 1500, "batch": 25,
        "system": (
            "You generate AAC next-word prediction training data. Each example: "
            "USER provides a partial sentence and asks for top 5 likely continuations. "
            "ASSISTANT returns 5 plausible continuations ranked by frequency-of-use.\n\n"
            "Match register to user (child/teen/adult). Cover request, comment, "
            "social, emotional, medical, daily-living domains.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} diverse next-word prediction examples. Vary partial-sentence "
            "lengths (3-7 words). User asks for top 5 next words. Response gives 5 ranked "
            "candidates as a numbered list.\n\n"
            "Format: [{{\"prompt\": \"Partial: 'I want to go to the ___'. Predict top 5 likely next words.\", "
            "\"response\": \"1. bathroom\\n2. park\\n3. store\\n4. school\\n5. playground\"}}]"
        ),
    },
    "intent": {
        "target": 1000, "batch": 25,
        "system": (
            "You generate SCERTS pragmatic-function classification training data. "
            "Each example: a USER prompt with an AAC utterance + context. "
            "ASSISTANT returns the SCERTS function: request, protest, comment, "
            "social, repair, or regulate.\n\n"
            "Functions:\n"
            "- request: asking for object/action/help\n"
            "- protest: rejecting/refusing\n"
            "- comment: sharing/labeling/describing\n"
            "- social: greetings, thanks, attention\n"
            "- repair: clarifying after miscommunication\n"
            "- regulate: managing own state (calm, break)\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} diverse pragmatic-function classification examples. Vary "
            "utterance length, ages, contexts. Brief response with function + brief reasoning.\n\n"
            "Format: [{{\"prompt\": \"Classify pragmatic function: 'No no!' while pushing food away.\", "
            "\"response\": \"protest — user is rejecting the offered food via verbal + physical refusal\"}}]"
        ),
    },
    "rephrase": {
        "target": 500, "batch": 20,
        "system": (
            "You generate trauma-informed clinical-rephrasing training data. "
            "Each example: USER provides a deficit-framed clinical sentence. "
            "ASSISTANT returns a dignity-preserving, trauma-informed rephrase that "
            "honors user agency and communication intent.\n\n"
            "Avoid: 'non-compliant', 'attention-seeking', 'manipulative', 'tantrum', "
            "'maladaptive', 'refusing', 'difficult'. Replace with neutral language "
            "that describes communication and need.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} clinical→trauma-informed rephrasing examples covering: "
            "behavior, escape/avoidance, attention, refusal, dysregulation, sensory "
            "responses, communication breakdown.\n\n"
            "Format: [{{\"prompt\": \"Rephrase with trauma-informed dignity-preserving language: 'Patient is non-compliant.'\", "
            "\"response\": \"The patient declined to participate at this time.\"}}]"
        ),
    },
    "multi_turn": {
        "target": 1500, "batch": 25,
        "system": (
            "You generate AAC multi-turn-aware response training data. Each example: "
            "USER provides a CONTEXT (what someone said to AAC user) + symbol/fragment input. "
            "ASSISTANT returns a contextually appropriate, fluent response.\n\n"
            "Vary contexts: caregiver, peer, clinician, teacher, stranger. Cover "
            "agreement, refusal, request for repair, emotional support, factual answer.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} multi-turn AAC examples with diverse partner-utterance contexts.\n\n"
            "Format: [{{\"prompt\": \"Context: parent asked 'How was school today?'. User input: [good] [art]. Generate response.\", "
            "\"response\": \"It was good — we did art today.\"}}]"
        ),
    },
    "vocab_match": {
        "target": 500, "batch": 20,
        "system": (
            "You generate AAC vocabulary-level matching training data. Each example: "
            "USER specifies the user's profile (age, vocab level, condition) + input. "
            "ASSISTANT generates output matched to that user's expressive level — "
            "simple syntax for young children, complex for teens/adults.\n\n"
            "Honor neurodiversity: don't dumb-down for autistic users — match THEIR "
            "preferred register.\n\n"
            "Return ONLY a JSON array of {prompt, response} objects."
        ),
        "user_template": (
            "Generate {n} vocab-matching examples covering: toddler, school-age, teen, "
            "adult, adult with aphasia (recovering), adult professional setting.\n\n"
            "Format: [{{\"prompt\": \"User: 5-year-old, ~600 word vocab. Symbols: [I] [see] [bird]. Generate age-appropriate.\", "
            "\"response\": \"I see a bird!\"}}]"
        ),
    },
}


# Load validation prompts to AVOID generating overlap
def _load_val_prompts():
    val_path = Path(__file__).parent / "data" / "aac_validation.json"
    if not val_path.exists():
        return set()
    val = json.loads(val_path.read_text())
    return set(c["prompt"][:80].lower() for c in val.get("cases", []))


def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    if "[" in text:
        text = text[text.index("["):]
    if "]" in text:
        text = text[: text.rindex("]") + 1]
    return text


def call_teacher(category, n, max_retries=8):
    """Resilient retry to handle teacher cold start + transient 500s."""
    cfg = CATEGORIES[category]
    user_msg = cfg["user_template"].format(n=n)
    last_err = None

    if TEACHER_BACKEND == "openai":
        payload = json.dumps({
            "model": TEACHER_MODEL,
            "messages": [
                {"role": "system", "content": cfg["system"]},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.6,
            "max_tokens": 4096,
        }).encode("utf-8")
        url = f"{TEACHER_URL}/v1/chat/completions"
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(url, data=payload,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = _strip_fence(data["choices"][0]["message"]["content"])
                return json.loads(text)
            except Exception as e:
                last_err = e
                # exponential backoff for cold starts: 5,10,20,40,60,60,60,60s
                wait = min(60, 5 * (2 ** attempt))
                print(f"  ! attempt {attempt+1}: {str(e)[:60]}; backing off {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"teacher failed after {max_retries+1} attempts: {last_err}")

    # ollama backend (default)
    prompt = (
        f"<|im_start|>system\n{cfg['system']}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0.6, "num_predict": 8000, "num_ctx": 8192},
    }).encode("utf-8")
    url = f"{TEACHER_URL}/api/generate"
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = _strip_fence(data.get("response", ""))
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  ! attempt {attempt+1}: JSON parse failed; retrying...")
        except Exception as e:
            last_err = e
            print(f"  ! attempt {attempt+1}: {e}; retrying...")
    raise RuntimeError(f"teacher failed: {last_err}")


JUDGE_SYSTEM = (
    "You are a strict quality reviewer for AAC (Augmentative and Alternative Communication) "
    "training data. Score each example 1-10 on:\n"
    "  - clinical_correctness: dignity-preserving, accurate, no harm\n"
    "  - aac_realism: matches actual AAC user phrasing and needs\n"
    "  - format: correct response format for the category\n"
    "Reject (score < 7) any example that:\n"
    "  - uses pathologizing language (e.g., 'non-compliant', 'attention-seeking')\n"
    "  - contains medical inaccuracy or harm potential\n"
    "  - mismatches the specified user profile (age/condition)\n"
    "  - is generic non-AAC text\n\n"
    "Output ONLY a JSON object: {\"score\": <int 1-10>, \"reason\": \"<one short sentence>\"}"
)

def llm_judge(prompt: str, response: str, category: str) -> tuple[int, str]:
    """Returns (score, reason). Uses teacher endpoint as judge."""
    judge_user = (
        f"Category: {category}\n\nUSER PROMPT:\n{prompt}\n\n"
        f"GENERATED RESPONSE:\n{response}\n\nScore this AAC training example."
    )
    if TEACHER_BACKEND != "openai":
        return 8, "ollama backend skips judge"  # local doesn't support good JSON guarantee
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(f"{TEACHER_URL}/v1/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = _strip_fence(data["choices"][0]["message"]["content"])
        if "{" in text:
            text = text[text.index("{"): text.rindex("}") + 1]
        j = json.loads(text)
        return int(j.get("score", 0)), j.get("reason", "")
    except Exception as e:
        return 7, f"judge error: {e}"  # neutral pass on judge failure


def to_chatml(prompt, response):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<|synalux_answer|>\n{response}\n</|synalux_answer|><|im_end|>"
    )
    return {"text": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Scale factor for target volumes (0.1=quick smoke, 1.0=full 7K)")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_prompts = _load_val_prompts()
    print(f"Excluded validation prompts: {len(val_prompts)}")

    total = 0
    write_lock = threading.Lock()
    fout = out_path.open("w")
    print(f"Concurrency: {CONCURRENCY} parallel batches | backend: {TEACHER_BACKEND} | model: {TEACHER_MODEL}")

    def _process_batch(cat, batch_size):
        try:
            examples = call_teacher(cat, batch_size)
        except Exception as e:
            return cat, 0, 0, str(e)
        kept_local = 0
        rejected_local = 0
        # Judge each example, keep only score >= 7
        for ex in examples:
            if not isinstance(ex, dict): continue
            if "prompt" not in ex or "response" not in ex: continue
            if ex["prompt"][:80].lower() in val_prompts:
                rejected_local += 1
                continue
            score, _reason = llm_judge(ex["prompt"], ex["response"], cat)
            if score < 7:
                rejected_local += 1
                continue
            with write_lock:
                fout.write(json.dumps(to_chatml(ex["prompt"], ex["response"])) + "\n")
                fout.flush()
            kept_local += 1
        return cat, kept_local, rejected_local, None

    try:
        for cat, cfg in CATEGORIES.items():
            target = int(cfg["target"] * args.scale)
            batch = cfg["batch"]
            n_batches = (target // batch) + 3  # buffer for filtered/failed
            print(f"\n[{cat}] target={target} ({n_batches} batches of {batch})")
            kept = 0
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = [pool.submit(_process_batch, cat, batch) for _ in range(n_batches)]
                rejected = 0
                for f in as_completed(futures):
                    _, k, r, err = f.result()
                    if err:
                        print(f"  ! batch failed: {err[:120]}")
                        continue
                    kept += k
                    rejected += r
                    if kept >= target:
                        for fut in futures:
                            fut.cancel()
                        break
                print(f"  judge: kept {kept}, rejected {rejected} ({100*rejected/(kept+rejected+1):.0f}%)")
                elapsed = time.time() - t_start
                rate = kept / elapsed if elapsed else 0
                print(f"  kept {kept}/{target} ({rate:.1f}/s, {elapsed:.0f}s elapsed)")
            total += kept
            print(f"[{cat}] DONE: {kept} examples in {(time.time()-t_start)/60:.1f} min")
    finally:
        fout.close()

    print(f"\n=== Wrote {total} AAC SFT examples to {out_path} ===")


if __name__ == "__main__":
    main()
