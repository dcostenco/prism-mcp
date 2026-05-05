"""Modal H100 — generate 3000+ AAC phrases per language using Qwen2.5-72B.

Strategy:
  1. First pass: generate the canonical 3000-phrase EN list (categorized).
  2. Per-language pass: each of the 14 locales gets its own H100:4 container
     that translates the EN list into native phrases. Locale-specific notes
     ensure idiomatic, not literal, translations.

Volume per language: 200 phrases × 15 categories = 3,000.
Per-call output: 100 phrases (so 2 calls per category, both for generation
and translation), keeping each response under the 4096-token max.

Categories (each ~200 phrases for total ~3000):
  - greetings_social        ("Hello, how are you?", "See you tomorrow", ...)
  - basic_needs             ("I'm thirsty", "Can I have water please")
  - food_eating             ("I want a sandwich", "I'm full now")
  - feelings_emotions       ("I feel happy today", "I'm a little nervous")
  - questions_requests      ("What time is it?", "Can you help me?")
  - school_learning         ("I finished my homework", "Can I read this?")
  - health_medical          ("My head hurts", "I need my medicine")
  - daily_activities        ("Time to brush teeth", "Let's go to the park")
  - family_relationships    ("Mom is coming home", "I miss grandma")
  - safety_emergency        ("Call 911", "I'm not safe", "I'm lost")  ← CRITICAL
  - time_calendar           ("Today is Monday", "See you next week")
  - choices_preferences     ("I prefer the blue one", "Maybe later")
  - clinical_aba_friendly   ("I need a break", "Can we use the visual schedule")
  - polite_communication    ("Excuse me", "Thank you very much")
  - clarification_repair    ("I didn't understand", "Can you say that again?")

Run:
  cd /Users/admin/prism/training
  modal run --detach modal_offline_phrases.py::main

Fetch results:
  modal run modal_offline_phrases.py::fetch
"""
import json
import os
import time
from pathlib import Path

import modal

app = modal.App("prism-offline-phrases")

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub==0.27.0",
    )
)
_hf = os.environ.get("HF_TOKEN", "").strip()
image = _image.env({"HF_TOKEN": _hf}) if _hf else _image

vol = modal.Volume.from_name("prism-offline-phrases", create_if_missing=True)

TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Per-language metadata for idiomatic translation
LANGS = {
    "en":      ("English",                          "Standard American English. Natural everyday speech."),
    "es":      ("Spanish",                           "Neutral Latin American Spanish. Use 'tú' (informal) for child speakers."),
    "fr":      ("French",                            "Standard metropolitan French. Use 'tu' for child speakers."),
    "pt":      ("Portuguese (Brazilian)",            "Brazilian Portuguese, informal. Common everyday phrasings."),
    "ro":      ("Romanian",                          "Standard Romanian. Use familiar form for child speakers."),
    "uk":      ("Ukrainian",                         "Modern Ukrainian (NOT surzhyk, NOT Russian). Cyrillic script."),
    "ru":      ("Russian",                           "Standard Russian. Cyrillic script. Informal/familiar form for child speakers."),
    "de":      ("German",                            "Standard Hochdeutsch. Use 'du' for child speakers."),
    "ja":      ("Japanese",                          "Casual everyday Japanese. Mix kanji+hiragana naturally. Child-friendly tone."),
    "ko":      ("Korean",                            "Standard Seoul Korean. Hangul. Polite '-요' form for child speakers."),
    "zh-Hans": ("Chinese Simplified (Mandarin)",     "Mainland Mandarin, Simplified characters (简体中文). Common everyday phrasing."),
    "zh-Hant": ("Chinese Traditional (Taiwan)",      "Taiwanese Mandarin with Traditional characters (繁體中文). Use Taiwan-specific vocabulary (e.g., 公車 not 公交车)."),
    "zh-HK":   ("Cantonese (Hong Kong)",             "Hong Kong Cantonese in Traditional characters (廣東話). Use HK-specific vocabulary (e.g., 巴士, 的士). Vocabulary differs from Taiwan Mandarin."),
    "ar":      ("Arabic (Modern Standard)",          "Modern Standard Arabic (MSA, فصحى). Right-to-left script. Natural conversational MSA."),
}

CATEGORIES = [
    # ── Greetings / social (split by speaker age) ──
    ("greetings_child",                "casual greetings + farewells from a child speaker, simple words"),
    ("greetings_adult_formal",         "polite/professional greetings, introductions, formal farewells"),
    ("social_smalltalk",               "weather, weekend plans, how-was-your-day chitchat"),
    # ── Basic needs (split by domain) ──
    ("needs_thirst_hunger",            "I'm thirsty / hungry / want water / need a snack — only food+drink needs"),
    ("needs_bathroom",                 "bathroom requests, urgency, hygiene-related needs"),
    ("needs_rest_comfort",             "I'm tired, need a break, need a hug, want my blanket, want to sit down"),
    # ── Food / eating ──
    ("food_preferences",               "I like / I don't like / favorite food, allergies, what's for dinner"),
    ("food_meals_restaurant",          "ordering at a restaurant, asking for a menu, kid meals, drive-thru"),
    # ── Feelings (split positive / negative / overwhelmed) ──
    ("feelings_positive",              "happy, excited, calm, proud, grateful, in love, content"),
    ("feelings_negative",              "sad, angry, frustrated, scared, jealous, lonely, embarrassed"),
    ("feelings_overwhelmed",           "sensory overload, meltdown signals, too loud, too bright, need to leave"),
    # ── Questions / requests ──
    ("questions_basic_wh",             "what is, who is, where is, simple wh-questions"),
    ("questions_complex",              "when, why, how, would you, could you, asking for explanation"),
    ("requests_permission",            "may I, can I have, is it okay if, asking permission politely"),
    # ── School / learning ──
    ("school_classroom",               "in the classroom — pay attention, question, listen, line up"),
    ("school_homework_learning",       "homework, studying, reading, writing, math problems"),
    ("school_social_peers",            "talking to classmates, group projects, recess, lunch with friends"),
    # ── Health / medical (split) ──
    ("health_pain",                    "head hurts, stomach hurts, throat hurts — all body-pain expressions"),
    ("health_sickness",                "fever, vomiting, dizzy, can't breathe well — illness symptoms"),
    ("health_doctor_meds",             "doctor appointments, medication reminders, dentist, allergy management"),
    # ── Daily activities (split by time of day) ──
    ("daily_morning_routine",          "wake up, brush teeth, get dressed, breakfast, school bus"),
    ("daily_evening_routine",          "dinner, bath, pajamas, story, bedtime"),
    ("daily_outings_activities",       "going to the park, store, friend's house, after-school"),
    # ── Family ──
    ("family_immediate",               "mom, dad, siblings — direct family, household interactions"),
    ("family_extended_friends",        "grandparents, aunts, uncles, cousins, close family friends"),
    # ── Safety / emergency (split — CRITICAL) ──
    ("safety_critical",                "uncancellable emergencies — abuse, breathing, can't move, lost, abduction. EXACT WORDING MATTERS — do not soften."),
    ("safety_urgent",                  "I need help, I'm scared, I want my mom, call dad — non-life-threatening but urgent"),
    ("safety_medical",                 "I fell, it hurts, I feel sick, I need my medicine, my [body part] is bleeding"),
    # ── Time / calendar ──
    ("time_clock_dates",               "telling time, days of week, months, today/tomorrow/yesterday"),
    ("time_scheduling_waiting",        "appointment is at X, waiting for, see you next week, in a minute"),
    # ── Choices / clinical / polite / repair ──
    ("choices_preferences",            "I prefer, I'd rather, my favorite, maybe later, definitely yes/no"),
    ("clinical_breaks_sensory",        "I need a break, the lights are too bright, can I use my fidget"),
    ("clinical_visual_supports",       "show me the picture, visual schedule, what's next, first-then board"),
    ("polite_thanks_sorry",            "thank you, you're welcome, sorry, excuse me, no thank you"),
    ("clarification_repair",           "didn't understand, please repeat, can you slow down, what does that mean"),
]


def _call(llm, tokenizer, sp, sys_p: str, user_p: str) -> list[str]:
    """Single vLLM call returning a parsed JSON array of strings."""
    chat = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    outs = llm.generate([prompt], sp)
    text = outs[0].outputs[0].text.strip()
    if "[" in text and "]" in text:
        text = text[text.index("["): text.rindex("]") + 1]
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            return []
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []


@app.function(
    image=image,
    gpu="H100:4",
    timeout=10800,
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
)
def gen_for_lang(lang_code: str, en_phrases: list[list[str]] | None = None):
    """Generate phrases for ONE language. If en_phrases given, translate them;
    otherwise generate the canonical EN list (only happens for 'en')."""
    import vllm
    from vllm import SamplingParams

    name, note = LANGS[lang_code]
    print(f"[{lang_code}] loading {TEACHER_MODEL}...", flush=True)
    llm = vllm.LLM(
        model=TEACHER_MODEL,
        dtype="bfloat16",
        max_model_len=12288,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=4096)
    tokenizer = llm.get_tokenizer()

    # Target: ~3,000-3,500 unique phrases per language with high diversity.
    # 35 narrow categories × 100 unique each. Each cat → 1 call of 150 raw
    # phrases (deduper keeps ~100). Narrow cat scope means within-cat
    # similarity stays low, so dedup yield is higher per call.
    PER_CAT_TOTAL = 150
    PER_CALL = 150
    n_calls_per_cat = 1

    out_by_cat: dict[str, list[str]] = {}

    for i, (cat, hint) in enumerate(CATEGORIES):
        cat_phrases: list[str] = []

        if en_phrases is None:
            # generate EN canonical — 2 calls per category, ask for distinct
            # subsets to maximize diversity instead of duplicate batches
            for batch_idx in range(n_calls_per_cat):
                already = cat_phrases[:]  # what's already generated for this cat
                sys_p = (
                    f"You generate AAC (augmentative and alternative communication) phrases. "
                    f"Category: {cat}. Topic: {hint}. "
                    "Phrases must be SHORT, natural everyday utterances 2-14 words long. "
                    "Cover children, teens, and adults. Mix declarative, interrogative, and imperative. "
                    "AAC users may have motor / cognitive / autism / cerebral palsy profiles — keep "
                    "phrasing simple and direct. SAFETY-CRITICAL phrases must be UNAMBIGUOUS. "
                    "Each phrase MUST be DISTINCT from the others — do NOT repeat. "
                    "Output ONLY a JSON array of phrase strings. No commentary, no fences."
                )
                avoid_block = ""
                if already:
                    sample = already[-min(50, len(already)):]
                    avoid_block = (
                        f"\n\nDo NOT repeat any of these {len(sample)} already-generated "
                        f"phrases — generate {PER_CALL} NEW distinct ones:\n"
                        f"{json.dumps(sample, ensure_ascii=False)}"
                    )
                user_p = (
                    f"Generate exactly {PER_CALL} distinct {cat} phrases (batch "
                    f"{batch_idx+1}/{n_calls_per_cat}). JSON array only.{avoid_block}"
                )
                arr = _call(llm, tokenizer, sp, sys_p, user_p)
                # dedupe across batches within this category
                seen = {p.lower() for p in cat_phrases}
                for p in arr:
                    if p.lower() not in seen:
                        cat_phrases.append(p)
                        seen.add(p.lower())
        else:
            # translate the EN phrases for this category — 2 calls (split EN into halves)
            cat_en = en_phrases[i]
            half = (len(cat_en) + 1) // 2
            chunks = [cat_en[:half], cat_en[half:]] if len(cat_en) > PER_CALL else [cat_en]
            for chunk in chunks:
                if not chunk:
                    continue
                sys_p = (
                    f"You translate AAC phrases from English into {name}. {note} "
                    "Translate IDIOMATICALLY (not literally). Keep phrases natural, short, and "
                    "appropriate for the same speaker (child/teen/adult). For SAFETY-CRITICAL "
                    "phrases (category: safety_emergency), translation MUST preserve the exact "
                    "intent — do not soften or paraphrase emergency signals. "
                    "Output ONLY a JSON array of strings, same length and order as input. "
                    "No commentary, no fences."
                )
                user_p = (
                    f"Translate these {len(chunk)} {cat} phrases into {name}. "
                    "Return JSON array of exactly the same length, in the same order.\n\n"
                    f"Input:\n{json.dumps(chunk, ensure_ascii=False)}"
                )
                arr = _call(llm, tokenizer, sp, sys_p, user_p)
                if len(arr) == len(chunk):
                    cat_phrases.extend(arr)
                else:
                    # length mismatch — keep what we got, pad with chunk fallback
                    print(f"[{lang_code}/{cat}] translation length mismatch: got {len(arr)} expected {len(chunk)}", flush=True)
                    cat_phrases.extend(arr or chunk)

        out_by_cat[cat] = cat_phrases
        print(f"[{lang_code}] {cat:25s}  {len(cat_phrases):3d} phrases", flush=True)

    # Save to volume
    out_path = Path(f"/vol/{lang_code}.json")
    out_path.write_text(json.dumps(out_by_cat, ensure_ascii=False, indent=2))
    vol.commit()
    total = sum(len(v) for v in out_by_cat.values())
    print(f"[{lang_code}] TOTAL {total} phrases -> /vol/{lang_code}.json", flush=True)
    return {"lang": lang_code, "total": total}


@app.local_entrypoint()
def main(only: str = ""):
    """Two-stage: (1) generate EN canonical, (2) translate it to all other langs in parallel.

    --only es,pt,ar  → translate ONLY these langs (skip stage 1 if en.json already on volume).
    """
    import subprocess

    requested = {c.strip() for c in only.split(",") if c.strip()} if only else set()

    # Stage 1 — only run if we don't have en.json already, OR if en is in requested
    en_local = "/tmp/en_phrases.json"
    has_en_local = False
    try:
        subprocess.run(
            ["modal", "volume", "get", "prism-offline-phrases", "en.json", en_local, "--force"],
            check=True, capture_output=True,
        )
        has_en_local = True
        print(f"Found existing en.json on volume — skipping Stage 1")
    except subprocess.CalledProcessError:
        has_en_local = False

    if not has_en_local or "en" in requested:
        print("Stage 1: generating EN canonical phrase list on H100:4...")
        en_result = gen_for_lang.remote("en", en_phrases=None)
        print(f"  en done: {en_result}")
        subprocess.run(
            ["modal", "volume", "get", "prism-offline-phrases", "en.json", en_local, "--force"],
            check=True,
        )

    en_by_cat = json.loads(open(en_local).read())
    en_phrases = [en_by_cat.get(cat, []) for cat, _ in CATEGORIES]

    other_langs = [c for c in LANGS if c != "en"]
    if requested:
        other_langs = [c for c in other_langs if c in requested]

    print(f"\nStage 2: translating to {len(other_langs)} languages in parallel on H100:4 each...")
    print(f"  langs: {other_langs}")
    results = list(gen_for_lang.starmap([(c, en_phrases) for c in other_langs]))
    print("\n=== summary ===")
    for r in results:
        print(f"  {r['lang']:10s}  {r['total']} phrases")


@app.local_entrypoint()
def fetch(out_dir: str = "/Users/admin/prism/training/data/offline_phrases"):
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    for code in LANGS:
        local = f"{out_dir}/{code}.json"
        try:
            subprocess.run(
                ["modal", "volume", "get", "prism-offline-phrases", f"{code}.json", local, "--force"],
                check=True, capture_output=True
            )
            n = sum(len(v) for v in json.loads(open(local).read()).values())
            print(f"  {code}.json  ({n} phrases)")
        except subprocess.CalledProcessError:
            print(f"  ! {code}.json not in volume")


if __name__ == "__main__":
    print("Run: modal run --detach modal_offline_phrases.py::main")
