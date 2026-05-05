"""Modal H100 v18 SFT — platform model: AAC + Video + offline-first TTS + multimodal routing.

Strategy (vs v17):
  - Same starting point: clean Qwen2.5-Coder-7B-Instruct (BF16, transformers-loadable)
  - Platform corpus: train_v18.jsonl ≈ 36K rows
      = v17 corpus (13.7K) — AAC text_correct, emergency, caregiver, tone, ask_ai
        + external glaive 6K + hermes 4K (Apache-2.0, function-calling polish)
        + caregiver_parse_extra ×2 boost (~3K) — addresses v17 caregiver
          regression (57.1% on held-out, target 85%)
        + video_script_gen 1.5K — Video Composer module (storyboard JSON)
        + tts_ssml_control 1.5K — offline-first SSML for Kokoro / Piper /
          espeak with online_premium fallback (Azure Neural)
        + voice_persona_pick 0.5K — five-class persona selection
        + word_predict_aac 2K — 3K+ dictionary, multilingual top-5
        + multimodal_tool_route 1.5K — analyze_camera_frame, classify_noise_event,
          analyze_voice_segment function calling + abstention
  - DoRA rank=256 across all 7 projections (unchanged from v17)
  - 6000 iters, LR=1.5e-5, cosine schedule
      * v17 ran 3000 iters at 2e-5 — bigger corpus needs more steps but
        slightly lower LR to avoid overshoot
  - IN-CONTAINER RELIABILITY GATE — adapter saved only if ALL pass
    (the 5 v17 gates + 5 new platform gates = 10 total):
      * BFCL probe >= 80%
      * AAC text_correct held-out >= 89%
      * Emergency Q&A held-out = 5/5 (HARD)
      * Caregiver held-out >= 85% (HARD — primary v18 target)
      * Tone-switch held-out >= 80%
      * video_script_valid >= 75% — JSON parses + ≥3 scenes
      * tts_ssml_valid >= 80% — XML parses, has prosody/emphasis, engine-correct tags
      * voice_persona_correct >= 80% — picks gold persona id
      * word_predict_valid >= 80% — JSON top-5 with no dups, single words
      * multimodal_route_correct >= 80% — right tool or correct abstention
  - Otherwise log fail in gate.json; adapter still saved for inspection

Local launch:
  cd /Users/admin/prism/training
  source venv/bin/activate
  python3 merge_v18_data.py --out data/train_v18.jsonl
  modal volume put prism-sft-data data/train_v18.jsonl /train_v18.jsonl --force
  modal run --detach modal_v18_sft.py::train

Fetch adapter when done:
  modal run modal_v18_sft.py::fetch
"""
import json
import os
from pathlib import Path

import modal

app = modal.App("prism-v18-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "trl==0.12.2",
        "peft==0.13.2",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub",
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)

# Volumes:
#   prism-sft-data   - holds train_v18.jsonl (uploaded by client)
#   prism-v18        - holds the trained adapter + gate report
data_vol = modal.Volume.from_name("prism-sft-data", create_if_missing=True)
out_vol = modal.Volume.from_name("prism-v18", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


@app.function(
    image=image,
    gpu="H100",
    timeout=21600,  # 6h — bigger corpus + more steps than v17
    cpu=8.0,
    memory=64000,
    volumes={"/data": data_vol, "/out": out_vol},
)
def run_sft_with_gate():
    """SFT then in-container reliability gate. Save adapter only if all gates pass."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print(f"=== v18 SFT starting; base = {BASE_MODEL} ===")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print(f"loaded {BASE_MODEL}; params={model.num_parameters():_}")

    train_path = Path("/data/train_v18.jsonl")
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} missing — upload first with:\n"
            "  modal volume put prism-sft-data train_v18.jsonl /train_v18.jsonl --force"
        )
    rows = []
    with train_path.open() as f:
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
    print(f"loaded examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    lora = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir="/out/v18_run",
        num_train_epochs=1,
        # 6000 steps × batch 2 × grad_accum 8 = 96,000 example-forwards over
        # ~36K rows ≈ 2.7 epochs. Lower per-epoch count than v17 (3.5) because
        # the corpus is more heterogeneous (external function-calling +
        # platform categories) and we want to avoid overfitting to glaive/
        # hermes/SSML formatting quirks.
        max_steps=6000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tok,
        args=cfg,
    )
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"train done in {train_secs:.0f}s")

    # Always save adapter (gates run after; user can choose to deploy or not)
    adapter_dir = "/out/v18_adapter"
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    out_vol.commit()
    print(f"adapter saved to {adapter_dir}")

    gates = _run_reliability_gate(model, tok)
    gates["train_secs"] = round(train_secs, 1)
    Path("/out/v18_gate.json").write_text(json.dumps(gates, indent=2))
    out_vol.commit()

    if not gates["pass"]:
        print(f"⚠️  RELIABILITY GATE FAILED: {gates['failures']}")
        print("Adapter saved but should NOT be deployed to production.")
    else:
        print("✅ All reliability gates passed.")

    return gates


# ── In-container held-out evals ─────────────────────────────────────────────
#
# Identical structure to v17 — same cases, same thresholds. Caregiver target
# stays at 0.85 (the v17 regression is the primary thing v18 must fix).

_BFCL_HELDOUT = [
    {
        "tools": [{"type": "function", "function": {"name": "add", "description": "Add two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}],
        "user": "What is 17 plus 26?",
        "expect_call": "add",
        "expect_args": {"a": 17, "b": 26},
    },
    {
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "What's the weather in Paris?",
        "expect_call": "get_weather",
        "expect_args": {"city": "Paris"},
    },
    {
        "tools": [{"type": "function", "function": {"name": "convert_currency", "description": "Convert currency", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}}}],
        "user": "How much is 100 dollars in euros?",
        "expect_call": "convert_currency",
        "expect_args_subset": {"from_currency": "USD", "to_currency": "EUR"},
    },
    {
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "Tell me a joke about clouds.",
        "expect_call": None,
    },
    {
        "tools": [{"type": "function", "function": {"name": "search_database", "description": "Search internal database", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
        "user": "What's the meaning of life?",
        "expect_call": None,
    },
]

_AAC_TEXT_CORRECT_HELDOUT = [
    {"input": "i wantto eat bowlof rice", "expected": "I want to eat bowl of rice"},
    {"input": "imhungry can i hav water", "expected": "I'm hungry, can I have water"},
    {"input": "ineedto goto bathroom", "expected": "I need to go to bathroom"},
    {"input": "can yu help me please", "expected": "Can you help me please"},
    {"input": "iwant to playoutside", "expected": "I want to play outside"},
    {"input": "i ned hlp now", "expected": "I need help now"},
    {"input": "wer is mom", "expected": "Where is mom"},
    {"input": "Hello, how are you?", "expected": "Hello, how are you?"},
    {"input": "ican't breathe", "expected": "I can't breathe"},
]

_TEXT_CORRECT_SYS = (
    "You are a fast text-cleanup engine for an AAC (augmentative and alternative communication) "
    "app used by users with motor impairments. Fix obvious typos, missing spaces, dropped letters, "
    "transposed letters. Fix voice-transcript word-boundary errors. Capitalize 'I' and the first word. "
    "DO NOT rewrite the user's voice. DO NOT add or remove content. DO NOT translate. "
    "Return ONLY the corrected text, no quotes, no explanation, no preamble."
)

_EMERGENCY_HELDOUT = [
    {"q": "Where is the patient?", "accept": ["123 Oak Street", "Springfield"]},
    {"q": "How old is the patient?", "accept": ["8", "eight"]},
    {"q": "What happened?", "accept": ["cannot breathe", "can't breathe", "breath"]},
    {"q": "Does the patient have allergies?", "accept": ["penicillin"]},
    {"q": "Is the patient verbal?", "accept": ["nonverbal", "non-verbal", "AAC"]},
]

_CAREGIVER_HELDOUT = [
    {
        "note": "Add 'I feel sick' to Help",
        "expect_type": "add_phrase",
        "expect_text": "i feel sick",
        "expect_category": "help",
    },
    {
        "note": "Remove the phrase 'I want candy' from Food",
        "expect_type": "remove_phrase",
        "expect_text": "i want candy",
        "expect_category": "food",
    },
    {
        "note": "Move 'bathroom' to top of Help category",
        "expect_type": "reorder_phrase",
        "expect_category": "help",
    },
    {
        "note": "Add 'I'm tired' to Feelings",
        "expect_type": "add_phrase",
        "expect_text": "i'm tired",
        "expect_category": "feelings",
    },
    {
        "note": "Tom had a rough morning today, lots of meltdowns before breakfast",
        "expect_type": "note_only",
    },
    {
        "note": "Boost the word 'please' so it shows higher",
        "expect_type": "boost_word",
        "expect_text": "please",
    },
]

_CAREGIVER_SYS = (
    "You are an AAC app configuration assistant. Parse the caregiver instruction into a JSON array "
    "of action objects. Return ONLY the JSON array. Action types: add_phrase {categoryId, text}, "
    "remove_phrase {phraseText, categoryId}, reorder_phrase {phraseId, newSortOrder, categoryId}, "
    "add_category {name, icon}, add_sequence {name, categoryId, steps}, remove_sequence {sequenceName}, "
    "boost_word {word, boostCount}, note_only {} for clinical observations."
)

_TONE_HELDOUT = [
    {"input": "I can't breathe please help", "accept": ["urgent"]},
    {"input": "I'm so sad today", "accept": ["gentle"]},
    {"input": "What time is my next medication?", "accept": ["clinical"]},
    {"input": "Let's play with the blocks!", "accept": ["playful"]},
    {"input": "I want water please", "accept": ["neutral", "gentle"]},
    {"input": "fire fire help", "accept": ["urgent"]},
    {"input": "tell me a joke", "accept": ["playful"]},
]

_TONE_SYS = (
    "You are an AAC response engine. Pick the appropriate tone register and prefix your response with "
    "a tone tag in <tone:gentle|urgent|clinical|playful|neutral> form. Pick urgent for emergencies, "
    "gentle for distress without emergency, clinical for medical/factual, playful for fun/social, "
    "neutral for everything else. The synalux portal strips the tone tag before delivery. "
    "After the tag, give a brief, AAC-appropriate response."
)

_EMERGENCY_SCRIPT = (
    "Name: Alex Garcia, Age: 8, Location: 123 Oak Street, Springfield, Room 4. "
    "Conditions: epilepsy. Allergies: penicillin. Medications: Keppra 250mg. "
    "Callback: 555-0123. Patient is nonverbal, uses AAC."
)

_EMERGENCY_SYS_TPL = (
    "You are an emergency-response AI on a phone call. Answer questions from a 911 operator using "
    "ONLY the facts in the script. If a fact is not in the script, say \"I don't have that information.\" "
    "Be concise. One sentence per answer.\n\nEmergency script:\n{script}"
)

_BFCL_SYS_TPL = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

# ─── v18 platform gates ────────────────────────────────────────────────────

_VIDEO_SYS = (
    "You are a BCBA / SLP authoring a visual social story for a child who uses AAC. "
    "Output a JSON object with exactly these keys: \"title\" (string, 3-10 words), "
    "\"scenes\" (array of 4-7 objects), and \"total_duration_s\" (integer, sum of "
    "scene durations). Each scene MUST have: \"caption\" (≤8 words), \"narration\" "
    "(1-2 short sentences), \"image_prompt\" (≤20 words), \"duration_s\" (4-12). "
    "Return JSON only, no markdown."
)
_VIDEO_HELDOUT = [
    "Topic: doctor visit\nTarget age: 5-7\nKey skill: emotional regulation\n\nWrite the storyboard JSON now.",
    "Topic: fire drill at school\nTarget age: 7-9\nKey skill: safety awareness\n\nWrite the storyboard JSON now.",
    "Topic: trying a new food\nTarget age: 3-5\nKey skill: sensory regulation\n\nWrite the storyboard JSON now.",
    "Topic: morning routine\nTarget age: 5-7\nKey skill: following routine\n\nWrite the storyboard JSON now.",
]

_SSML_SYS = (
    "You are an SSML author for an AAC speech engine that runs OFFLINE-FIRST "
    "(Kokoro / Piper / espeak) with an ONLINE fallback to Azure Neural. The input "
    "names which engine; produce SSML that engine can render. For offline_neural use "
    "<prosody rate/pitch/volume>, <break time>, <emphasis> only. For online_premium "
    "you may also use <mstts:express-as> and numeric prosody. Output SSML only."
)
_SSML_HELDOUT = [
    {"text": "I see you're feeling sad. It's okay.", "persona": "caregiver_warm", "emotion": "gentle", "engine": "offline_neural"},
    {"text": "Fire alarm. Walk to the field now.", "persona": "urgent_alert", "emotion": "urgent", "engine": "offline_neural"},
    {"text": "Once upon a time, a fox lived at the edge of a forest.", "persona": "narrator_calm", "emotion": "calm", "engine": "online_premium"},
    {"text": "I'm proud of you for asking for a break.", "persona": "caregiver_warm", "emotion": "encouraging", "engine": "online_premium"},
    {"text": "Time for medication. One tablet with water.", "persona": "clinical_neutral", "emotion": "neutral", "engine": "offline_neural"},
]
_SSML_OFFLINE_FORBIDDEN = ("phoneme", "voice", "audio", "mark", "say-as", "sub", "express-as")

_PERSONA_SYS = (
    "You are a voice-selection module for an AAC speech engine. Pick exactly one "
    "persona id from: caregiver_warm, child_excited, narrator_calm, urgent_alert, "
    "clinical_neutral. Return ONLY the persona id."
)
_PERSONA_HELDOUT = [
    {"context": "Adult comforting a child after a meltdown.", "utt": "I'm here. Take your time.", "gold": "caregiver_warm"},
    {"context": "Child showing off a drawing.", "utt": "Look! It's a dinosaur with a hat!", "gold": "child_excited"},
    {"context": "Bedtime story narration.", "utt": "The moon rose quietly over the town.", "gold": "narrator_calm"},
    {"context": "Emergency 911 call from AAC user.", "utt": "I cannot breathe. Send help.", "gold": "urgent_alert"},
    {"context": "SLP confirming a phoneme target.", "utt": "Say sh. Now sh-oo.", "gold": "clinical_neutral"},
]

_WP_SYS = (
    "You are an AAC word-prediction engine. Given the language and a partial "
    "utterance ending with ___ (a single underscore-pair indicating the next word), "
    "return the 5 most likely next words. Output JSON: {\"top5\": [\"w1\",\"w2\",\"w3\",\"w4\",\"w5\"]}. "
    "Words MUST be in the requested language, single words, no duplicates."
)
_WP_HELDOUT = [
    {"lang": "en", "prefix": "I want", "must_contain_any": ["to", "the", "a", "more"]},
    {"lang": "en", "prefix": "I am feeling", "must_contain_any": ["sad", "happy", "tired", "sick", "good"]},
    {"lang": "en", "prefix": "Where is", "must_contain_any": ["the", "mom", "dad", "my"]},
    {"lang": "en", "prefix": "Help me", "must_contain_any": ["please", "with", "now", "stand", "walk"]},
    {"lang": "es", "prefix": "Yo quiero", "must_contain_any": ["agua", "comer", "ir", "más", "ayuda"]},
    {"lang": "fr", "prefix": "Je veux", "must_contain_any": ["aller", "manger", "boire", "plus"]},
]

_MULTIMODAL_TOOLS = [
    {"type": "function", "function": {"name": "analyze_camera_frame", "description": "Analyze the most recent camera frame.", "parameters": {"type": "object", "properties": {"frame_id": {"type": "string"}, "target": {"type": "string", "enum": ["hand", "eye", "face", "scene"]}}, "required": ["frame_id", "target"]}}},
    {"type": "function", "function": {"name": "classify_noise_event", "description": "Classify a recent ambient sound.", "parameters": {"type": "object", "properties": {"audio_id": {"type": "string"}, "window_s": {"type": "number"}}, "required": ["audio_id", "window_s"]}}},
    {"type": "function", "function": {"name": "analyze_voice_segment", "description": "Analyze a captured voice segment.", "parameters": {"type": "object", "properties": {"audio_id": {"type": "string"}, "target": {"type": "string", "enum": ["emotion", "distress", "speaker_id"]}}, "required": ["audio_id", "target"]}}},
]
_MULTIMODAL_HELDOUT = [
    {"user": "Calibrate hand pointer using frame cam_001.", "expect_call": "analyze_camera_frame", "expect_args": {"frame_id": "cam_001", "target": "hand"}},
    {"user": "There was a loud noise — what was it? Audio mic_404, last 8 seconds.", "expect_call": "classify_noise_event", "expect_args": {"audio_id": "mic_404", "window_s": 8}},
    {"user": "Did the user sound distressed in segment mic_77?", "expect_call": "analyze_voice_segment", "expect_args": {"audio_id": "mic_77", "target": "distress"}},
    {"user": "Describe the room. Frame cam_42.", "expect_call": "analyze_camera_frame", "expect_args": {"frame_id": "cam_42", "target": "scene"}},
    {"user": "Translate 'water' into French.", "expect_call": None},
    {"user": "Tell me a story about a fox.", "expect_call": None},
]


def _generate(model, tok, system: str, user: str, max_new: int = 200) -> str:
    import torch
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    chat.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant"
    idx = full.rfind(marker)
    if idx >= 0:
        full = full[idx + len(marker):]
    full = full.replace("<|im_end|>", "").strip()
    return full


def _normalize(s: str) -> str:
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


def _bfcl_passes(resp: str, case: dict) -> bool:
    import re
    expect_call = case.get("expect_call")
    has_tool = "<tool_call>" in resp
    if expect_call is None:
        return not has_tool
    if not has_tool:
        return False
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return False
    if obj.get("name") != expect_call:
        return False
    args = obj.get("arguments", {}) or {}
    if "expect_args" in case:
        for k, v in case["expect_args"].items():
            if str(args.get(k)).lower() != str(v).lower():
                return False
    if "expect_args_subset" in case:
        for k, v in case["expect_args_subset"].items():
            if str(args.get(k)).lower() != str(v).lower():
                return False
    return True


def _run_reliability_gate(model, tok):
    failures = []
    checks = {}

    # 1) BFCL probe — threshold unchanged from v17 (>= 80%)
    bfcl_passes = 0
    for case in _BFCL_HELDOUT:
        sys_p = _BFCL_SYS_TPL.format(tools_json=json.dumps(case["tools"], ensure_ascii=False))
        try:
            resp = _generate(model, tok, sys_p, case["user"], max_new=200)
        except Exception as e:
            resp = f"<error: {e}>"
        if _bfcl_passes(resp, case):
            bfcl_passes += 1
        else:
            print(f"  BFCL fail: user='{case['user'][:60]}' resp='{resp[:120]}'")
    bfcl_pct = bfcl_passes / len(_BFCL_HELDOUT)
    checks["bfcl"] = {"passed": bfcl_passes, "total": len(_BFCL_HELDOUT), "pct": round(bfcl_pct, 3)}
    print(f"  bfcl: {bfcl_passes}/{len(_BFCL_HELDOUT)} = {bfcl_pct*100:.1f}%")
    if bfcl_pct < 0.80:
        failures.append(f"bfcl {bfcl_pct*100:.1f}% < 80%")

    # 2) AAC text_correct — >= 89%
    tc_passes = 0
    for case in _AAC_TEXT_CORRECT_HELDOUT:
        try:
            resp = _generate(model, tok, _TEXT_CORRECT_SYS, f'Language: en. Input: "{case["input"]}"', max_new=80)
        except Exception:
            resp = ""
        nr, ne = _normalize(resp), _normalize(case["expected"])
        ok = nr == ne or (len(set(nr.split()) & set(ne.split())) / max(len(set(ne.split())), 1) >= 0.85)
        if ok:
            tc_passes += 1
        else:
            print(f"  text_correct fail: input='{case['input']}' resp='{resp[:80]}' expected='{case['expected']}'")
    tc_pct = tc_passes / len(_AAC_TEXT_CORRECT_HELDOUT)
    checks["text_correct"] = {"passed": tc_passes, "total": len(_AAC_TEXT_CORRECT_HELDOUT), "pct": round(tc_pct, 3)}
    print(f"  text_correct: {tc_passes}/{len(_AAC_TEXT_CORRECT_HELDOUT)} = {tc_pct*100:.1f}%")
    if tc_pct < 0.89:
        failures.append(f"text_correct {tc_pct*100:.1f}% < 89%")

    # 3) Emergency Q&A — must be 5/5
    em_passes = 0
    em_sys = _EMERGENCY_SYS_TPL.format(script=_EMERGENCY_SCRIPT)
    for case in _EMERGENCY_HELDOUT:
        try:
            resp = _generate(model, tok, em_sys, case["q"], max_new=80)
        except Exception:
            resp = ""
        nr = _normalize(resp)
        ok = any(_normalize(a) in nr for a in case["accept"])
        if ok:
            em_passes += 1
        else:
            print(f"  emergency fail: q='{case['q']}' resp='{resp[:80]}'")
    checks["emergency"] = {"passed": em_passes, "total": len(_EMERGENCY_HELDOUT), "pct": round(em_passes / len(_EMERGENCY_HELDOUT), 3)}
    print(f"  emergency: {em_passes}/{len(_EMERGENCY_HELDOUT)}")
    if em_passes < len(_EMERGENCY_HELDOUT):
        failures.append(f"emergency {em_passes}/{len(_EMERGENCY_HELDOUT)} (need 5/5 — HARD)")

    # 4) Caregiver note parsing — >= 85% (PRIMARY v18 TARGET; v17 hit 57%)
    cg_passes = 0
    for case in _CAREGIVER_HELDOUT:
        try:
            resp = _generate(model, tok, _CAREGIVER_SYS, case["note"], max_new=200)
        except Exception:
            resp = ""
        ok = False
        try:
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            if "[" in cleaned and "]" in cleaned:
                cleaned = cleaned[cleaned.index("["): cleaned.rindex("]") + 1]
            arr = json.loads(cleaned)
            if isinstance(arr, list) and arr:
                first = arr[0]
                t = first.get("type", "")
                payload = first.get("payload", {}) or {}
                if t == case["expect_type"]:
                    ok = True
                    if "expect_text" in case:
                        text_val = (payload.get("text") or payload.get("phraseText") or payload.get("word") or "").lower()
                        if case["expect_text"].lower() not in text_val:
                            ok = False
                    if "expect_category" in case:
                        cat_val = (payload.get("categoryId") or "").lower()
                        if cat_val != case["expect_category"].lower():
                            ok = False
        except Exception:
            ok = False
        if ok:
            cg_passes += 1
        else:
            print(f"  caregiver fail: note='{case['note'][:60]}' resp='{resp[:120]}'")
    cg_pct = cg_passes / len(_CAREGIVER_HELDOUT)
    checks["caregiver"] = {"passed": cg_passes, "total": len(_CAREGIVER_HELDOUT), "pct": round(cg_pct, 3)}
    print(f"  caregiver: {cg_passes}/{len(_CAREGIVER_HELDOUT)} = {cg_pct*100:.1f}%")
    if cg_pct < 0.85:
        failures.append(f"caregiver {cg_pct*100:.1f}% < 85%  ⚠️ v18 PRIMARY TARGET")

    # 5) Tone-switch — >= 80%
    import re
    from xml.etree import ElementTree as ET
    tone_passes = 0
    for case in _TONE_HELDOUT:
        try:
            resp = _generate(model, tok, _TONE_SYS, case["input"], max_new=80)
        except Exception:
            resp = ""
        m = re.search(r"<tone:(gentle|urgent|clinical|playful|neutral)>", resp)
        if m and m.group(1) in case["accept"]:
            tone_passes += 1
        else:
            print(f"  tone fail: input='{case['input']}' resp='{resp[:80]}' want={case['accept']}")
    tone_pct = tone_passes / len(_TONE_HELDOUT)
    checks["tone"] = {"passed": tone_passes, "total": len(_TONE_HELDOUT), "pct": round(tone_pct, 3)}
    print(f"  tone: {tone_passes}/{len(_TONE_HELDOUT)} = {tone_pct*100:.1f}%")
    if tone_pct < 0.80:
        failures.append(f"tone {tone_pct*100:.1f}% < 80%")

    # 6) Video script — >= 75% (JSON parses + ≥ 3 scenes)
    vs_passes = 0
    for user in _VIDEO_HELDOUT:
        try:
            resp = _generate(model, tok, _VIDEO_SYS, user, max_new=900)
        except Exception:
            resp = ""
        ok = False
        try:
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            if "{" in cleaned:
                cleaned = cleaned[cleaned.index("{"): cleaned.rindex("}") + 1]
            obj = json.loads(cleaned)
            scenes = obj.get("scenes", [])
            if isinstance(scenes, list) and len(scenes) >= 3:
                if all(isinstance(s, dict) and {"caption","narration","image_prompt","duration_s"} <= set(s.keys()) for s in scenes):
                    ok = True
        except Exception:
            ok = False
        if ok:
            vs_passes += 1
        else:
            print(f"  video_script fail: user='{user[:50]}' resp='{resp[:100]}'")
    vs_pct = vs_passes / len(_VIDEO_HELDOUT)
    checks["video_script"] = {"passed": vs_passes, "total": len(_VIDEO_HELDOUT), "pct": round(vs_pct, 3)}
    print(f"  video_script: {vs_passes}/{len(_VIDEO_HELDOUT)} = {vs_pct*100:.1f}%")
    if vs_pct < 0.75:
        failures.append(f"video_script {vs_pct*100:.1f}% < 75%")

    # 7) TTS SSML — >= 80% (XML parses, has prosody/emphasis, engine-correct tags)
    ssml_passes = 0
    for case in _SSML_HELDOUT:
        user = (
            f"Text: \"{case['text']}\"\nVoice persona: {case['persona']}\n"
            f"Emotion: {case['emotion']}\nEngine: {case['engine']}\n\nReturn the SSML now."
        )
        try:
            resp = _generate(model, tok, _SSML_SYS, user, max_new=400)
        except Exception:
            resp = ""
        ok = False
        s = resp.strip()
        s = re.sub(r"^```(?:xml|ssml)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        if s.lstrip().startswith("<speak"):
            parseable = s.replace("<speak ", "<speak xmlns:mstts='https://www.w3.org/2001/mstts' "
                                  if "xmlns:mstts" not in s else "<speak ", 1) if case["engine"] == "online_premium" else s
            try:
                ET.fromstring(parseable)
                if re.search(r"<(prosody|emphasis|mstts:express-as)\b", s):
                    if case["engine"] == "offline_neural":
                        if not any(re.search(rf"<(?:[a-z]+:)?{tag}\b", s) for tag in _SSML_OFFLINE_FORBIDDEN):
                            ok = True
                    else:
                        ok = True
            except ET.ParseError:
                ok = False
        if ok:
            ssml_passes += 1
        else:
            print(f"  tts_ssml fail: engine={case['engine']} resp='{resp[:120]}'")
    ssml_pct = ssml_passes / len(_SSML_HELDOUT)
    checks["tts_ssml"] = {"passed": ssml_passes, "total": len(_SSML_HELDOUT), "pct": round(ssml_pct, 3)}
    print(f"  tts_ssml: {ssml_passes}/{len(_SSML_HELDOUT)} = {ssml_pct*100:.1f}%")
    if ssml_pct < 0.80:
        failures.append(f"tts_ssml {ssml_pct*100:.1f}% < 80%")

    # 8) Voice persona — >= 80%
    persona_passes = 0
    persona_set = {"caregiver_warm", "child_excited", "narrator_calm", "urgent_alert", "clinical_neutral"}
    for case in _PERSONA_HELDOUT:
        user = f"Context: {case['context']}\nUtterance: {case['utt']}\n\nWhich persona?"
        try:
            resp = _generate(model, tok, _PERSONA_SYS, user, max_new=20)
        except Exception:
            resp = ""
        norm = resp.strip().lower().strip(".'\"`")
        picked = next((p for p in persona_set if p in norm), None)
        if picked == case["gold"]:
            persona_passes += 1
        else:
            print(f"  persona fail: gold={case['gold']} resp='{resp[:60]}'")
    persona_pct = persona_passes / len(_PERSONA_HELDOUT)
    checks["voice_persona"] = {"passed": persona_passes, "total": len(_PERSONA_HELDOUT), "pct": round(persona_pct, 3)}
    print(f"  voice_persona: {persona_passes}/{len(_PERSONA_HELDOUT)} = {persona_pct*100:.1f}%")
    if persona_pct < 0.80:
        failures.append(f"voice_persona {persona_pct*100:.1f}% < 80%")

    # 9) Word predict — >= 80% (valid JSON, 5 unique single words, includes any expected)
    wp_passes = 0
    for case in _WP_HELDOUT:
        user = f"Language: {case['lang']}\nPrefix: {case['prefix']} ___\n\nReturn the top-5 JSON now."
        try:
            resp = _generate(model, tok, _WP_SYS, user, max_new=80)
        except Exception:
            resp = ""
        ok = False
        try:
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            if "{" in cleaned:
                cleaned = cleaned[cleaned.index("{"): cleaned.rindex("}") + 1]
            obj = json.loads(cleaned)
            top5 = obj.get("top5", [])
            if isinstance(top5, list) and len(top5) == 5 and all(isinstance(w, str) and w.strip() for w in top5):
                if len({w.lower() for w in top5}) == 5 and all(len(w.split()) == 1 for w in top5):
                    if "must_contain_any" in case:
                        if any(w.lower() in {x.lower() for x in top5} for w in case["must_contain_any"]):
                            ok = True
                    else:
                        ok = True
        except Exception:
            ok = False
        if ok:
            wp_passes += 1
        else:
            print(f"  word_predict fail: prefix='{case['prefix']}' resp='{resp[:120]}'")
    wp_pct = wp_passes / len(_WP_HELDOUT)
    checks["word_predict"] = {"passed": wp_passes, "total": len(_WP_HELDOUT), "pct": round(wp_pct, 3)}
    print(f"  word_predict: {wp_passes}/{len(_WP_HELDOUT)} = {wp_pct*100:.1f}%")
    if wp_pct < 0.80:
        failures.append(f"word_predict {wp_pct*100:.1f}% < 80%")

    # 10) Multimodal tool route — >= 80% (right tool with right args, or correct abstain)
    mm_sys = (
        "You are Qwen, an AAC orchestrator. You may call functions to drive multimodal "
        "analysis tools. If the user request needs camera, audio, or voice analysis, "
        "call the appropriate tool. Otherwise answer directly without calling any tool.\n\n"
        "# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{json.dumps(_MULTIMODAL_TOOLS, ensure_ascii=False)}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    )
    mm_passes = 0
    for case in _MULTIMODAL_HELDOUT:
        try:
            resp = _generate(model, tok, mm_sys, case["user"], max_new=200)
        except Exception:
            resp = ""
        if _bfcl_passes(resp, case):
            mm_passes += 1
        else:
            print(f"  multimodal fail: user='{case['user'][:60]}' resp='{resp[:120]}'")
    mm_pct = mm_passes / len(_MULTIMODAL_HELDOUT)
    checks["multimodal_route"] = {"passed": mm_passes, "total": len(_MULTIMODAL_HELDOUT), "pct": round(mm_pct, 3)}
    print(f"  multimodal_route: {mm_passes}/{len(_MULTIMODAL_HELDOUT)} = {mm_pct*100:.1f}%")
    if mm_pct < 0.80:
        failures.append(f"multimodal_route {mm_pct*100:.1f}% < 80%")

    return {"pass": not failures, "failures": failures, "checks": checks}


@app.local_entrypoint()
def upload_data(local_path: str = "/Users/admin/prism/training/data/train_v18.jsonl"):
    """Upload merged training file to volume."""
    import subprocess
    p = Path(local_path)
    if not p.exists():
        print(f"ERROR: {p} not found. Run merge_v18_data.py --out data/train_v18.jsonl first.")
        return
    print(f"Uploading {p} ({p.stat().st_size/1e6:.1f} MB) to prism-sft-data:/train_v18.jsonl")
    subprocess.run(
        ["modal", "volume", "put", "prism-sft-data", str(p), "/train_v18.jsonl", "--force"],
        check=True,
    )
    print("upload complete")


@app.local_entrypoint()
def train():
    """Submit v18 SFT job to Modal H100. Run upload_data first."""
    print("Submitting v18 SFT to Modal H100...")
    result = run_sft_with_gate.remote()
    print(json.dumps(result, indent=2)[:3000])


@app.local_entrypoint()
def fetch(out_dir: str = "/Users/admin/prism/training/models/v18_modal"):
    """Pull v18 adapter from Modal Volume."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    print(f"fetching v18_adapter to {out_dir}...")
    subprocess.run(
        ["modal", "volume", "get", "prism-v18", "v18_adapter", out_dir, "--force"],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "get", "prism-v18", "v18_gate.json", f"{out_dir}/v18_gate.json", "--force"],
        check=False,
    )
    print(f"  done -> {out_dir}")


if __name__ == "__main__":
    print("Run via: modal run --detach modal_v18_sft.py::train")
