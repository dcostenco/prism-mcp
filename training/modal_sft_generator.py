"""Modal H100 batch SFT data generator for prism-coder v16.

Generates training data via Qwen2.5-32B-Instruct in vLLM offline batch mode.
Uses parallel containers (.map) — far faster + simpler than the web_server pattern.

Categories generated:
  text_correct        1500   AAC noisy input -> corrected utterance
  emergency_qa         500   Emergency script Q -> factual answer
  caregiver_parse      500   Caregiver note -> JSON action array
  translate            500   English -> {Spanish/French/Russian/Japanese/etc}
  ask_ai_aac           500   AAC kid question -> simple 2-3 sentence answer
  word_predict_aac    1000   Partial sentence -> top-5 next words (AAC vocab)
  memory_checkout       80   Restore memory -> tool call (synalux gap)
  hipaa                 80   Compliance -> hipaa tool call (synalux gap)
  contrastive_extra    200   Re-balance ledger/forget pairs (more session_save_ledger,
                             more session_forget_memory examples to flip the bias)
  no_tool_extra        200   General programming Qs that LOOK like Prism keywords
                             but should abstain
  TOTAL              ~5060

Local launch:
  cd /Users/admin/prism/training
  modal run --detach modal_sft_generator.py::main
Fetch results when done:
  modal run modal_sft_generator.py::fetch
"""
import json
import os
import time
from pathlib import Path

import modal

app = modal.App("prism-sft-gen-72b")

# Build the image. Only inject HF_TOKEN if it's actually set — passing an
# empty string corrupts huggingface_hub auth and causes "Invalid repository ID"
# errors on otherwise-public models (which is what killed the v1 attempt).
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # vllm 0.6.6.post1 is the last stable that ships its own pinned
        # transformers/tokenizers compatible with Qwen2.5 series. Newer
        # vllm rewires the tokenizer plumbing and requires pin gymnastics.
        "vllm==0.6.6.post1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub==0.27.0",
    )
)
_hf_token = os.environ.get("HF_TOKEN", "").strip()
image = _image.env({"HF_TOKEN": _hf_token}) if _hf_token else _image

vol = modal.Volume.from_name("prism-sft-gen-72b", create_if_missing=True)

# Top open-source instruction model — quality-first per "zero error assumption"
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ── System prompts for each generation category ─────────────────────────────

GEN_SPECS = {
    "text_correct": {
        "target": 1500, "batch": 30,
        "system": (
            "You generate AAC text-correction training examples. Each example: a malformed AAC user "
            "input (typos, missing spaces, voice-transcript word boundaries, hurried punctuation) and "
            "the most-likely intended utterance. Cover: word-boundary fixes (\"bowlof rice\"->\"bowl of rice\"), "
            "dropped letters, transposed letters, spurious commas, voice 'umm'/'uhh' fillers, missing apostrophes. "
            "Include emergency phrases too — those MUST be preserved verbatim, just with correct spelling. "
            "Mix English and a sprinkle of Spanish/French inputs. "
            "Return ONLY a JSON array of {input, lang, expected} objects — no explanation, no fences."
        ),
        "user": "Generate {n} diverse AAC text-correction examples. About 70% English, 15% Spanish, 10% French, 5% emergency phrases (preserve exactly).",
    },
    "emergency_qa": {
        "target": 500, "batch": 25,
        "system": (
            "You generate emergency-phone-call Q&A training data for an AAC emergency-response AI. "
            "Each example: a 911 operator's question + a CONTEXT BLOCK containing the emergency script "
            "(name, age, location, medical conditions, allergies, medications, callback). The answer must "
            "extract ONLY facts present in the script. If the question asks something not in the script, "
            "the answer must be \"I don't have that information.\" "
            "Vary scripts: child/teen/adult, different conditions (epilepsy, asthma, diabetes, autism, "
            "cardiac, anaphylaxis), different locations (home, school, hospital, public). "
            "Vary questions: location, age, what happened, allergies, meds, conditions, callback, who's "
            "with the patient, what time, who is calling, AAC vs verbal, address, etc. "
            "Return ONLY a JSON array of {script, question, answer} objects."
        ),
        "user": "Generate {n} emergency Q&A examples. Each must include the full script context plus one operator question and the correct extracted answer.",
    },
    "caregiver_parse": {
        "target": 500, "batch": 25,
        "system": (
            "You generate caregiver-note parsing training examples for an AAC app. Each: a natural-language "
            "caregiver/BCBA instruction + the structured JSON action array it should parse to. "
            "Action types: add_phrase {categoryId, text}, remove_phrase {phraseText, categoryId}, "
            "reorder_phrase {phraseId, newSortOrder, categoryId}, add_category {name, icon}, "
            "add_sequence {name, categoryId, steps:[{label, options:[string]}]}, remove_sequence {sequenceName}, "
            "boost_word {word, boostCount}, note_only {} (for clinical observations). "
            "Available categoryIds: help, food, feelings, school, quick, animals, colors, body, time. "
            "Return ONLY a JSON array of {note, actions} objects — actions is the JSON array of action objects."
        ),
        "user": "Generate {n} diverse caregiver note examples. Cover all 8 action types. Mix simple single-action and complex multi-action notes. Include note_only cases for pure observations.",
    },
    "translate": {
        "target": 500, "batch": 25,
        "system": (
            "You generate translation training examples for an AAC app. Each: a short AAC utterance "
            "(2-10 words) + source language + target language + ONE acceptable translation. "
            "Source = English most often; targets must cover all 12 supported languages: Spanish, French, "
            "Portuguese, Romanian, Ukrainian, Russian, German, Japanese, Korean, Chinese, Arabic. "
            "Also include reverse-direction (Spanish->English, etc) and a few cross-pairs. "
            "Use AAC-typical content: requests, feelings, basic needs, social greetings, emergency phrases. "
            "Return ONLY a JSON array of {text, fromLang, toLang, translation} objects."
        ),
        "user": "Generate {n} translation examples covering all 12 target languages roughly evenly. Keep utterances short and AAC-typical.",
    },
    "ask_ai_aac": {
        "target": 500, "batch": 25,
        "system": (
            "You generate AAC-helper Q&A training examples. The AI is a friendly helper for a child who "
            "uses an AAC device. Responses MUST be 2-3 short sentences, simple words, encouraging tone. "
            "Cover: math (single-digit + and -), science basics (animals, weather, body, colors), social "
            "questions (manners, feelings, family), school (counting, letters, shapes), safety (don't "
            "talk to strangers, hot/cold). Use simple vocabulary appropriate for ages 4-10. "
            "Return ONLY a JSON array of {question, answer} objects."
        ),
        "user": "Generate {n} diverse kid-AAC questions and 2-3 sentence answers. Keep it kind, simple, encouraging.",
    },
    "word_predict_aac": {
        "target": 1000, "batch": 30,
        "system": (
            "You generate word-prediction training data for AAC. Each: a partial sentence (with a blank "
            "marked '___') + the top 5 most-likely next words for an AAC user. Use AAC core vocabulary "
            "(I, want, need, like, more, please, help, mom, dad, eat, drink, go, play, stop, no, yes). "
            "Order from most likely to least likely. Vary user profile: child, teen, adult; different "
            "contexts (mealtime, school, play, medical, social). "
            "Return ONLY a JSON array of {prefix, top5} objects — top5 is a list of 5 single-word strings."
        ),
        "user": "Generate {n} word-prediction examples. Cover diverse AAC contexts. Predictions must be plausible AAC vocabulary.",
    },
    "memory_checkout": {
        "target": 80, "batch": 20,
        "system": (
            "You generate tool-call training examples for the SYNALUX memory_checkout tool. "
            "Tool: memory_checkout(project, version) — restores project memory to a specific version. "
            "Each example: a USER prompt naturally requesting memory restoration to a version, plus the "
            "correct tool call. Include <think> reasoning that explicitly DEBATES the tempting alternatives "
            "(session_restore_handoff, memory_history, session_load_context, session_checkout) and explains "
            "why memory_checkout is correct because the user named a SPECIFIC version number. "
            "Return ONLY JSON: [{prompt, think, args}]. args = {project, version}."
        ),
        "user": "Generate {n} diverse memory_checkout examples. Vary phrasings: 'restore to v3', 'roll back memory to version 2', 'go back to version 5 of prism-mcp', etc. Project names varied. Versions 1-15.",
    },
    "hipaa": {
        "target": 80, "batch": 20,
        "system": (
            "You generate tool-call training examples for the SYNALUX hipaa tool. "
            "Tool: hipaa(action, text, data, encrypted, event, user, client_id) — verifies HIPAA compliance, "
            "encrypts/decrypts PHI, audits PHI access, etc. Action values: verify, encrypt, decrypt, audit, log_access. "
            "Each example: a USER prompt about HIPAA/PHI/compliance, plus the correct tool call. "
            "Include <think> reasoning that explicitly debates session_health_check (tempting because both "
            "involve 'check') and explains why hipaa is correct because the topic is PHI/compliance. "
            "Return ONLY JSON: [{prompt, think, args}]. args is a subset of hipaa params."
        ),
        "user": "Generate {n} diverse hipaa tool examples. Vary actions, vary subjects (patient data, clinical notes, AAC user info, etc).",
    },
    "contrastive_extra": {
        "target": 200, "batch": 25,
        "system": (
            "You generate contrastive SFT examples to FIX bias in two confused tool pairs:\n"
            "(A) session_save_ledger vs session_save_experience — current model wrongly defaults to "
            "save_experience for 'record this work' phrasing. session_save_ledger is for END-OF-SESSION "
            "WORK LOGS (summary of what was done). session_save_experience is for STRUCTURED LEARNING "
            "EVENTS (correction/success/failure with context+action+outcome).\n"
            "(B) session_forget_memory vs knowledge_forget — current model wrongly defaults to "
            "knowledge_forget for 'remove memory about X'. session_forget_memory deletes a SPECIFIC "
            "memory entry (by id or query). knowledge_forget purges KNOWLEDGE entries by project/age.\n"
            "Mix BOTH directions: some prompts where the right tool is _ledger / _forget_memory, and some "
            "where the right tool is _experience / knowledge_forget. Each example must have a <think> "
            "block that names the wrong tool and explains the discriminating signal. "
            "Return JSON: [{prompt, think, right_tool, args}]."
        ),
        "user": "Generate {n} contrastive examples ~50/50 split between pair A and pair B, ~50/50 right-tool A vs right-tool B. Cover diverse phrasings.",
    },
    "no_tool_extra": {
        "target": 200, "batch": 25,
        "system": (
            "You generate NO-TOOL adversarial training examples. Each: a USER prompt that USES Prism "
            "tool keywords ('session', 'memory', 'forget', 'search', 'load', 'export', 'knowledge', "
            "'compact', 'health') in a GENERAL PROGRAMMING context — NOT a request to use Prism tools. "
            "ASSISTANT response: <think> trace names the tempting Prism tool, explains why it's tempting "
            "(the keyword), then identifies the discriminating signal (general programming question). "
            "Then a brief 1-2 sentence direct answer wrapped in <|synalux_answer|>. "
            "Cover: PHP session_start, Express middleware, BFS/DFS search, LSTM forget gates, "
            "PostgreSQL pg_dump, CSV export, RAG retrieval, garbage collection, system health checks, etc. "
            "Return JSON: [{prompt, think, answer}]."
        ),
        "user": "Generate {n} diverse NO-TOOL adversarial examples. Each must trick on a different keyword.",
    },
    # ── v16.1 corrective categories (target the 3 v16 regressions) ──
    "session_load_context_extra": {
        "target": 500, "batch": 20,
        "system": (
            "You generate tool-call training examples for the SYNALUX session_load_context tool — the "
            "BIGGEST and MOST IMPORTANT memory tool. session_load_context(project, level) loads the full "
            "context for a project at the start of a session. Triggered by phrases like: 'load context', "
            "'show me where we left off', 'pull up the project', 'resume work on X', 'what was our last "
            "state on Y', 'boot up the X context', 'get me up to speed', etc. "
            "<think> reasoning MUST debate tempting alternatives — especially memory_checkout (which "
            "restores to a SPECIFIC version, NOT the same as load_context which gets latest). Other "
            "tempting tools: session_search_memory, session_save_handoff, memory_history. The "
            "discriminating signal for session_load_context is: user wants the CURRENT/LATEST project "
            "state to start a new session, NOT a specific historical version. "
            "Args: {project: str, level?: 'shallow'|'deep'}. "
            "Return ONLY JSON: [{prompt, think, args}]."
        ),
        "user": "Generate {n} diverse session_load_context examples. Vary phrasings, project names, levels. CRITICAL: <think> must explicitly compare against memory_checkout (which is for version restore, not session start).",
    },
    "caregiver_parse_extra": {
        "target": 1500, "batch": 20,
        "system": (
            "You generate caregiver-note parsing training examples. Each: a natural caregiver/BCBA "
            "instruction + the structured JSON action array it parses to. "
            "Action types: add_phrase {categoryId, text}, remove_phrase {phraseText, categoryId}, "
            "reorder_phrase {phraseId, newSortOrder, categoryId}, add_category {name, icon}, "
            "add_sequence {name, categoryId, steps:[{label, options:[string]}]}, remove_sequence "
            "{sequenceName}, boost_word {word, boostCount}, note_only {} (clinical observation). "
            "categoryIds: help, food, feelings, school, quick, animals, colors, body, time. "
            "EMPHASIZE difficult cases: reorder_phrase (often missed), multi-action notes (chaining "
            "add+remove+boost), notes that LOOK actionable but are actually note_only ('Tom had a "
            "tough morning'). Each example must include a 'description' field per action. "
            "Return ONLY JSON: [{note, actions}]."
        ),
        "user": "Generate {n} diverse caregiver notes. About 30% must be reorder_phrase or multi-action; 20% should be note_only that LOOKS actionable but is just an observation.",
    },
    "emergency_qa_extra": {
        "target": 300, "batch": 15,
        "system": (
            "You generate emergency-phone-call Q&A training data. Each: a 911 operator's question + "
            "a CONTEXT BLOCK containing the emergency script (name, age, location, conditions, "
            "allergies, medications, callback). Answer extracts ONLY facts in the script. If asked "
            "for something NOT in the script, answer: \"I don't have that information.\" "
            "FOCUS on edge cases: questions with negation ('Is the patient NOT alone?'), questions "
            "asking about absent fields ('What's the patient's blood type?' when not in script), "
            "questions asking for inferences ('Is the patient verbal?' when script says 'nonverbal'), "
            "and questions requiring number extraction ('How many medications?'). "
            "Vary scripts: child/teen/adult, multiple conditions, multiple medications, different "
            "locations including in-vehicle, public transit, school. "
            "Return ONLY JSON: [{script, question, answer}]."
        ),
        "user": "Generate {n} edge-case emergency Q&A examples. Stress-test answer extraction, especially when info is absent (must say 'I don't have that information').",
    },
    "checkout_vs_loadcontext": {
        "target": 200, "batch": 15,
        "system": (
            "You generate CONTRASTIVE training examples for the v16 regression where the model "
            "incorrectly routes 'load context for X project' to memory_checkout instead of "
            "session_load_context.\n\n"
            "RIGHT tool: session_load_context — loads the CURRENT/LATEST state of a project at "
            "session start. Args: {project, level?}.\n"
            "TEMPTING WRONG tool: memory_checkout — restores memory to a SPECIFIC numeric version "
            "(rollback). Args: {project, version}.\n\n"
            "Discriminating signal: did the user mention a SPECIFIC VERSION NUMBER? If yes → "
            "memory_checkout. If no → session_load_context.\n\n"
            "Generate a 50/50 mix:\n"
            "- 50% prompts where the right tool is session_load_context. Phrasings include 'load', "
            "'show', 'pull up', 'resume', 'context', 'where we left off' — but NEVER mention a "
            "version number.\n"
            "- 50% prompts where the right tool is memory_checkout. Phrasings include 'restore to "
            "version 3', 'roll back memory to v5', 'checkout version 2', 'go back to v7'.\n\n"
            "<think> trace MUST name BOTH tools by name and explain the discriminating signal "
            "(presence/absence of a version number) before committing to the right one.\n"
            "Return ONLY JSON: [{prompt, think, right_tool, args}]."
        ),
        "user": "Generate {n} contrastive examples. EXACTLY 50/50 split: half session_load_context (no version number), half memory_checkout (with version number). The discriminator is ALWAYS 'did the user name a version'.",
    },
    # ── v16.2 BFCL-format categories (target the official Berkeley harness) ──
    # These produce examples in STANDARD Qwen FC format (<tool_call>...</tool_call>)
    # — separate from the synalux format used for prism-mcp routing — so the
    # model learns to switch formats based on the system prompt's tool block.
    "bfcl_simple_python": {
        "target": 700, "batch": 14,
        "system": (
            "You generate BFCL `simple_python` training rows. EXACT SHAPE — no deviations:\n"
            "[{\n"
            "  \"tools\": [ { \"type\": \"function\", \"function\": { \"name\": \"<snake_case>\", \"description\": \"...\", \"parameters\": { \"type\": \"object\", \"properties\": {...}, \"required\": [...] } } } ],\n"
            "  \"user\": \"<natural-language request>\",\n"
            "  \"call\": { \"name\": \"<must match tools[0].function.name>\", \"arguments\": { ... typed values ... } }\n"
            "}]\n\n"
            "RULES (violations make the row useless):\n"
            "  1. `tools` MUST be an ARRAY of ONE object, never a string.\n"
            "  2. `call` MUST be an OBJECT with `name` and `arguments` — never a Python-call string like \"add(a=1, b=2)\".\n"
            "  3. `arguments` values must match the declared parameter types (int → integer, str → string, etc).\n"
            "  4. snake_case for function and parameter names.\n\n"
            "EXAMPLE (the only acceptable shape):\n"
            "[{\n"
            "  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"calculate_area\",\"description\":\"Compute rectangle area.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"length\":{\"type\":\"number\"},\"width\":{\"type\":\"number\"}},\"required\":[\"length\",\"width\"]}}}],\n"
            "  \"user\": \"What is the area of a 10 by 5 rectangle?\",\n"
            "  \"call\": {\"name\":\"calculate_area\",\"arguments\":{\"length\":10,\"width\":5}}\n"
            "}]\n\n"
            "Vary domains across rows: math, geo, finance, weather, calendar, file ops, network, "
            "units, image, search, recipe, fitness, scheduling, vehicles, hardware, gaming, biology, "
            "chemistry. Return ONLY the JSON array. NO markdown fences. NO preamble."
        ),
        "user": "Generate {n} diverse simple_python rows STRICTLY in the shape above. Each row independent.",
    },
    "bfcl_multiple": {
        "target": 700, "batch": 14,
        "system": (
            "You generate BFCL `multiple` training rows. EXACT SHAPE:\n"
            "[{\n"
            "  \"tools\": [ {\"type\":\"function\",\"function\":{...}}, ... 2-4 of them ... ],\n"
            "  \"user\": \"<request that ONLY one of the tools can answer>\",\n"
            "  \"call\": {\"name\":\"<must match exactly one tools[i].function.name>\",\"arguments\":{...}}\n"
            "}]\n\n"
            "RULES:\n"
            "  1. `tools` MUST be an ARRAY of 2-4 OBJECTS — never a string.\n"
            "  2. `call` MUST be an OBJECT with name+arguments — never a Python-call string.\n"
            "  3. Wrong tools should be plausibly tempting (same domain, similar names) but only ONE "
            "matches the user's actual request.\n\n"
            "EXAMPLE:\n"
            "[{\n"
            "  \"tools\": [\n"
            "    {\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},\n"
            "    {\"type\":\"function\",\"function\":{\"name\":\"get_forecast\",\"description\":\"5-day forecast\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\",\"days\"]}}}\n"
            "  ],\n"
            "  \"user\": \"What is the weather in Tokyo right now?\",\n"
            "  \"call\": {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}\n"
            "}]\n\n"
            "Return ONLY the JSON array. NO markdown fences. NO preamble."
        ),
        "user": "Generate {n} diverse multiple rows STRICTLY in the shape above.",
    },
    "bfcl_parallel": {
        "target": 700, "batch": 14,
        "system": (
            "You generate BFCL `parallel` style training examples. Each example: ONE user request "
            "that requires 2-4 PARALLEL invocations of the SAME tool with different arguments. "
            "Output JSON: [{tools, user, calls}] where tools has ONE function and calls is an "
            "array of 2-4 {name, arguments} objects (all with the same name).\n"
            "Examples: 'Get weather for NYC, LA, and Tokyo' → 3 calls to get_weather. "
            "'Convert 100 USD to EUR, GBP, JPY' → 3 calls to convert_currency.\n"
            "Return ONLY JSON array."
        ),
        "user": "Generate {n} parallel BFCL examples. ONE tool, 2-4 parallel invocations with different arguments per example.",
    },
    "bfcl_parallel_multiple": {
        "target": 700, "batch": 14,
        "system": (
            "You generate BFCL `parallel_multiple` style training examples. Each example: ONE user "
            "request that requires 2-4 invocations of DIFFERENT tools (chosen from 3-6 available "
            "tools). Output JSON: [{tools, user, calls}] where tools has 3-6 functions and calls "
            "is an array of 2-4 {name, arguments} objects with distinct names.\n"
            "Example: 'Book flight to Tokyo, reserve hotel for May 5-10, and rent a car' → "
            "book_flight + reserve_hotel + rent_car.\n"
            "Return ONLY JSON array."
        ),
        "user": "Generate {n} parallel_multiple BFCL examples. 3-6 tools per example, 2-4 invocations of DISTINCT tools.",
    },
    "bfcl_simple_java": {
        "target": 500, "batch": 14,
        "system": (
            "You generate BFCL `simple_java` rows. SAME SHAPE AS bfcl_simple_python. The shape rules "
            "are non-negotiable:\n"
            "  1. `tools` is an ARRAY OF ONE OBJECT (never a string like \"java.io\").\n"
            "  2. `call` is an OBJECT {name, arguments} (never a method-call string).\n\n"
            "EXAMPLE (the only acceptable shape):\n"
            "[{\n"
            "  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"writeToFile\",\"description\":\"Write content to a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}],\n"
            "  \"user\": \"Write 'Hello, world!' to output.txt\",\n"
            "  \"call\": {\"name\":\"writeToFile\",\"arguments\":{\"path\":\"output.txt\",\"content\":\"Hello, world!\"}}\n"
            "}]\n\n"
            "Java-flavored: camelCase method names, Java-typed params (String, int, long, double, "
            "float, boolean, ArrayList<T>, HashMap<K,V>, char, byte). Domains: JDBC, file IO, "
            "threading, collections, sockets, HTTP clients, JVM, Spring, Maven. Return ONLY the "
            "JSON array. NO markdown fences."
        ),
        "user": "Generate {n} simple_java rows STRICTLY in the shape above.",
    },
    "bfcl_simple_javascript": {
        "target": 500, "batch": 14,
        "system": (
            "You generate BFCL `simple_javascript` rows. SAME SHAPE AS bfcl_simple_python.\n"
            "  1. `tools` is an ARRAY of one object (never a string).\n"
            "  2. `call` is an OBJECT {name, arguments} (never a function-call string).\n\n"
            "EXAMPLE:\n"
            "[{\n"
            "  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"fetchJson\",\"description\":\"Fetch and parse JSON from a URL\",\"parameters\":{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"},\"timeoutMs\":{\"type\":\"number\"}},\"required\":[\"url\"]}}}],\n"
            "  \"user\": \"Get the JSON from https://api.example.com/users\",\n"
            "  \"call\": {\"name\":\"fetchJson\",\"arguments\":{\"url\":\"https://api.example.com/users\"}}\n"
            "}]\n\n"
            "JS-flavored: camelCase, JS-typed params (string, number, boolean, array, object, any, "
            "null). Domains: DOM, fetch/axios, React state, Node fs/path, Express middleware, "
            "MongoDB queries, npm scripts, RegExp, Date, JSON. Return ONLY the JSON array."
        ),
        "user": "Generate {n} simple_javascript rows STRICTLY in the shape above.",
    },
    # ── v17 categories ───────────────────────────────────────────────────────
    # Boosted bfcl_irrelevance (700 → 2000) targets v16.3's 17.5% irrelevance
    # and 6.8% live_irrelevance failures. New tone_switch teaches the model to
    # produce a <tone:*> tag based on text classification (synalux strips it
    # server-side before delivering to the user).
    "tone_switch": {
        "target": 2000, "batch": 25,
        "system": (
            "You generate AAC tone-classification training rows. Each row: an AAC user input plus the "
            "appropriate response with a TONE TAG prefix indicating the emotional/situational register the "
            "response should be delivered in. Tag format: <tone:gentle> | <tone:urgent> | <tone:clinical> | "
            "<tone:playful> | <tone:neutral>.\n\n"
            "Rules for tag selection:\n"
            "  - urgent: emergency phrases (help, choking, hurt, can't breathe, seizure, fire)\n"
            "  - gentle: distress without emergency (sad, scared, tired, hurting, lonely, missing someone)\n"
            "  - clinical: factual medical/caregiver questions (medications, schedule, allergies)\n"
            "  - playful: requests for play, jokes, fun, social interaction with peers\n"
            "  - neutral: requests, observations, generic Q&A without strong affect\n\n"
            "Vary inputs across all 5 tones (~20% each). Mix English with a sprinkle of Spanish/French.\n\n"
            "Return ONLY a JSON array of {input, lang, tone, response} objects. response must START with "
            "the tone tag like '<tone:gentle> Of course you can rest...'. No fences, no preamble."
        ),
        "user": "Generate {n} diverse AAC inputs with appropriate tone tags. Cover all 5 tones with rough balance. Mix simple requests, complex emotional moments, and edge cases (e.g. urgent-sounding but actually playful: 'help me build the lego!').",
    },
    "bfcl_irrelevance": {
        "target": 2000, "batch": 14,
        "system": (
            "You generate BFCL `irrelevance` style training examples. Each example presents 1-3 "
            "tools, but the user request CANNOT BE ANSWERED by any provided tool — the right "
            "behavior is to NOT call any tool and instead respond in natural language.\n"
            "Output JSON: [{tools, user, answer}] where tools is 1-3 functions, user is a request "
            "that does NOT match any provided tool, and answer is a brief 1-2 sentence natural "
            "language reply explaining the model can't help with that via tools (or stating a "
            "general-knowledge answer).\n"
            "Make the mismatch SUBTLE: tool keywords overlap with user wording but the user's "
            "actual intent is not what the tool does. E.g., tool=get_weather, user='What's the "
            "weather like in my dreams last night?' (poetic, not a real lookup).\n"
            "Return ONLY JSON array."
        ),
        "user": "Generate {n} irrelevance BFCL examples. The user request must NOT match any provided tool — model should respond naturally without calling a tool.",
    },
}

# Tokens used in the final training format:
# <|synalux_think|>...</|synalux_think|> + <|tool_call|>{...}</|tool_call|> + <|synalux_answer|>...</|synalux_answer|>


@app.function(
    image=image,
    # 72B BF16 = ~145GB weights. 2x H100 (160GB) leaves no room for KV cache.
    # 4x H100 (320GB) gives clean BF16 load + plenty of cache headroom.
    # Burns more compute per container but preserves bit-exact teacher quality
    # (no AWQ/GPTQ quantization). Per "zero error assumption".
    gpu="H100:4",
    timeout=14400,
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
)
def generate_category(category: str, n_total: int, batch_size: int):
    """Run one category through vLLM offline batch on a single H100."""
    import vllm
    from vllm import SamplingParams

    spec = GEN_SPECS[category]
    print(f"[{category}] loading {TEACHER_MODEL} on H100 (offline batch)...", flush=True)
    llm = vllm.LLM(
        model=TEACHER_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=4,       # 4x H100 = 320GB total
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
    )

    n_batches = (n_total + batch_size - 1) // batch_size
    print(f"[{category}] generating {n_total} examples in {n_batches} batches of {batch_size}")

    all_examples = []
    prompts = []
    for _ in range(n_batches):
        chat = [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["user"].format(n=batch_size)},
        ]
        # Use vLLM tokenizer's apply_chat_template
        prompt = llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[{category}] vLLM done in {dt:.1f}s; parsing...")

    # Parse each batch output as JSON array
    kept = 0
    for o in outs:
        text = o.outputs[0].text.strip()
        # tolerate code fences / preamble
        if "[" in text and "]" in text:
            text = text[text.index("["): text.rindex("]") + 1]
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                all_examples.extend(arr)
                kept += len(arr)
        except Exception:
            continue

    print(f"[{category}] kept {kept} examples")
    out_path = Path(f"/vol/{category}.jsonl")
    with out_path.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    vol.commit()
    return {"category": category, "kept": kept, "wall_s": round(dt + (time.time() - t0 - dt), 1)}


@app.local_entrypoint()
def main(categories: str = ""):
    """Run all (or selected) categories in parallel.

    Each category gets its own H100 container — they run concurrently.
    Total wall time = max(per-category time), not sum.
    """
    selected = categories.split(",") if categories else list(GEN_SPECS.keys())
    selected = [c.strip() for c in selected if c.strip() in GEN_SPECS]
    print(f"Generating {len(selected)} categories in parallel: {selected}")

    work = [(c, GEN_SPECS[c]["target"], GEN_SPECS[c]["batch"]) for c in selected]
    results = list(generate_category.starmap(work))

    print("\n=== Summary ===")
    total_kept = 0
    for r in results:
        print(f"  {r['category']:20s}  kept={r['kept']:5d}   {r['wall_s']:.0f}s")
        total_kept += r["kept"]
    print(f"  {'TOTAL':20s}  kept={total_kept:5d}")


@app.local_entrypoint()
def fetch(out_dir: str = "/Users/admin/prism/training/data/v16_gen_72b"):
    """Pull the .jsonl files back from the Modal Volume.

    Writes to data/v16_gen_72b/ so the existing 7B-teacher data remains as
    a comparison fallback.
    """
    import os
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    fetched = 0
    for cat in GEN_SPECS:
        local = f"{out_dir}/{cat}.jsonl"
        try:
            subprocess.run(
                ["modal", "volume", "get", "prism-sft-gen-72b", f"{cat}.jsonl", local, "--force"],
                check=True, capture_output=True
            )
            sz = os.path.getsize(local) if os.path.exists(local) else 0
            print(f"  fetched {cat}.jsonl -> {local} ({sz/1024:.1f} KB)")
            fetched += 1
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if e.stderr else "(no stderr)"
            print(f"  ! {cat}.jsonl not in volume: {err[:120]}")
    print(f"\n{fetched}/{len(GEN_SPECS)} categories fetched to {out_dir}")


if __name__ == "__main__":
    print("Run via: modal run --detach modal_sft_generator.py::main")
