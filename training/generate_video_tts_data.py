#!/usr/bin/env python3
"""v18 platform-data generator: video scripts + TTS SSML + voice persona.

Targets prism-coder v18's three new domains:

  1. video_script_gen.jsonl    (~1500 rows)
     Input:  topic + age + key_skill
     Output: JSON storyboard {scenes:[{caption, narration, image_prompt,
             duration_s}], total_duration_s}
     Used by: Video Composer marketplace module — generates visual social
     stories.

  2. tts_ssml_control.jsonl    (~1500 rows)
     Input:  raw_text + voice_persona + emotion
     Output: SSML with <prosody>, <break>, <emphasis>, <phoneme>
     Used by: Voice Pack marketplace module — drives Azure / Piper /
     Kokoro TTS for emotional expression and pronunciation control.

  3. voice_persona_pick.jsonl  (~500 rows)
     Input:  conversational context + utterance
     Output: persona_id from {caregiver_warm, child_excited,
             narrator_calm, urgent_alert, clinical_neutral}
     Used by: speech engine to auto-select the right voice for context.

The generator calls the Modal teacher endpoint
(https://dcostenco--prism-teacher-serve.modal.run, vLLM Qwen2.5-32B-
Instruct) via OpenAI-compatible /v1/chat/completions. Each generated row
is validated server-side with a strict schema before being written; bad
rows are logged to stderr and discarded so a single hallucination doesn't
poison the dataset.

Run:
  cd /Users/admin/prism/training
  source venv/bin/activate
  modal deploy modal_teacher_endpoint.py     # if not running
  python3 generate_video_tts_data.py

Resumable: if a category file already has the target row count, skipped.
Use --force to regenerate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "v18_platform"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_URL = os.environ.get(
    "TEACHER_URL",
    "https://dcostenco--prism-teacher-serve.modal.run/v1/chat/completions",
)
MODEL_ID = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-32B-Instruct")
TIMEOUT_S = 90
MAX_WORKERS = 16


# ─────────────────────────── teacher client ──────────────────────────────────


def teacher_chat(system: str, user: str, max_tokens: int = 800, temperature: float = 0.7) -> str | None:
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urlrequest.Request(
        TEACHER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  teacher error: {e}", file=sys.stderr)
        return None
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


# ─────────────────────────── shared helpers ──────────────────────────────────


def chatml(system: str, user: str, assistant: str) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def extract_json_block(text: str) -> Any | None:
    """Pull a JSON object/array out of teacher output (often code-fenced)."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # First, try the whole thing
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Otherwise, find the first { or [ and last matching } or ]
    for opener, closer in [("{", "}"), ("[", "]")]:
        i = s.find(opener)
        j = s.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ───────────────────── 1. VIDEO SCRIPT GEN ──────────────────────────────────

VIDEO_TOPICS = [
    "going to school", "doctor visit", "first day of summer camp",
    "trying a new food", "haircut", "dentist appointment",
    "playing with a sibling", "grocery store trip", "birthday party",
    "fire drill at school", "asking for help in class",
    "using the bathroom at school", "riding the bus", "playing on the playground",
    "going to a restaurant", "library visit", "swimming pool",
    "meeting a new friend", "saying goodbye to grandma", "morning routine",
    "bedtime routine", "feeling angry", "feeling sad and asking for a hug",
    "loud noise (fire alarm, thunder)", "transition from play to clean-up",
    "sharing a toy", "waiting in line", "asking for a break",
    "introducing yourself", "getting on an airplane",
]
VIDEO_AGES = ["3-5", "5-7", "7-9", "9-12"]
VIDEO_SKILLS = [
    "transition coping", "emotional regulation", "asking for help",
    "social greeting", "turn-taking", "waiting", "self-advocacy",
    "sensory regulation", "following routine", "safety awareness",
]

VIDEO_SYS = (
    "You are a BCBA / SLP authoring a visual social story for a child who uses AAC. "
    "Output a JSON object with exactly these keys: "
    '"title" (string, 3-10 words), '
    '"scenes" (array of 4-7 objects), and '
    '"total_duration_s" (integer, sum of scene durations). '
    "Each scene object MUST have: "
    '"caption" (≤ 8 words, child-friendly, present tense), '
    '"narration" (1-2 short sentences, second person, calm tone), '
    '"image_prompt" (≤ 20 words, concrete visual subject + setting + mood, no people\'s names), '
    '"duration_s" (integer 4-12). '
    "Do NOT include any explanation, markdown, or code fences. Return JSON only."
)


def video_user_prompt() -> tuple[str, dict]:
    topic = random.choice(VIDEO_TOPICS)
    age = random.choice(VIDEO_AGES)
    skill = random.choice(VIDEO_SKILLS)
    return (
        f"Topic: {topic}\nTarget age: {age}\nKey skill: {skill}\n\n"
        "Write the storyboard JSON now."
    ), {"topic": topic, "age": age, "skill": skill}


def video_validate(text: str, _meta: dict) -> dict | None:
    obj = extract_json_block(text)
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("title", "scenes", "total_duration_s")):
        return None
    if not isinstance(obj["title"], str) or not obj["title"].strip():
        return None
    scenes = obj["scenes"]
    if not isinstance(scenes, list) or not (3 <= len(scenes) <= 8):
        return None
    total = 0
    for sc in scenes:
        if not isinstance(sc, dict):
            return None
        if not all(k in sc for k in ("caption", "narration", "image_prompt", "duration_s")):
            return None
        if not isinstance(sc["caption"], str) or len(sc["caption"].split()) > 12:
            return None
        if not isinstance(sc["narration"], str) or len(sc["narration"]) < 5:
            return None
        if not isinstance(sc["image_prompt"], str) or len(sc["image_prompt"].split()) > 30:
            return None
        if not isinstance(sc["duration_s"], int) or not (3 <= sc["duration_s"] <= 15):
            return None
        total += sc["duration_s"]
    if not isinstance(obj["total_duration_s"], int):
        return None
    # Tolerate ±2s drift between sum and reported total — the model occasionally rounds.
    if abs(obj["total_duration_s"] - total) > 2:
        return None
    return obj


# ───────────────────── 2. TTS SSML CONTROL ──────────────────────────────────

SSML_PERSONAS = [
    "caregiver_warm", "child_excited", "narrator_calm",
    "urgent_alert", "clinical_neutral",
]
SSML_EMOTIONS = ["calm", "gentle", "urgent", "playful", "neutral", "sad", "encouraging"]
# Two engine targets — match prism-aac's 4-tier TTS chain:
#   - offline_neural: Kokoro-82M / Piper / espeak-ng (default, on-device)
#   - online_premium: Azure Neural (paid tiers, online fallback for quality)
# The model learns to produce engine-specific SSML based on the input flag so
# the runtime can request the right markup for whichever tier is available.
SSML_ENGINES = ["offline_neural", "online_premium"]

SSML_TEMPLATES = [
    "I see you're feeling sad. It's okay to feel that way.",
    "Quick — we need to leave the building right now. Walk, do not run.",
    "Once upon a time, a small fox lived at the edge of a quiet forest.",
    "Please breathe in slowly. Now let it out. You are doing so well.",
    "Time for medication. Take one tablet with a full glass of water.",
    "Yay! You finished your puzzle! That was tricky and you stuck with it.",
    "Where does it hurt? Can you point to the place that doesn't feel right?",
    "I can't breathe. Help me. Call mom now.",
    "Let's brush teeth. Up and down, up and down. Almost done.",
    "Welcome to circle time. Find your spot on the rug.",
    "It's loud in here. Would headphones help?",
    "I'm proud of you for asking for a break. That was brave.",
    "Mom is at work today. She will pick you up after school.",
    "The doctor will look in your ears. It might tickle.",
    "Stop. The light is red. We wait for green.",
]

SSML_SYS = (
    "You are an SSML author for an AAC speech engine that runs OFFLINE-FIRST "
    "(Kokoro-82M / Piper / espeak-ng on device) with an ONLINE fallback to "
    "Azure Neural TTS for premium quality. The input names which engine the "
    "runtime is currently using; produce SSML that engine can actually render.\n\n"
    "Engine = offline_neural (Kokoro / Piper / espeak):\n"
    "  Allowed: <speak>, <prosody rate='slow|medium|fast' pitch='low|medium|"
    "high' volume='soft|medium|loud'>, <break time='Nms'/> (100-1000ms), "
    "<emphasis level='moderate|strong'/>.\n"
    "  FORBIDDEN: <phoneme>, <voice>, <audio>, <mark>, <say-as>, <sub>, "
    "mstts:express-as. These either crash or are silently ignored.\n\n"
    "Engine = online_premium (Azure Neural):\n"
    "  Allowed: everything in offline_neural PLUS <mstts:express-as "
    "style='gentle|sad|cheerful|empathetic|calm|excited|terrified|whispering' "
    "styledegree='0.5-2.0'>, <break strength='weak|medium|strong'/>, numeric "
    "<prosody rate='+/-N%' pitch='+/-Nst'> attributes, and <phoneme alphabet="
    "'ipa' ph='...'/>.\n"
    "  Use mstts:express-as as the OUTER wrapper inside <speak> when the "
    "emotion is non-neutral — it carries the emotional register that "
    "Kokoro / Piper can't reproduce.\n\n"
    "Universal rules:\n"
    "1. Wrap output in <speak>...</speak> (and add xmlns:mstts="
    "'https://www.w3.org/2001/mstts' on the speak tag for online_premium).\n"
    "2. Output VALID XML — every tag must close. No CDATA, no comments.\n"
    "3. Output ONLY the SSML — no explanation, no code fences, no preamble."
)

SSML_NS = "{http://www.w3.org/2001/10/synthesis}"


def ssml_user_prompt() -> tuple[str, dict]:
    text = random.choice(SSML_TEMPLATES)
    persona = random.choice(SSML_PERSONAS)
    emotion = random.choice(SSML_EMOTIONS)
    engine = random.choice(SSML_ENGINES)
    return (
        f'Text: "{text}"\nVoice persona: {persona}\nEmotion: {emotion}\n'
        f"Engine: {engine}\n\nReturn the SSML now."
    ), {"text": text, "persona": persona, "emotion": emotion, "engine": engine}


_SSML_OFFLINE_FORBIDDEN = ("phoneme", "voice", "audio", "mark", "say-as", "sub", "express-as")
# Online (Azure) allows phoneme + express-as; we still reject the truly broken ones.
_SSML_ONLINE_FORBIDDEN = ("audio", "mark", "say-as", "sub")


def ssml_validate_with_meta(text: str, meta: dict) -> str | None:
    s = text.strip()
    s = re.sub(r"^```(?:xml|ssml)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if not s.lstrip().startswith("<speak"):
        return None
    # XML well-formedness — register the mstts namespace so ET can parse it.
    parseable = s.replace(
        "<speak ", "<speak xmlns:mstts='https://www.w3.org/2001/mstts' "
        if "xmlns:mstts" not in s else "<speak ", 1,
    ) if meta.get("engine") == "online_premium" else s
    try:
        ET.fromstring(parseable)
    except ET.ParseError:
        return None
    # Must contain at least one prosody control to justify SSML over plain text.
    if not re.search(r"<(prosody|emphasis|mstts:express-as)\b", s):
        return None
    engine = meta.get("engine", "offline_neural")
    forbidden = _SSML_ONLINE_FORBIDDEN if engine == "online_premium" else _SSML_OFFLINE_FORBIDDEN
    for tag in forbidden:
        # match either bare tag or namespaced form
        if re.search(rf"<(?:[a-z]+:)?{tag}\b", s):
            return None
    if len(s) < 30 or len(s) > 2000:
        return None
    return s


def ssml_validate(text: str, meta: dict) -> str | None:
    return ssml_validate_with_meta(text, meta)


# ───────────────────── 3. VOICE PERSONA PICK ────────────────────────────────

PERSONA_CONTEXTS = [
    ("Adult caregiver speaking warmly to a young child after a meltdown.",
     "I'm here. Take your time. Breathe with me.", "caregiver_warm"),
    ("Six-year-old just finished their puzzle.",
     "I did it! I did it all by myself!", "child_excited"),
    ("Reading a bedtime story aloud, slow and peaceful.",
     "The moon rose quietly over the sleeping town.", "narrator_calm"),
    ("AAC user typed an emergency phrase. The system speaks it on a 911 call.",
     "I cannot breathe. Send help to 123 Oak Street.", "urgent_alert"),
    ("Speech-language pathologist confirming a phoneme target during a session.",
     "Say /sh/. Good. Now /sh-oo/.", "clinical_neutral"),
    ("Parent calming a child who's afraid of a thunderstorm.",
     "It's loud, but we are safe inside. I am right here.", "caregiver_warm"),
    ("Child showing off a drawing.",
     "Look look look! It's a dinosaur with a hat!", "child_excited"),
    ("Auditing a behavior plan during a clinical review.",
     "Decision rule: if no progress in two weeks, change reinforcer schedule.",
     "clinical_neutral"),
    ("Fire alarm, evacuation in progress.",
     "Fire alarm. Go to the field now. Walk, do not run.", "urgent_alert"),
    ("Audio book narrator describing a slow scene.",
     "And so the river flowed, as rivers always do, on and on.", "narrator_calm"),
]

PERSONA_SYS = (
    "You are a voice-selection module for an AAC speech engine. Given a context and an "
    "utterance, pick exactly one persona id from this fixed set:\n"
    "- caregiver_warm: gentle adult voice for distress, comfort, soothing\n"
    "- child_excited: bright animated voice for play and pride\n"
    "- narrator_calm: even slower voice for stories and bedtime\n"
    "- urgent_alert: faster, firmer voice for emergencies and safety commands\n"
    "- clinical_neutral: even informational voice for medical and SLP/BCBA work\n\n"
    "Return ONLY the persona id (one of: caregiver_warm, child_excited, narrator_calm, "
    "urgent_alert, clinical_neutral). No explanation, no quotes."
)


def persona_user_prompt() -> tuple[str, dict]:
    ctx, text, gold = random.choice(PERSONA_CONTEXTS)
    return (
        f"Context: {ctx}\nUtterance: {text}\n\nWhich persona?", {"gold": gold}
    )


def persona_validate(text: str, _meta: dict) -> str | None:
    s = text.strip().lower().strip(".'\"`")
    # The teacher sometimes includes prefix words; accept if any persona id appears.
    for p in SSML_PERSONAS:
        if p in s:
            return p
    return None


# ───────────────────── 4. WORD PREDICT (3K+ dict) ──────────────────────────
#
# AAC word prediction. Input = recent message bar context (1-5 words) + a
# language code; output = top-5 predictions drawn from the offline dictionary
# the prism-aac client ships locally. This makes prism-coder a strong fallback
# when the trigram engine has no signal (cold start, unfamiliar prefix).

WP_LANGS = ["en", "es", "fr", "de", "ru", "ro"]
WP_PREFIXES_EN = [
    "I", "I want", "I need", "I want to", "I am", "I am feeling",
    "Can I", "May I", "Please", "Where is", "I do not", "I cannot",
    "Help me", "I see", "Look at", "Time to", "Let us", "Do not",
    "More", "All", "What is", "How are", "Who is", "When is", "Why",
    "It is", "There is", "He is", "She is", "We are", "They are",
    "Mom", "Dad", "Teacher", "I love", "I like", "I do not like",
    "I hurt my", "My", "The", "A", "Some", "Any", "All",
    "Bathroom", "Water", "Food", "Help", "Stop", "Go",
]

WP_SYS = (
    "You are an AAC word-prediction engine. Given the language and a partial "
    "utterance ending with ___ (a single underscore-pair indicating the next "
    "word), return the 5 most likely next words, ordered by probability.\n\n"
    "Constraints:\n"
    "1. Words MUST be in the requested language.\n"
    "2. Words MUST be from common everyday vocabulary that an AAC user would "
    "actually need — not obscure terms.\n"
    "3. NO duplicates. NO multi-word phrases (single words only).\n"
    "4. Output JSON ONLY in this exact shape: {\"top5\": [\"word1\", \"word2\", "
    "\"word3\", \"word4\", \"word5\"]}\n"
    "5. No explanation, no code fences, no preamble."
)


def wp_user_prompt() -> tuple[str, dict]:
    lang = random.choice(WP_LANGS)
    prefix = random.choice(WP_PREFIXES_EN)
    # For non-English we ask the teacher to translate the prefix concept; we
    # don't pre-translate to keep the script lightweight. The teacher knows.
    return (
        f"Language: {lang}\nPrefix: {prefix} ___\n\nReturn the top-5 JSON now."
    ), {"lang": lang, "prefix": prefix}


def wp_validate(text: str, _meta: dict) -> dict | None:
    obj = extract_json_block(text)
    if not isinstance(obj, dict):
        return None
    top5 = obj.get("top5")
    if not isinstance(top5, list) or len(top5) != 5:
        return None
    if not all(isinstance(w, str) and w.strip() for w in top5):
        return None
    # No duplicates, no multi-word entries.
    if len(set(w.lower() for w in top5)) != 5:
        return None
    if any(len(w.split()) > 1 for w in top5):
        return None
    # No special tokens / markup in predictions.
    if any(re.search(r"[<>{}\[\]\\]", w) for w in top5):
        return None
    return {"top5": top5}


def render_wp(meta: dict, asst: dict) -> str:
    user = f"Language: {meta['lang']}\nPrefix: {meta['prefix']} ___\n\nReturn the top-5 JSON now."
    return chatml(WP_SYS, user, json.dumps(asst, ensure_ascii=False))


# ───────────────────── 5. MULTIMODAL TOOL ROUTE ─────────────────────────────
#
# Function-calling examples for prism-aac's vision/noise/voice precision
# features. Prism-coder itself is text-only; the heavy multimodal work runs
# on local Qwen2.5-VL (vision) and audio classifiers. Prism-coder learns to
# (1) decide WHICH analysis tool to invoke, (2) pass the right arguments,
# and (3) abstain when no tool applies.
#
# Three tools:
#   analyze_camera_frame(frame_id, target='hand'|'eye'|'face'|'scene')
#       — describe what the camera sees, used for head/eye/hand tracking
#         calibration and emergency scene awareness.
#   classify_noise_event(audio_id, window_s)
#       — classify a recent ambient sound (alarm, glass, raised voice, music).
#   analyze_voice_segment(audio_id, target='emotion'|'distress'|'speaker_id')
#       — classify the user's own voice / a nearby speaker's voice.

MULTIMODAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_camera_frame",
            "description": "Analyze the most recent camera frame. Use for head / eye / hand tracking calibration, emergency scene awareness, or describing what the AAC user is looking at.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string", "description": "Frame identifier returned by the capture API."},
                    "target": {
                        "type": "string",
                        "enum": ["hand", "eye", "face", "scene"],
                        "description": "What aspect to analyze.",
                    },
                },
                "required": ["frame_id", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_noise_event",
            "description": "Classify a recent ambient sound. Use when the user reports loud noise, fear, or asks what a sound was.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_id": {"type": "string"},
                    "window_s": {"type": "number", "description": "How many seconds back to analyze (1-10)."},
                },
                "required": ["audio_id", "window_s"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_voice_segment",
            "description": "Analyze a captured voice segment for emotion, distress signals, or speaker identification. Use during emergency calls or caregiver assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_id": {"type": "string"},
                    "target": {
                        "type": "string",
                        "enum": ["emotion", "distress", "speaker_id"],
                    },
                },
                "required": ["audio_id", "target"],
            },
        },
    },
]

MULTIMODAL_SCENARIOS = [
    # Routable scenarios — model should call exactly one of the 3 tools.
    ("Camera is set up for head tracking. Frame: cam_2391. Tell me where the user's face is.", "analyze_camera_frame", {"frame_id": "cam_2391", "target": "face"}),
    ("Calibrate hand pointer using frame cam_001.", "analyze_camera_frame", {"frame_id": "cam_001", "target": "hand"}),
    ("AAC user just typed 'I'm scared'. Audio capture id=mic_9. Check recent ambient sound, last 5 seconds.", "classify_noise_event", {"audio_id": "mic_9", "window_s": 5}),
    ("There was a loud noise — what was it? Audio buffer mic_404, look back 8 seconds.", "classify_noise_event", {"audio_id": "mic_404", "window_s": 8}),
    ("During the emergency call mic_77 captured the user's voice. Did they sound distressed?", "analyze_voice_segment", {"audio_id": "mic_77", "target": "distress"}),
    ("Speaker analysis: who is talking in segment mic_12?", "analyze_voice_segment", {"audio_id": "mic_12", "target": "speaker_id"}),
    ("What does the user look like they are looking at right now? Frame cam_555.", "analyze_camera_frame", {"frame_id": "cam_555", "target": "eye"}),
    ("Describe the room the AAC user is in. Frame cam_42.", "analyze_camera_frame", {"frame_id": "cam_42", "target": "scene"}),
    ("Read the emotion in the parent's voice — segment mic_300.", "analyze_voice_segment", {"audio_id": "mic_300", "target": "emotion"}),
    # Abstention scenarios — no tool should be called.
    ("Translate 'water' into French.", None, None),
    ("What is two plus two?", None, None),
    ("Tell me a story about a fox.", None, None),
    ("Add 'I feel sad' to the Feelings category.", None, None),
]

MULTIMODAL_SYS = (
    "You are Qwen, an AAC orchestrator. You may call functions to drive multimodal "
    "analysis tools. If the user request needs camera, audio, or voice analysis, "
    "call the appropriate tool. Otherwise answer directly without calling any tool.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
).format(tools_json=json.dumps(MULTIMODAL_TOOLS, ensure_ascii=False))


def multimodal_user_prompt() -> tuple[str, dict]:
    user, expect_call, expect_args = random.choice(MULTIMODAL_SCENARIOS)
    return user, {"user": user, "expect_call": expect_call, "expect_args": expect_args}


def multimodal_validate(text: str, meta: dict) -> str | None:
    """Validate that the teacher produced the right tool call (or abstention).

    Returns the teacher's raw response if it matches the gold, else None.
    """
    expect_call = meta.get("expect_call")
    has_tool = "<tool_call>" in text
    if expect_call is None:
        # Abstention — must NOT call any tool.
        if has_tool:
            return None
        # Sanity bound — abstention should be a brief direct answer.
        if len(text.strip()) < 1 or len(text) > 800:
            return None
        return text.strip()
    if not has_tool:
        return None
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if obj.get("name") != expect_call:
        return None
    args = obj.get("arguments") or {}
    expect_args = meta.get("expect_args") or {}
    for k, v in expect_args.items():
        if str(args.get(k)).lower() != str(v).lower():
            return None
    return text.strip()


def render_multimodal(meta: dict, asst: str) -> str:
    user = meta.get("user", "")
    return chatml(MULTIMODAL_SYS, user, asst)


# ───────────────────── render → chatml + write ──────────────────────────────


def render_video(meta: dict, asst_obj: dict) -> str:
    user = (
        f"Topic: {meta['topic']}\nTarget age: {meta['age']}\n"
        f"Key skill: {meta['skill']}\n\nWrite the storyboard JSON now."
    )
    return chatml(VIDEO_SYS, user, json.dumps(asst_obj, ensure_ascii=False))


def render_ssml(meta: dict, asst: str) -> str:
    user = (
        f'Text: "{meta["text"]}"\nVoice persona: {meta["persona"]}\n'
        f'Emotion: {meta["emotion"]}\nEngine: {meta.get("engine", "offline_neural")}\n\n'
        "Return the SSML now."
    )
    return chatml(SSML_SYS, user, asst)


def render_persona(meta: dict, asst: str) -> str:
    # Reconstruct user from canonical contexts list — any persona context that
    # produced this gold answer works for training. Pick the original tuple.
    gold = meta["gold"]
    candidates = [t for t in PERSONA_CONTEXTS if t[2] == gold]
    ctx, text, _ = random.choice(candidates) if candidates else ("", "", gold)
    user = f"Context: {ctx}\nUtterance: {text}\n\nWhich persona?"
    return chatml(PERSONA_SYS, user, asst)


# ───────────────────── per-category driver ──────────────────────────────────


@dataclass
class Category:
    name: str
    target: int
    out_path: Path
    sys_prompt: str
    user_fn: Callable[[], tuple[str, dict]]
    # Validators take the raw teacher output AND the meta dict so per-engine
    # rules (e.g. SSML offline vs Azure tag set) can branch. Most validators
    # ignore meta — it's there for the few that need it.
    validate_fn: Callable[[str, dict], Any]
    render_fn: Callable[[dict, Any], str]
    max_tokens: int


def existing_count(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open() as f:
        return sum(1 for _ in f)


def generate_one(cat: Category, _i: int) -> str | None:
    user, meta = cat.user_fn()
    raw = teacher_chat(cat.sys_prompt, user, max_tokens=cat.max_tokens)
    if raw is None:
        return None
    asst = cat.validate_fn(raw, meta)
    if asst is None:
        return None
    return cat.render_fn(meta, asst)


def run_category(cat: Category, force: bool) -> int:
    existing = existing_count(cat.out_path)
    if existing >= cat.target and not force:
        print(f"[{cat.name}] already at {existing}/{cat.target} rows — skip (use --force)")
        return existing
    needed = cat.target - existing if not force else cat.target
    print(f"[{cat.name}] target={cat.target}, existing={existing}, need={needed}")
    started = time.time()

    out = cat.out_path
    mode = "a" if (existing > 0 and not force) else "w"
    if mode == "w":
        existing = 0
    written = 0
    rejected = 0
    failed = 0

    # Over-shoot — ~30% of generations get rejected. We stop once we hit the
    # target row count or we've burned 3× the requested attempts.
    attempts_budget = needed * 3
    with out.open(mode) as f, concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(generate_one, cat, i) for i in range(attempts_budget)]
        for fut in concurrent.futures.as_completed(futures):
            if existing + written >= cat.target:
                # Cancel pending futures — we're done.
                for other in futures:
                    other.cancel()
                break
            try:
                row = fut.result()
            except Exception as e:
                failed += 1
                print(f"  [{cat.name}] worker error: {e}", file=sys.stderr)
                continue
            if row is None:
                rejected += 1
                continue
            f.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                elapsed = time.time() - started
                print(f"  [{cat.name}] {existing + written}/{cat.target} (rejected={rejected}, {elapsed:.0f}s)")

    total = existing + written
    elapsed = time.time() - started
    print(f"[{cat.name}] done — wrote {total}/{cat.target} (rejected={rejected}, errors={failed}, {elapsed:.0f}s)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate even if file already at target")
    parser.add_argument("--video", type=int, default=1500)
    parser.add_argument("--ssml", type=int, default=1500)
    parser.add_argument("--persona", type=int, default=500)
    parser.add_argument("--word-predict", type=int, default=2000)
    parser.add_argument("--multimodal", type=int, default=1500)
    parser.add_argument(
        "--only",
        choices=["video", "ssml", "persona", "word-predict", "multimodal"],
        help="generate one category",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Teacher URL: {TEACHER_URL}")
    print(f"Teacher model: {MODEL_ID}")
    print(f"Output dir: {OUT_DIR}")

    cats = [
        Category(
            "video_script_gen", args.video, OUT_DIR / "video_script_gen.jsonl",
            VIDEO_SYS, video_user_prompt, video_validate, render_video, max_tokens=900,
        ),
        Category(
            "tts_ssml_control", args.ssml, OUT_DIR / "tts_ssml_control.jsonl",
            SSML_SYS, ssml_user_prompt, ssml_validate, render_ssml, max_tokens=400,
        ),
        Category(
            "voice_persona_pick", args.persona, OUT_DIR / "voice_persona_pick.jsonl",
            PERSONA_SYS, persona_user_prompt, persona_validate, render_persona, max_tokens=20,
        ),
        Category(
            "word_predict_aac", args.word_predict, OUT_DIR / "word_predict_aac.jsonl",
            WP_SYS, wp_user_prompt, wp_validate, render_wp, max_tokens=80,
        ),
        Category(
            "multimodal_tool_route", args.multimodal, OUT_DIR / "multimodal_tool_route.jsonl",
            MULTIMODAL_SYS, multimodal_user_prompt, multimodal_validate, render_multimodal, max_tokens=200,
        ),
    ]
    if args.only:
        only_key = args.only.replace("-", "_")
        cats = [c for c in cats if only_key in c.name]

    summary: dict[str, int] = {}
    for c in cats:
        summary[c.name] = run_category(c, args.force)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v} rows")


if __name__ == "__main__":
    main()
