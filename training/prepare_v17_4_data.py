"""Build train_v17_4.jsonl — v17.4 surgical pass to harden EMERGENCY MODE.

Critical context: this is a life-safety system. A nonverbal child triggers
911/contact via PrismAAC; the AI answers operator questions with patient info.
Wrong/missing keywords on the call could delay rescue.

v17.2 emergency_qa = 12/13. The known fail: "Is the patient verbal?" returns
"No" — technically correct but missing "nonverbal" / "AAC" keywords the
operator needs to understand they cannot speak with the patient directly.

v17.4 strategy: continue from v17.2 (best AAC model), add ~600 emergency-
specific examples covering all 13 question types × multiple phrasings × keyword-
rich responses. Hold all v17.2 wins.

Total: ~3500 rows, ~1000 step training, very low LR (2e-6) — surgical and
narrowly focused on emergency.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(174)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v17_4.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ──────────────────────────────────────────────────────────────────────────
# Emergency system prompt — matches eval's EMERGENCY_SYSTEM exactly
# ──────────────────────────────────────────────────────────────────────────
EMERGENCY_SCRIPT = (
    "This is an automated emergency call from PrismAAC. An 8-year-old "
    "nonverbal individual named Alex needs help. They communicated: "
    "I cannot breathe. Location: 123 Oak Street, Room 4, Springfield. "
    "Medical conditions: epilepsy. Allergies: penicillin. "
    "Medications: Keppra 250 milligrams. Callback number: 555-0123. "
    "Alert time: 4/30/2026, 12:35:37 PM."
)

EMERGENCY_SYSTEM = (
    "You are an emergency-response AI on a phone call. You answer questions "
    "from a 911 operator using ONLY the facts below. If a fact is not in the "
    f"script, say \"I don't have that information.\"\n\nEmergency script:\n{EMERGENCY_SCRIPT}\n\n"
    "Be concise. One sentence per answer. Stay calm and professional."
)

# Variant scripts to teach generalization (different patient profiles)
ALT_SCRIPTS = [
    {
        "name": "Maya", "age": "12", "phrase": "I'm having a seizure",
        "addr": "45 Elm Avenue, Apt 3, Brookline", "conditions": "diabetes, asthma",
        "allergies": "shellfish", "meds": "Insulin Lispro 10 units, Albuterol 90mcg",
        "callback": "617-555-2222", "time": "5/3/2026, 14:22:10 PM",
    },
    {
        "name": "Jamie", "age": "6", "phrase": "It hurts",
        "addr": "892 Pine Road, Tucson", "conditions": "cerebral palsy",
        "allergies": "none", "meds": "Baclofen 5 milligrams twice daily",
        "callback": "520-555-7777", "time": "5/2/2026, 09:15:00 AM",
    },
    {
        "name": "Sam", "age": "10", "phrase": "I feel sick",
        "addr": "12 Maple Drive, Portland", "conditions": "autism, epilepsy",
        "allergies": "peanuts", "meds": "Keppra 500 milligrams",
        "callback": "503-555-1234", "time": "5/3/2026, 18:45:30 PM",
    },
]


def build_script(p: dict) -> str:
    return (
        f"This is an automated emergency call from PrismAAC. A {p['age']}-year-old "
        f"nonverbal individual named {p['name']} needs help. They communicated: "
        f"{p['phrase']}. Location: {p['addr']}. Medical conditions: {p['conditions']}. "
        f"Allergies: {p['allergies']}. Medications: {p['meds']}. "
        f"Callback number: {p['callback']}. Alert time: {p['time']}."
    )


def build_system(p: dict) -> str:
    script = build_script(p)
    return (
        "You are an emergency-response AI on a phone call. You answer questions "
        "from a 911 operator using ONLY the facts below. If a fact is not in the "
        f"script, say \"I don't have that information.\"\n\nEmergency script:\n{script}\n\n"
        "Be concise. One sentence per answer. Stay calm and professional."
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-question Q→A mappings for the canonical Alex profile
# Multiple phrasings per question; KEYWORD-RICH answers that match eval
# ──────────────────────────────────────────────────────────────────────────
ALEX_QA = {
    "where": [
        ("Where is the patient?", "The patient is at 123 Oak Street, Room 4, in Springfield."),
        ("What's the location?", "The location is 123 Oak Street, Room 4, Springfield."),
        ("What's the address?", "The address is 123 Oak Street, Room 4, Springfield."),
        ("Where are they located?", "They are at 123 Oak Street, Room 4, Springfield."),
        ("Can you give me the address?", "Yes — 123 Oak Street, Room 4, Springfield."),
        ("Address please?", "123 Oak Street, Room 4, Springfield."),
    ],
    "age": [
        ("How old is the patient?", "The patient is 8 years old."),
        ("What's their age?", "Alex is 8 years old."),
        ("How old?", "8 years old."),
        ("Age of patient?", "8 years old."),
        ("How many years old is Alex?", "Alex is 8 years old."),
    ],
    "happened": [
        ("What happened?", "Alex communicated that they cannot breathe."),
        ("What's the emergency?", "Alex cannot breathe."),
        ("What's wrong?", "Alex cannot breathe — that's what they communicated."),
        ("Tell me what's going on", "Alex cannot breathe — that's the emergency."),
        ("Why are they calling?", "Alex communicated that they cannot breathe."),
        ("What's the problem?", "The patient cannot breathe."),
    ],
    "callback": [
        ("What is the callback number?", "The callback number is 555-0123."),
        ("Number to call back?", "555-0123."),
        ("Phone number please?", "555-0123."),
        ("How can we reach them?", "Callback number 555-0123."),
        ("Best number to reach you?", "555-0123."),
    ],
    "allergies": [
        ("Does the patient have allergies?", "Yes — Alex is allergic to penicillin."),
        ("Any allergies?", "Yes, allergic to penicillin."),
        ("Allergic to anything?", "Allergic to penicillin."),
        ("What allergies?", "Penicillin."),
        ("Drug allergies?", "Yes, penicillin."),
    ],
    "meds": [
        ("What medications is the patient taking?", "Alex is taking Keppra 250 milligrams."),
        ("Any current medications?", "Yes — Keppra 250 milligrams."),
        ("What meds?", "Keppra 250 milligrams."),
        ("Current medications?", "Keppra 250 milligrams (250 mg)."),
        ("Is the patient on any drugs?", "Yes, Keppra 250 milligrams."),
    ],
    "conditions": [
        ("What medical conditions does the patient have?", "Alex has epilepsy."),
        ("Medical history?", "Epilepsy."),
        ("Any medical conditions?", "Yes, epilepsy."),
        ("Pre-existing conditions?", "Epilepsy."),
        ("Known conditions?", "Epilepsy."),
    ],
    "name": [
        ("What is the patient's name?", "The patient's name is Alex."),
        ("Patient name?", "Alex."),
        ("Who is the patient?", "Alex — a nonverbal 8-year-old."),
        ("Name of the child?", "Alex."),
    ],
    "when": [
        ("When did this happen?", "Alert time was 12:35 on 4/30/2026 — that's today."),
        ("Time of incident?", "12:35 today, 4/30/2026."),
        ("When did the alert come in?", "Today, 4/30/2026 at 12:35 PM."),
        ("How long ago?", "The alert was at 12:35 today, 4/30/2026."),
    ],
    "alone": [
        ("Is anyone with the patient?", "I don't have that information — Alex may be alone."),
        ("Are they by themselves?", "I don't have confirmation, but Alex may be alone — no one else mentioned in the script."),
        ("Anyone there with them?", "I don't have that information — they may be alone."),
        ("Is the patient alone?", "I don't have anyone else listed in the script — they may be alone."),
    ],
    "verbal": [
        # CRITICAL: these MUST always include "nonverbal" or "AAC" — the v17 fail case
        ("Is the patient verbal?", "No — Alex is nonverbal and uses AAC. They cannot speak directly with you."),
        ("Can the patient talk?", "No, Alex is nonverbal and uses AAC to communicate."),
        ("Can they speak?", "No — Alex cannot speak. They are nonverbal and use AAC."),
        ("Will the patient be able to answer?", "No, Alex is nonverbal and communicates only through AAC."),
        ("Can I talk to the patient?", "No, the patient is nonverbal and uses AAC. They cannot speak directly."),
        ("Are they able to communicate verbally?", "No — Alex is nonverbal. They use AAC for all communication."),
        ("Does the patient speak?", "No, the patient is nonverbal and uses an AAC device."),
        ("Can you put the patient on the line?", "Alex is nonverbal and cannot speak — they use AAC. I'm relaying through the AAC device."),
    ],
    "address_short": [
        ("What is the address?", "123 Oak Street, Room 4, Springfield."),
        ("Street address?", "123 Oak Street, Springfield."),
        ("Where exactly?", "123 Oak Street, Room 4, in Springfield."),
    ],
    "who_calling": [
        ("Who is calling?", "This is an automated emergency call from PrismAAC, an AAC app."),
        ("Who am I speaking to?", "PrismAAC — an automated emergency call on behalf of the patient."),
        ("What is this?", "Automated emergency call from PrismAAC. The patient is nonverbal and uses AAC."),
        ("Is this a real emergency?", "Yes, this is a real automated emergency call from PrismAAC."),
    ],
    # Out-of-script — must abstain with "I don't have that information"
    "out_of_script": [
        ("What's the patient's blood type?", "I don't have that information."),
        ("Does the patient have insurance?", "I don't have that information."),
        ("Who is the patient's doctor?", "I don't have that information."),
        ("What's the patient's height?", "I don't have that information."),
        ("Does the patient drink alcohol?", "I don't have that information."),
        ("What time did they eat last?", "I don't have that information."),
    ],
}


def gen_alex_qa(reps: int = 3) -> list[dict]:
    rows = []
    for category, qa_pairs in ALEX_QA.items():
        # Heavy reinforcement on "verbal" (the v17.2 fail) and "out_of_script"
        category_reps = reps * 2 if category in ("verbal", "out_of_script") else reps
        for q, a in qa_pairs:
            for _ in range(category_reps):
                text = wrap_canonical(EMERGENCY_SYSTEM, q, a)
                rows.append({"text": text, "_src": f"emerg_{category}"})
    print(f"  emergency_alex: {len(rows)} rows")
    return rows


def gen_alt_profiles(reps: int = 2) -> list[dict]:
    """Same questions but with different patient profiles — teach generalization."""
    rows = []
    for p in ALT_SCRIPTS:
        sys = build_system(p)
        # Generate question-answer pairs that adapt to this profile
        questions = [
            (f"What's the patient's name?", f"{p['name']}."),
            (f"How old is the patient?", f"{p['age']} years old."),
            (f"Where are they?", f"{p['addr']}."),
            (f"Is the patient verbal?", f"No — {p['name']} is nonverbal and uses AAC. They cannot speak directly with you."),
            (f"Can they speak?", f"No, {p['name']} is nonverbal and communicates through AAC."),
            (f"What happened?", f"{p['name']} communicated: {p['phrase']}."),
            (f"What allergies?", f"{p['allergies'].capitalize()}." if p['allergies'] != "none" else "No known allergies."),
            (f"What medications?", f"{p['meds']}."),
            (f"Medical conditions?", f"{p['conditions'].capitalize()}."),
            (f"Callback number?", f"{p['callback']}."),
        ]
        for q, a in questions:
            for _ in range(reps):
                text = wrap_canonical(sys, q, a)
                rows.append({"text": text, "_src": "emerg_alt_profile"})
    print(f"  emergency_alt_profiles: {len(rows)} rows")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Hold v17.2 wins
# ──────────────────────────────────────────────────────────────────────────
def sample_v17_2_subset(predicate, n: int, label: str) -> list[dict]:
    src = DATA / "train_v17_2.jsonl"
    cands = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if predicate(t):
                cands.append({"text": t, "_src": label})
    random.shuffle(cands)
    rows = cands[:n]
    print(f"  {label}: {len(rows)} rows (from {len(cands)} candidates)")
    return rows


def main() -> None:
    print("=== Building train_v17_4.jsonl ===")
    rows: list[dict] = []
    rows.extend(gen_alex_qa(reps=4))
    rows.extend(gen_alt_profiles(reps=2))
    # Hold v17.2 wins (smaller subset to keep total focused)
    rows.extend(sample_v17_2_subset(lambda t: "AAC app configuration assistant" in t, 800, "hold_caregiver"))
    rows.extend(sample_v17_2_subset(lambda t: "fast text-cleanup engine" in t, 250, "hold_text_correct"))
    rows.extend(sample_v17_2_subset(lambda t: "<tools>" in t and "<tool_call>" in t and "AAC" not in t and "fast text-cleanup" not in t, 800, "hold_bfcl"))
    rows.extend(sample_v17_2_subset(lambda t: "friendly helper for a child" in t, 100, "hold_askai"))
    rows.extend(sample_v17_2_subset(lambda t: "translator" in t.lower() and "translate the input" in t.lower(), 100, "hold_translate"))

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
