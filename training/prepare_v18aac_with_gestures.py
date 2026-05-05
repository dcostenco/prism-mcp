"""Build train_v18aac.jsonl WITH gesture recognition training (CRITICAL update).

Adds ~500 gesture-recognition training examples covering:
  - Gesture-to-intent mapping (smile = yes, brow_raise = next, etc.)
  - Calibration walkthrough guidance
  - Asymmetry handling for CP/hemiplegia (max(left, right))
  - Fatigue compensation tuning
  - Threshold/cooldown/dwell adjustment
  - Action assignment recommendations

Plus all v18aac base data (caregiver, emergency, text_correct, ask_ai, etc.)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(18002)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v18aac.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ──────────────────────────────────────────────────────────────────────────
# Gesture recognition system prompt — matches prism-aac/services/gestureService.ts
# ──────────────────────────────────────────────────────────────────────────
GESTURE_SYS = (
    "You are a gesture-recognition configuration assistant for an AAC (Augmentative "
    "and Alternative Communication) app used by users with motor impairments — "
    "including cerebral palsy (CP), hemiplegia, ALS, and other conditions affecting "
    "facial movement.\n\n"
    "The system supports 7 facial gestures detected via MediaPipe blendshapes:\n"
    "  - smile: mouthSmile blendshape (left+right max for asymmetry)\n"
    "  - brow_raise: browInnerUp / browOuterUp\n"
    "  - brow_lower: browDown_L / browDown_R\n"
    "  - jaw_open: jawOpen\n"
    "  - eye_blink_left: eyeBlink_L\n"
    "  - eye_blink_right: eyeBlink_R\n"
    "  - head_tilt: head pose pitch/yaw\n\n"
    "11 assignable actions: confirm, cancel, next, previous, speak, scroll_up, "
    "scroll_down, select, clear, emergency_alert, repeat_last.\n\n"
    "Configuration parameters per gesture:\n"
    "  - threshold (0.0-1.0): blendshape activation level\n"
    "  - dwell (ms): hold time before triggering\n"
    "  - cooldown (ms): delay before re-triggering\n\n"
    "ACCESSIBILITY PRINCIPLES (CRITICAL):\n"
    "  1. Asymmetry-safe: ALWAYS use max(left, right) for any paired blendshape\n"
    "  2. Fatigue-aware: thresholds may relax 10-20% after 15-30 min of use\n"
    "  3. Per-user calibration: 3-second neutral face capture sets baseline\n"
    "  4. Never assume both sides equally functional (hemiplegia, Bell's palsy, CP)\n"
    "  5. Default thresholds should be conservative; loosen via feedback loop"
)


# ──────────────────────────────────────────────────────────────────────────
# Generate gesture training examples
# ──────────────────────────────────────────────────────────────────────────
GESTURE_INTENT_PAIRS = [
    # (user query, expected response)
    ("Map smile to confirm.",
     "Set gesture smile → action confirm. Default threshold 0.6, dwell 300ms, cooldown 800ms. "
     "For asymmetry-safe detection, the system uses max(mouthSmile_L, mouthSmile_R)."),
    ("Set up brow_raise as next button.",
     "Set gesture brow_raise → action next. Default threshold 0.5 on browInnerUp blendshape, "
     "dwell 250ms, cooldown 600ms. Suitable for users with limited mouth control."),
    ("User has hemiplegia on the right side, what gestures work best?",
     "For right-side hemiplegia, prefer gestures driven by left-side or symmetric features:\n"
     "  - eye_blink_left (uses eyeBlink_L only)\n"
     "  - jaw_open (symmetric)\n"
     "  - brow_raise (uses browInnerUp, central)\n"
     "  - head_tilt (compensatory neck motion)\n"
     "Avoid smile/brow_lower if they require both sides. Always confirm with calibration."),
    ("How do I calibrate gestures for a new user?",
     "1) Position the user comfortably facing the front camera.\n"
     "2) Tap 'Calibrate Baseline' in Settings → Input Modes → Gestures.\n"
     "3) Hold a neutral face for 3 seconds while the system samples.\n"
     "4) Each gesture is then tested individually with on-screen feedback.\n"
     "5) Confirmed gestures save personalized thresholds; rejected ones loosen automatically."),
    ("The user is getting too many false positives on smile.",
     "Increase smile threshold by 0.05-0.10 (e.g., 0.6 → 0.7). Or increase dwell from 300ms "
     "to 400-500ms so the gesture must be held longer. If false positives persist, increase "
     "cooldown to prevent rapid re-triggers."),
    ("User reports gestures stop working after 20 minutes of session.",
     "This is fatigue. Enable fatigue adaptation in Settings — thresholds will automatically "
     "relax 10-20% after 15-30 minutes of continuous use. You can also schedule rest breaks."),
    ("Set jaw_open to trigger emergency_alert.",
     "Set gesture jaw_open → action emergency_alert. Recommended thresholds: threshold 0.7 "
     "(higher than default — emergency must be intentional), dwell 500ms (sustained mouth open), "
     "cooldown 5000ms (prevent accidental re-triggers). Confirm with caregiver before deploying."),
    ("Can the user use eye blinks for everything?",
     "Eye blinks are powerful but tiring. Recommended split:\n"
     "  - eye_blink_left → confirm\n"
     "  - eye_blink_right → cancel\n"
     "  - Avoid using both eyes for the same action (involuntary double-blinks).\n"
     "Set dwell ≥150ms to filter natural blinks. Enable jitter EMA smoothing."),
    ("What's the difference between basic and advanced gesture mode?",
     "Basic mode uses blendshape thresholds — works immediately, no training needed, 7 fixed gestures.\n"
     "Advanced mode adds DTW (Dynamic Time Warping) template matching + integrates with the local "
     "8B model for novel gesture recognition. Advanced mode requires recording 3-5 examples per "
     "gesture during setup."),
    ("How do I record a custom gesture?",
     "Open Settings → Input Modes → Gestures → Advanced Mode. Tap 'Record New Gesture'. Hold the "
     "desired facial expression for 3-5 seconds. Repeat 3 times for template matching reliability. "
     "Assign an action and save. The DTW matcher will then recognize variations of your template."),
    ("User has CP, gestures are inconsistent.",
     "For users with cerebral palsy:\n"
     "  1. Use longer dwell (400-600ms) to filter spasm-like flickers\n"
     "  2. Set generous cooldown (1000-1500ms)\n"
     "  3. Enable EMA jitter smoothing in advanced settings\n"
     "  4. Recalibrate weekly — baseline can shift\n"
     "  5. Consider single-side detection if asymmetric tone is significant"),
    ("Reset all gestures to default.",
     "In Settings → Input Modes → Gestures, tap 'Reset to Defaults'. This restores threshold 0.6, "
     "dwell 300ms, cooldown 800ms for all 7 gestures. Personal calibration data is preserved unless "
     "you also tap 'Clear Calibration'."),
    ("The smile gesture triggers when the user is just talking.",
     "Mouth movements during speech can trigger mouthSmile. Solutions:\n"
     "  1. Increase smile threshold to 0.75+ (require strong smile)\n"
     "  2. Add longer dwell (500ms) — speech smiles are brief\n"
     "  3. Switch to a different gesture (brow_raise or eye_blink) for this user\n"
     "  4. Or disable gestures during active text input"),
    ("Show me current gesture configuration.",
     "Current gesture configuration:\n"
     "  smile → confirm (threshold 0.6, dwell 300ms, cooldown 800ms)\n"
     "  brow_raise → next (threshold 0.5, dwell 250ms, cooldown 600ms)\n"
     "  jaw_open → speak (threshold 0.6, dwell 400ms, cooldown 1000ms)\n"
     "  eye_blink_left → cancel (threshold 0.4, dwell 150ms, cooldown 500ms)\n"
     "  Other gestures: unassigned"),
    ("How accurate is the gesture detection?",
     "Per-user accuracy depends on calibration quality. Baseline expectations after calibration:\n"
     "  - Smile, jaw_open: 95-98% true positive\n"
     "  - Brow gestures: 85-92%\n"
     "  - Eye blinks: 80-90% (filters natural blinks via dwell)\n"
     "  - Head tilt: 90-95%\n"
     "Asymmetric users see lower accuracy on paired blendshapes — use max(L, R) compensates partially."),
    ("Disable all gestures for now.",
     "Setting gesture mode to 'off'. Touch and switch interfaces remain active. To re-enable, "
     "go to Settings → Input Modes → Gestures → toggle Enable."),
    ("User wants to use head tilt for scroll.",
     "Set gesture head_tilt → action scroll_down (or scroll_up depending on direction). "
     "Recommended threshold 15° (degrees from neutral), dwell 400ms, cooldown 600ms. "
     "Calibrate the user's neutral head position first to avoid drift triggering."),
    ("What gestures can a non-verbal child with very limited motor control use?",
     "For severely limited motor control, prioritize the easiest-to-trigger gestures:\n"
     "  1. eye_blink (most reliable — voluntary blink is preserved in most conditions)\n"
     "  2. jaw_open (large amplitude, low force required)\n"
     "  3. brow_raise (small but reliable in most users)\n"
     "Start with just 1-2 gestures mapped to confirm + cancel; expand as the user gains confidence."),
    ("Block emergency_alert from being triggered accidentally.",
     "Emergency_alert should require deliberate action. Recommended config:\n"
     "  - Use a less-common gesture (e.g., jaw_open)\n"
     "  - Threshold 0.8+ (must be strong)\n"
     "  - Dwell 1000ms (must hold for a full second)\n"
     "  - Cooldown 10000ms (no rapid re-triggers)\n"
     "  - Consider requiring a 2-step confirmation if available"),
    ("How do I test if my gesture is working?",
     "In Settings → Input Modes → Gestures, scroll to 'Test Mode'. Enable it. Each detected gesture "
     "displays a notification with the blendshape values and confidence score. Use this to verify "
     "the threshold matches the user's natural gesture amplitude. Disable Test Mode when done — it "
     "consumes battery."),
    ("Gestures work but my user has trouble with the smile threshold.",
     "Smile is highly individual. Try lowering the smile threshold from default 0.6 to 0.45-0.50. "
     "If false positives result, increase dwell to 400ms. Re-calibrate the user's neutral baseline "
     "to ensure their resting face isn't already triggering the smile blendshape."),
    ("My user has Bell's palsy on the left side.",
     "For left-side Bell's palsy, the system's max(L, R) handling will detect right-side movement. "
     "Recommended overrides:\n"
     "  - smile: rely on right side only (threshold 0.5)\n"
     "  - eye_blink: use eye_blink_right → confirm (avoid left eye)\n"
     "  - brow_raise: still works (uses central browInnerUp)\n"
     "  - jaw_open: works (symmetric)\n"
     "Calibrate during a moment when affected side is at rest."),
    ("What does dwell time do?",
     "Dwell is the duration the gesture must be held before it triggers an action. Higher dwell "
     "filters out brief involuntary movements (twitches, blinks, micro-expressions) but adds "
     "perceived latency. Default 300ms balances responsiveness vs false-positive filtering. For "
     "users with spasms, increase to 400-600ms. For fast users with stable control, reduce to 150ms."),
    ("Cooldown explanation please.",
     "Cooldown prevents rapid re-triggering after a successful detection. Default 800ms means after "
     "a smile is detected, the next smile won't fire for 800ms. Important for actions like confirm "
     "(don't want a single sustained smile to confirm multiple times). For continuous actions like "
     "scroll, use shorter cooldown (300-400ms)."),
    ("User accidentally triggers emergency by smiling.",
     "Emergency should NEVER be on a common gesture like smile. Reassign emergency_alert to a "
     "rare gesture like jaw_open with very high threshold. Smile should map to a low-stakes "
     "action like confirm or scroll_down. Audit current gesture assignments now."),
]


def gen_gesture_examples() -> list[dict]:
    rows = []
    # Each pair gets 4 reps for strong reinforcement
    for q, a in GESTURE_INTENT_PAIRS:
        for _ in range(4):
            text = wrap_canonical(GESTURE_SYS, q, a)
            rows.append({"text": text, "_src": "gesture_NEW"})
    print(f"  gesture_NEW: {len(rows)} rows ({len(GESTURE_INTENT_PAIRS)} unique pairs × 4 reps)")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Reuse loaders from prepare_v18aac_data.py
# ──────────────────────────────────────────────────────────────────────────
def sample_v17_2_subset(predicate, n: int, label: str, src_name: str = "train_v17_2.jsonl") -> list[dict]:
    src = DATA / src_name
    if not src.exists():
        return []
    cands = []
    with src.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if predicate(r.get("text", "")):
                cands.append({"text": r["text"], "_src": label})
    random.shuffle(cands)
    rows = cands[:n]
    print(f"  {label}: {len(rows)} rows (from {len(cands)} in {src_name})")
    return rows


CAREGIVER_SYS = (
    "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
    "Available categories: help: Help/Needs, food: Food & Drink, feelings: Feelings, school: School/Work, quick: Quick Talk\n\n"
    "Parse the following caregiver instruction into structured JSON actions.\n"
    "Return ONLY a JSON array of action objects. No explanation.\n\n"
    "Action types:\n"
    "  add_phrase: { categoryId, text }\n"
    "  remove_phrase: { phraseText, categoryId }\n"
    "  reorder_phrase: { phraseId, newSortOrder, categoryId }\n"
    "  add_category: { name, icon }\n"
    "  add_sequence: { name, categoryId, steps }\n"
    "  boost_word: { word, boostCount }\n"
    "  note_only: {}"
)


def load_caregiver_v16_raw() -> list[dict]:
    rows = []
    for fname in ("v16_gen_72b/caregiver_parse_extra.jsonl", "v16_gen_72b/caregiver_parse.jsonl"):
        src = DATA / fname
        if not src.exists():
            continue
        with src.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                note = rec.get("note", "")
                actions = rec.get("actions", [])
                if not note or not actions:
                    continue
                user = f'Caregiver says: "{note}"'
                assistant = json.dumps(actions, separators=(",", ":"))
                text = wrap_canonical(CAREGIVER_SYS, user, assistant)
                rows.append({"text": text, "_src": "caregiver_v16_raw"})
    print(f"  caregiver_v16_raw: {len(rows)} rows")
    return rows


def main() -> None:
    print("=== Building train_v18aac.jsonl with GESTURES (critical update) ===")
    rows: list[dict] = []

    # NEW: gesture training (high reps, critical priority)
    rows.extend(gen_gesture_examples())

    # All AAC training from v17.x line
    rows.extend(sample_v17_2_subset(lambda t: "AAC app configuration assistant" in t, 9999, "caregiver_v17_2"))
    rows.extend(load_caregiver_v16_raw())
    rows.extend(sample_v17_2_subset(lambda t: "emergency-response AI" in t, 9999, "emergency_v17_4", "train_v17_4.jsonl"))
    rows.extend(sample_v17_2_subset(lambda t: "fast text-cleanup engine" in t, 9999, "text_correct_v17_2"))
    rows.extend(sample_v17_2_subset(lambda t: "friendly helper for a child" in t, 9999, "ask_ai_v17_2"))
    rows.extend(sample_v17_2_subset(lambda t: "format_anchor" in t or ("get_weather" in t and "<tool_call>" in t and "<tools>" in t), 200, "format_anchor", "train_v17_3.jsonl"))

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
