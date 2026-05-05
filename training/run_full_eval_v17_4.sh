#!/usr/bin/env bash
# Rigorous post-training evaluation of v17.1 — matches v12's 3-run methodology.
#
# Eval suite:
#   1. BFCL official (3 runs with --shuffle for statistical confidence)
#   2. AAC realigned (caregiver, text_correct, emergency_qa, translate, ask_ai)
#   3. Caregiver targeted re-test (20+ boost_word + 10+ reorder_phrase variants)
#   4. Format consistency probe (does the model emit canonical <tool_call>?)
#
# Reliability gates (must ALL pass for promotion to prism-coder:7b):
#   - BFCL median ≥ 95.0%   (vs v12's 100%, v17's 71.2%)
#   - BFCL 3-run StdDev ≤ 2.0%  (consistency)
#   - caregiver ≥ 6/7  (vs v17's 4/7 = 57%)
#   - emergency_qa ≥ 12/13 (vs v17's 12/13 = 92% — HARD HOLD, no regression)
#   - text_correct ≥ 13/15 (vs v17's 13/15 — HOLD)
#   - translate ≥ 7/8 (vs v17's 7/8 — HOLD)
#   - ask_ai = 5/5 (vs v17's 5/5 — HARD HOLD, life-safety adjacent)
#
# Output: /tmp/v17_4_full_eval.md

set -uo pipefail

MODEL_TAG="${MODEL_TAG:-prism-coder:7b-v17-4}"
BASE_DIR=/Users/admin/prism/training
REPORT=/tmp/v17_4_full_eval.md
LOG=/tmp/v17_4_full_eval.log
START_TS=$(date +%s)

cd "$BASE_DIR"
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== v17.1 full evaluation starting (model=$MODEL_TAG) ==="
# NB: avoid `ollama list | grep -q` — early grep exit triggers SIGPIPE on ollama list, fails under pipefail
OLLAMA_TAGS=$(ollama list 2>&1)
echo "$OLLAMA_TAGS" | grep -F "$MODEL_TAG" > /dev/null || { log "FATAL: Ollama tag $MODEL_TAG not registered"; exit 1; }

# ─── 1. BFCL official — 3 runs with shuffle ─────────────────────────────────
log "[1/4] BFCL official — 3 shuffled runs..."
BFCL_RESULTS=()
for run in 1 2 3; do
  log "  run $run/3..."
  RUN_LOG="/tmp/bfcl_v17_4_run${run}.log"
  python3 bfcl_eval.py --model "$MODEL_TAG" --shuffle 2>&1 | tee "$RUN_LOG" | tail -10 | tee -a "$LOG"
  SCORE=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$RUN_LOG" | tail -1 | grep -oE "[0-9.]+" | head -1)
  BFCL_RESULTS+=("$SCORE")
  log "  run $run score: ${SCORE}%"
done

# Compute median + stddev
BFCL_STATS=$(python3 -c "
import statistics
scores = [${BFCL_RESULTS[0]:-0}, ${BFCL_RESULTS[1]:-0}, ${BFCL_RESULTS[2]:-0}]
median = statistics.median(scores)
stdev = statistics.stdev(scores) if len(scores) > 1 else 0
print(f'{median:.1f},{stdev:.2f}')
")
BFCL_MEDIAN=$(echo "$BFCL_STATS" | cut -d, -f1)
BFCL_STDDEV=$(echo "$BFCL_STATS" | cut -d, -f2)
log "  BFCL: median=${BFCL_MEDIAN}% stddev=${BFCL_STDDEV}%"

# ─── 2. AAC realigned ───────────────────────────────────────────────────────
log "[2/4] AAC realigned eval..."
MODEL="$MODEL_TAG" python3 run_aac_realigned.py 2>&1 | tail -30 | tee -a "$LOG"
AAC_RESULTS_FILE="results/aac_realigned_${MODEL_TAG//[:\/]/_}.json"
if [ -f "$AAC_RESULTS_FILE" ]; then
  log "  AAC results -> $AAC_RESULTS_FILE"
  AAC_SUMMARY=$(python3 -c "
import json
d = json.load(open('$AAC_RESULTS_FILE'))
for k, v in d['by_task'].items():
    print(f'  {k:20s} {v[\"passed\"]}/{v[\"total\"]} = {v[\"pct\"]:.1%}')
")
  echo "$AAC_SUMMARY" | tee -a "$LOG"
fi

# ─── 3. Caregiver targeted re-test (boost_word + reorder_phrase) ────────────
log "[3/4] Caregiver targeted re-test..."
CARE_LOG=/tmp/v17_4_caregiver_targeted.log
MODEL="$MODEL_TAG" python3 - <<'PY' 2>&1 | tee "$CARE_LOG" | tail -20 | tee -a "$LOG"
import os, json, urllib.request, time

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ["MODEL"]
SYS = (
    "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
    "Available categories: help, food, feelings, school, quick\n"
    "Parse caregiver instruction. Return ONLY a JSON array. No explanation.\n"
    "Action types: add_phrase, remove_phrase, add_category, add_sequence, "
    "reorder_phrase, boost_word, note_only"
)

# 20 boost_word phrasings (the v17 weak spot)
BOOST = [
    ("He's using 'because' a lot now", "boost_word"),
    ("Started saying 'please' frequently this week", "boost_word"),
    ("Really likes the word 'help' lately", "boost_word"),
    ("Uses 'more' often during sessions", "boost_word"),
    ("We're hearing 'thank you' a lot today", "boost_word"),
    ("Says 'no' all the time now", "boost_word"),
    ("Spontaneously using 'water' more", "boost_word"),
    ("Hearing more 'happy' from her", "boost_word"),
    ("Boost 'play' please", "boost_word"),
    ("Can you boost 'tired'", "boost_word"),
    ("Their favorite word right now is 'mom'", "boost_word"),
    ("Has been saying 'go' a lot", "boost_word"),
    ("Keeps saying 'open'", "boost_word"),
    ("Loves the word 'cookie'", "boost_word"),
    ("Repeats 'milk' often", "boost_word"),
    # 10 reorder_phrase phrasings
    ("Move 'bathroom' to top of Help category", "reorder_phrase"),
    ("Make 'water' the first phrase in Food", "reorder_phrase"),
    ("Reorder 'tired' to position 1 in Feelings", "reorder_phrase"),
    ("Put 'hello' at the top of Quick", "reorder_phrase"),
    ("Promote 'home' to first in School category", "reorder_phrase"),
]

correct = 0
fails = []
for note, expected in BOOST:
    body = json.dumps({
        "model": MODEL, "system": SYS,
        "prompt": f'Caregiver says: "{note}"',
        "stream": False, "options": {"temperature": 0.0, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode()).get("response", "")
    # Look for expected action type in response
    if expected in resp:
        correct += 1
    else:
        fails.append((note, expected, resp[:120]))

total = len(BOOST)
print(f"Caregiver targeted: {correct}/{total} = {correct/total:.1%}")
for note, exp, resp in fails[:5]:
    print(f"  FAIL note={note!r} expected={exp} resp={resp!r}")
PY

# ─── 4. Format consistency probe ────────────────────────────────────────────
log "[4/4] Format consistency probe..."
FORMAT_LOG=/tmp/v17_4_format_probe.log
MODEL="$MODEL_TAG" python3 - <<'PY' 2>&1 | tee "$FORMAT_LOG" | tail -20 | tee -a "$LOG"
import os, json, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ["MODEL"]
TOOLS_SYS = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions.\n\n"
    "<tools>\n"
    '[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
    "\n</tools>\n\n"
    "Return tool calls within <tool_call></tool_call> XML tags."
)
canonical = mixed = bad = 0
for q in ("What's the weather in Paris?", "Weather in Tokyo?", "Tell me Madrid weather",
         "How's the weather in Berlin?", "Get weather for London"):
    body = json.dumps({
        "model": MODEL, "system": TOOLS_SYS, "prompt": q, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode()).get("response", "")
    has_canonical = "<tool_call>" in resp and "</tool_call>" in resp
    has_legacy = "<|tool_call|>" in resp
    if has_canonical and not has_legacy:
        canonical += 1
    elif has_legacy and not has_canonical:
        bad += 1
    else:
        mixed += 1

print(f"Format consistency: canonical={canonical}/5  legacy={bad}/5  mixed={mixed}/5")
PY

# ─── Build report ───────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START_TS ))
{
  echo "# v17.1 Full Evaluation Report"
  echo
  echo "**Model**: \`$MODEL_TAG\`"
  echo "**Eval time**: $(date)  (${ELAPSED}s wall)"
  echo
  echo "## BFCL — 3-run statistical eval"
  echo
  echo "| Run | Score |"
  echo "|---|---|"
  echo "| 1 | ${BFCL_RESULTS[0]:-?}% |"
  echo "| 2 | ${BFCL_RESULTS[1]:-?}% |"
  echo "| 3 | ${BFCL_RESULTS[2]:-?}% |"
  echo
  echo "**Median**: ${BFCL_MEDIAN}%   **StdDev**: ${BFCL_STDDEV}%"
  echo
  echo "**Reference**: v12 = 100.0%, v17 = 71.2%, gate = ≥95.0%"
  echo
  echo "## AAC realigned"
  echo '```'
  echo "$AAC_SUMMARY"
  echo '```'
  echo
  echo "## Caregiver targeted (boost_word + reorder_phrase)"
  echo '```'
  tail -25 "$CARE_LOG"
  echo '```'
  echo
  echo "## Format consistency"
  echo '```'
  tail -3 "$FORMAT_LOG"
  echo '```'
  echo
  echo "## Reliability gates"
  echo
  echo "| Gate | Threshold | Result |"
  echo "|---|---|---|"
  echo "| BFCL median | ≥ 95.0% | ${BFCL_MEDIAN}% |"
  echo "| BFCL stddev | ≤ 2.0% | ${BFCL_STDDEV}% |"
  echo "| caregiver | ≥ 6/7 | (see AAC table) |"
  echo "| emergency_qa | ≥ 12/13 (HARD) | (see AAC table) |"
  echo "| text_correct | ≥ 13/15 | (see AAC table) |"
  echo "| translate | ≥ 7/8 | (see AAC table) |"
  echo "| ask_ai | = 5/5 (HARD) | (see AAC table) |"
  echo
  echo "## Promotion decision"
  echo "If ALL gates pass: \`ollama cp $MODEL_TAG prism-coder:7b\`"
  echo "If gates fail: do NOT promote — review individual failures."
} > "$REPORT"

log "=== eval complete in ${ELAPSED}s — see $REPORT ==="
echo
cat "$REPORT"
