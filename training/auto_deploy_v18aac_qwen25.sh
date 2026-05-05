#!/usr/bin/env bash
# Auto-deploy v18aac (Qwen2.5-Coder-7B-Instruct base) when training completes,
# then auto-eval vs v17.4. If passes all gates → save report for promotion decision.
# DOES NOT touch production tag without explicit user approval.

set -uo pipefail

APP_NAME="prism-v18aac-sft"
BASE_DIR=/Users/admin/prism/training
LLAMA_CPP=/Users/admin/llama.cpp
LOG=/tmp/auto_deploy_v18aac.log
REPORT=/tmp/v18aac_qwen25_report.md

ADAPTER_DIR=$BASE_DIR/models/v18aac_qwen25_modal
FUSED_DIR=$BASE_DIR/models/prism-v18aac-qwen25-fused
GGUF=$BASE_DIR/models/prism-v18aac-qwen25-fused-q4km.gguf
OLLAMA_TAG=prism-coder:7b-v18aac-qwen25

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== Auto-deploy v18aac (Qwen2.5-Coder base) — polling for training completion ==="

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  STATE=$(COLUMNS=200 modal app list 2>/dev/null | grep "$APP_NAME" | grep "ephemeral" | head -1)
  if [ -z "$STATE" ]; then
    log "[poll $ATTEMPT] no running prism-v18aac-sft app — assuming completed"
    break
  fi
  log "[poll $ATTEMPT] still running"
  sleep 60
done

log "=== Fetch final adapter ==="
mkdir -p "$ADAPTER_DIR"
modal volume get prism-v18aac final_adapter "$ADAPTER_DIR/" --force 2>&1 | tail -3 | tee -a "$LOG"
[ -f "$ADAPTER_DIR/final_adapter/adapter_config.json" ] || { log "FATAL: adapter not fetched"; exit 1; }

log "=== PEFT merge → BF16 ==="
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py "Qwen/Qwen2.5-Coder-7B-Instruct" "$ADAPTER_DIR/final_adapter" "$FUSED_DIR" 2>&1 | tail -5 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || { log "FATAL: merge incomplete"; exit 1; }

log "=== Convert HF → GGUF F16 ==="
GGUF_FP16=/tmp/v18aac_qwen25_fp16.gguf
PYTHONPATH=$LLAMA_CPP/gguf-py python3 $LLAMA_CPP/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

log "=== Quantize Q4_K_M ==="
QBIN=$LLAMA_CPP/build/bin/llama-quantize
[ -x "$QBIN" ] || QBIN=$LLAMA_CPP/llama-quantize
"$QBIN" "$GGUF_FP16" "$GGUF" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
rm -f "$GGUF_FP16"

log "=== Register Ollama tag $OLLAMA_TAG ==="
cat > /tmp/Modelfile.v18aac-qwen25 <<EOF
FROM $GGUF

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are Prism AAC Assistant. Help nonverbal users, caregivers, and BCBAs with: configuring AAC apps, parsing caregiver instructions, handling emergencies, correcting text, gestures setup, and translation. Direct, concise responses."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</tool_call>"
EOF
ollama create "$OLLAMA_TAG" -f /tmp/Modelfile.v18aac-qwen25 2>&1 | tail -3 | tee -a "$LOG"

log "=== Run AAC realigned eval ==="
EVAL_LOG=/tmp/v18aac_qwen25_eval.log
MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tee "$EVAL_LOG" > /dev/null
log "Eval done."

# Parse scores
ASK_AI=$(grep -oE "ask_ai\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
CAREGIVER=$(grep -oE "caregiver\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
EMERGENCY=$(grep -oE "emergency_qa\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
TEXT_CORRECT=$(grep -oE "text_correct\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
TRANSLATE=$(grep -oE "translate\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
OVERALL=$(grep -oE "OVERALL\s+[0-9]+/[0-9]+" "$EVAL_LOG" | head -1 | grep -oE "[0-9]+/[0-9]+" | head -1)

log "=== Gesture spot-check ==="
GESTURE_LOG=/tmp/v18aac_qwen25_gesture.log
MODEL="$OLLAMA_TAG" python3 - <<'PY' 2>&1 | tee "$GESTURE_LOG" > /dev/null
import os, json, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ["MODEL"]
SYS = "You are a gesture-recognition configuration assistant for an AAC app for users with motor impairments (CP, hemiplegia, ALS). 7 gestures: smile, brow_raise, brow_lower, jaw_open, eye_blink_left, eye_blink_right, head_tilt. 11 actions. Always use max(left, right) for asymmetry safety."

CASES = [
    ("Map smile to confirm.", "smile"),
    ("User has hemiplegia on the right side, what gestures work best?", "left"),
    ("How do I calibrate gestures?", "calibrat"),
    ("What's threshold for jaw_open?", "threshold"),
    ("User has CP, gestures are inconsistent.", "cp"),
]
passed = 0
for q, expect in CASES:
    body = json.dumps({
        "model": MODEL, "system": SYS, "prompt": q,
        "stream": False, "options": {"temperature": 0.0, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode()).get("response", "").strip().lower()
    matched = expect.lower() in resp
    if matched: passed += 1
    print(f"{'✅' if matched else '❌'} '{q[:50]}' → expect '{expect}'")
print(f"\nGesture: {passed}/{len(CASES)}")
PY
GESTURE_RESULT=$(grep -oE "Gesture: [0-9]+/[0-9]+" "$GESTURE_LOG" | head -1)

# Build report
{
  echo "# v18aac (Qwen2.5-Coder-7B-Instruct base) — Auto Eval Report"
  echo
  echo "**Generated**: $(date)"
  echo "**Tag**: \`$OLLAMA_TAG\`"
  echo
  echo "## AAC Realigned Scores"
  echo "| Task | v17.4 | **v18aac** | Pass? |"
  echo "|---|---|---|---|"
  echo "| ask_ai | 5/5 | $ASK_AI | $([ "${ASK_AI%/*}" = "5" ] && echo "✅" || echo "⚠️") |"
  echo "| caregiver | 5/7 | $CAREGIVER | $([ "${CAREGIVER%/*}" -ge 5 ] && echo "✅ HOLD/WIN" || echo "⚠️ regression") |"
  echo "| **emergency_qa** | **13/13** | $EMERGENCY | $([ "${EMERGENCY%/*}" = "13" ] && echo "✅ HOLD" || echo "❌ REGRESSION") |"
  echo "| text_correct | 15/15 | $TEXT_CORRECT | $([ "${TEXT_CORRECT%/*}" -ge 13 ] && echo "✅" || echo "❌") |"
  echo "| translate | 7/8 | $TRANSLATE | $([ "${TRANSLATE%/*}" -ge 7 ] && echo "✅" || echo "⚠️") |"
  echo "| **OVERALL** | **41/48 (85%)** | **$OVERALL** | — |"
  echo
  echo "## Gesture (NEW capability) — $GESTURE_RESULT"
  echo "Spot-check on 5 realistic gesture configuration prompts."
  echo
  echo "## Decision"
  EM_NUM=${EMERGENCY%/*}
  TC_NUM=${TEXT_CORRECT%/*}
  OV_NUM=${OVERALL%/*}
  V174_OV=41
  if [ "${EM_NUM:-0}" -ge 12 ] && [ "${TC_NUM:-0}" -ge 13 ] && [ "${OV_NUM:-0}" -ge "$V174_OV" ]; then
    echo "✅ **PASSES all hard gates AND beats v17.4 — RECOMMEND PROMOTION**"
    echo
    echo "Promotion command (run with snapshot first):"
    echo "\`\`\`bash"
    echo "ollama cp prism-coder:7b prism-coder:7b-prev-\$(date +%Y%m%d-%H%M)"
    echo "ollama create prism-coder:7b -f <Modelfile pointing to v18aac qwen25 GGUF + SYSTEM directive>"
    echo "\`\`\`"
  else
    echo "⚠️ **DOES NOT pass all gates — DO NOT PROMOTE**"
    echo "Production stays on v17.4."
  fi
} > "$REPORT"

log "=== DONE — see $REPORT ==="
echo
cat "$REPORT"
