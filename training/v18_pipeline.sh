#!/usr/bin/env bash
# v18 autonomous platform pipeline.
#
# Steps:
#   0. Generate platform data (video / TTS / persona / word_predict / multimodal)
#      via Modal teacher endpoint into data/v18_platform/
#   1. Merge train_v17 + external + caregiver-boost + platform → data/train_v18.jsonl
#   2. Upload train_v18.jsonl to Modal volume prism-sft-data
#   3. Submit modal_v18_sft.py::train (detached) — capture app id
#   4. Wait for training to finish (poll modal app list)
#   5. Fetch adapter via modal_v18_sft.py::fetch
#   6. PEFT merge → GGUF Q4_K_M → Ollama tag prism-coder:7b-v18
#   7. AAC realigned eval + BFCL official eval
#   8. Final report at /tmp/v18_pipeline_report.md
#
# Stops on the first hard failure with a message in the report.

set -uo pipefail

LOG=/tmp/v18_pipeline.log
REPORT=/tmp/v18_pipeline_report.md
TRAIN_DATA=/Users/admin/prism/training/data/train_v18.jsonl
ADAPTER_LOCAL=/Users/admin/prism/training/models/v18_modal/v18_adapter
FUSED_DIR=/Users/admin/prism/training/models/prism-v18-fused
GGUF_OUT=/Users/admin/prism/training/models/prism-v18-fused-q4km.gguf
MODELFILE=/tmp/Modelfile.v18
OLLAMA_TAG=prism-coder:7b-v18
LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
START_TS=$(date +%s)

cd "$BASE_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  {
    echo "# v18 Pipeline — FAILED"
    echo
    echo "**Stopped at**: $(date)"
    echo "**Reason**: $1"
    echo
    echo "## Log tail"
    echo '```'
    tail -50 "$LOG"
    echo '```'
  } > "$REPORT"
  exit 1
}

: > "$LOG"
log "=== v18 autonomous platform pipeline starting ==="

source venv/bin/activate 2>/dev/null || true

# ─── 0. Generate platform data (video/TTS/persona/word_predict/multimodal) ──
log "[0/8] Generating platform data via Modal teacher endpoint..."
# Teacher endpoint must be deployed first:
#   modal deploy modal_teacher_endpoint.py
# The script is resumable — already-generated rows are kept unless --force.
python3 generate_video_tts_data.py 2>&1 | tail -40 | tee -a "$LOG" || \
  fail "platform data generation failed"

# Verify all 5 platform files are non-empty.
for cat in video_script_gen tts_ssml_control voice_persona_pick word_predict_aac multimodal_tool_route; do
  f="data/v18_platform/${cat}.jsonl"
  [ -s "$f" ] || fail "platform file $f missing or empty"
  rows=$(wc -l < "$f")
  log "  $cat: $rows rows"
done

# ─── 1. Merge train_v18.jsonl ───────────────────────────────────────────────
log "[1/8] Merging train_v17 + external + caregiver-boost + platform → $TRAIN_DATA"
python3 merge_v18_data.py --out "$TRAIN_DATA" 2>&1 | tail -30 | tee -a "$LOG"
[ -s "$TRAIN_DATA" ] || fail "merge produced empty output"
total_rows=$(wc -l < "$TRAIN_DATA")
log "  train_v18.jsonl: ${total_rows} rows"

# ─── 2. Upload to Modal volume ──────────────────────────────────────────────
log "[2/8] Uploading $TRAIN_DATA to prism-sft-data..."
modal volume put prism-sft-data "$TRAIN_DATA" /train_v18.jsonl --force 2>&1 | tail -3 | tee -a "$LOG"
modal volume ls prism-sft-data 2>&1 | grep -q "train_v18.jsonl" || fail "upload verification failed — train_v18.jsonl not found in prism-sft-data"

# ─── 3. Submit training (detached) ──────────────────────────────────────────
log "[3/8] Submitting v18 SFT to Modal H100..."
modal run --detach modal_v18_sft.py::train 2>&1 | tee -a "$LOG" | tail -20
sleep 5
TRAIN_APP=$(modal app list 2>/dev/null | grep "prism-v18-sft" | grep -v "stopped\|stopping" | awk -F'│' '{print $2}' | xargs | head -1)
log "  training app: $TRAIN_APP"
[ -n "$TRAIN_APP" ] || fail "could not determine training app id"

# ─── 4. Wait for training ───────────────────────────────────────────────────
log "[4/8] Waiting for v18 training (6000 steps) to complete..."
while :; do
  state=$(modal app list 2>/dev/null | grep "$TRAIN_APP" | awk -F'│' '{print $4}' | xargs)
  step=$(modal app logs "$TRAIN_APP" 2>/dev/null | grep -oE '[0-9]+/6000' | tail -1)
  if echo "$state" | grep -q "stopped"; then
    log "  training complete (state=stopped, last step=$step)"
    break
  fi
  if [ -z "$state" ]; then
    log "  training app no longer in list — assuming finished"
    break
  fi
  log "  state=$state step=$step — waiting 90s"
  sleep 90
done

# ─── 5. Fetch adapter + gate report ─────────────────────────────────────────
log "[5/8] Fetching adapter..."
mkdir -p "$(dirname "$ADAPTER_LOCAL")"
modal volume get prism-v18 v18_adapter "$(dirname "$ADAPTER_LOCAL")" --force 2>&1 | tail -2 | tee -a "$LOG"
modal volume get prism-v18 v18_gate.json /tmp/v18_gate.json --force 2>&1 | tail -1 | tee -a "$LOG"
[ -d "$ADAPTER_LOCAL" ] || fail "adapter dir missing at $ADAPTER_LOCAL"
gate_pass=$(python3 -c "import json; print(json.load(open('/tmp/v18_gate.json')).get('pass', False))" 2>/dev/null || echo False)
log "  in-container gate: pass=$gate_pass"

# ─── 6. Fuse via PEFT + GGUF + Ollama ───────────────────────────────────────
log "[6/8] Fusing PEFT adapter + converting to GGUF + registering Ollama..."
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "$ADAPTER_LOCAL" \
  "$FUSED_DIR" 2>&1 | tail -25 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || fail "fusion missing config.json"
log "  fused -> $FUSED_DIR"

GGUF_FP16=/tmp/v18_fp16.gguf
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -10 | tee -a "$LOG"
[ -f "$GGUF_FP16" ] || fail "f16 GGUF not produced"
QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || QUANTIZE_BIN="$LLAMA_CPP/llama-quantize"
"$QUANTIZE_BIN" "$GGUF_FP16" "$GGUF_OUT" Q4_K_M 2>&1 | tail -8 | tee -a "$LOG"
[ -f "$GGUF_OUT" ] || fail "Q4_K_M GGUF not produced"
rm -f "$GGUF_FP16"
log "  GGUF -> $GGUF_OUT ($(du -h "$GGUF_OUT" | cut -f1))"

cat > "$MODELFILE" <<EOF
FROM $GGUF_OUT

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
EOF
ollama create "$OLLAMA_TAG" -f "$MODELFILE" 2>&1 | tail -5 | tee -a "$LOG"
ollama list | grep "v18" | tee -a "$LOG" || fail "Ollama tag not registered"

# ─── 7. AAC realigned + BFCL official eval ──────────────────────────────────
log "[7/8] Running AAC realigned eval against $OLLAMA_TAG..."
MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tail -40 | tee -a "$LOG" || true
AAC_RESULT=$(ls -t results/aac_realigned_*v18*.json 2>/dev/null | head -1)

log "[8/8] Running BFCL official on Modal..."
modal run --detach run_bfcl_official.py::main 2>&1 | tail -10 | tee -a "$LOG" || true
log "  BFCL submitted detached. Polling..."
for i in $(seq 1 80); do
  if modal volume ls bfcl-results 2>/dev/null | grep -q "scores.txt"; then
    modal run run_bfcl_official.py::fetch 2>&1 | tail -3 | tee -a "$LOG"
    break
  fi
  sleep 60
done

# ─── Report ─────────────────────────────────────────────────────────────────
END_TS=$(date +%s)
ELAPSED=$(( (END_TS - START_TS) / 60 ))
{
  echo "# v18 Pipeline — FINISHED"
  echo
  echo "**Started**: $(date -r "$START_TS")"
  echo "**Finished**: $(date)"
  echo "**Elapsed**: ${ELAPSED} min"
  echo
  echo "## Artifacts"
  echo "- Training data: \`$TRAIN_DATA\` ($(wc -l < "$TRAIN_DATA") rows)"
  echo "- Steps trained: 6000 (~2.7 epochs over ~36K platform corpus)"
  echo "- Adapter: \`$ADAPTER_LOCAL\`"
  echo "- Fused model: \`$FUSED_DIR\`"
  echo "- GGUF: \`$GGUF_OUT\` ($(du -h "$GGUF_OUT" 2>/dev/null | cut -f1))"
  echo "- Ollama tag: \`$OLLAMA_TAG\`"
  echo
  echo "## In-container reliability gate"
  echo '```json'
  cat /tmp/v18_gate.json 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(no gate.json)"
  echo '```'
  echo
  echo "## AAC realigned eval"
  if [ -n "${AAC_RESULT:-}" ] && [ -f "$AAC_RESULT" ]; then
    echo "Result: \`$AAC_RESULT\`"
    python3 -c "
import json
d = json.load(open('$AAC_RESULT'))
s = d.get('summary', d)
for k,v in s.items():
    print(f'- **{k}**: {v}')
" 2>/dev/null
  fi
  echo
  echo "## BFCL official"
  if [ -f bfcl_scores.txt ]; then
    echo '```'; cat bfcl_scores.txt; echo '```'
  else
    echo "  (BFCL still running)"
  fi
  echo
  echo "## Cutover gates (recommend promotion only if ALL pass)"
  echo "- BFCL official ≥ 65 %"
  echo "- AAC overall ≥ 90 %"
  echo "- Emergency = 13/13"
  echo "- Caregiver ≥ 85 %  ⚠️ v18 PRIMARY TARGET (v17 hit 57 %)"
  echo "- Tone-switch ≥ 80 %"
  echo "- Video script ≥ 75 %  (Video Composer module)"
  echo "- TTS SSML ≥ 80 %  (offline-first + online_premium fallback)"
  echo "- Voice persona ≥ 80 %"
  echo "- Word predict ≥ 80 %  (3K+ dictionary, multilingual)"
  echo "- Multimodal route ≥ 80 %  (camera / noise / voice tool calling)"
  echo
  echo "## Log tail"
  echo '```'; tail -60 "$LOG"; echo '```'
} > "$REPORT"

log "=== v18 pipeline complete; report at $REPORT ==="
