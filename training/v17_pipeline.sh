#!/usr/bin/env bash
# v17 autonomous pipeline.
#
# Steps:
#   1. Wait for SFT generation (already submitted as ap-QLUR0JRCdMzTBxhTMSghUT)
#   2. Fetch generated data into data/v16_gen_72b/ (overwrite the 3 deltas)
#   3. Merge into data/train_v17.jsonl via merge_v16_data.py
#   4. Upload train_v17.jsonl to Modal volume prism-sft-data
#   5. Submit modal_v17_sft.py::train (detached) — capture app id
#   6. Wait for training to finish (poll modal app list)
#   7. Fetch adapter via modal_v17_sft.py::fetch
#   8. PEFT merge → GGUF Q4_K_M → Ollama tag prism-coder:7b-v17
#   9. AAC realigned eval + BFCL official eval
#  10. Final report at /tmp/v17_pipeline_report.md
#
# Stops on the first hard failure with a message in the report.

set -uo pipefail

LOG=/tmp/v17_pipeline.log
REPORT=/tmp/v17_pipeline_report.md
GEN_APP=ap-QLUR0JRCdMzTBxhTMSghUT
TRAIN_DATA=/Users/admin/prism/training/data/train_v17.jsonl
ADAPTER_LOCAL=/Users/admin/prism/training/models/v17_modal/v17_adapter
FUSED_DIR=/Users/admin/prism/training/models/prism-v17-fused
GGUF_OUT=/Users/admin/prism/training/models/prism-v17-fused-q4km.gguf
MODELFILE=/tmp/Modelfile.v17
OLLAMA_TAG=prism-coder:7b-v17
LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
START_TS=$(date +%s)

cd "$BASE_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  {
    echo "# v17 Pipeline — FAILED"
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
log "=== v17 autonomous pipeline starting ==="

# ─── 1. Wait for SFT generation to complete ─────────────────────────────────
log "[1/10] Waiting for SFT generation (app $GEN_APP)..."
while :; do
  state=$(modal app list 2>/dev/null | grep "$GEN_APP" | awk -F'│' '{print $4}' | xargs)
  if echo "$state" | grep -q "stopped"; then
    log "  generation complete (state=stopped)"
    break
  fi
  if [ -z "$state" ]; then
    log "  app no longer in list — assuming finished"
    break
  fi
  log "  state=$state — waiting 60s"
  sleep 60
done

# ─── 2. Fetch generated data ────────────────────────────────────────────────
log "[2/10] Fetching generated data..."
mkdir -p data/v16_gen_72b
for cat in tone_switch caregiver_parse_extra bfcl_irrelevance; do
  modal volume get prism-sft-gen-72b "${cat}.jsonl" "data/v16_gen_72b/${cat}.jsonl" --force 2>&1 | tail -1 | tee -a "$LOG"
  if [ ! -s "data/v16_gen_72b/${cat}.jsonl" ]; then
    fail "fetched ${cat}.jsonl is empty or missing"
  fi
  rows=$(wc -l < "data/v16_gen_72b/${cat}.jsonl")
  log "  ${cat}: ${rows} rows"
done

# ─── 3. Merge into train_v17.jsonl ──────────────────────────────────────────
log "[3/10] Merging into $TRAIN_DATA..."
source venv/bin/activate 2>/dev/null || true
python3 merge_v16_data.py --out "$TRAIN_DATA" 2>&1 | tail -30 | tee -a "$LOG"
[ -s "$TRAIN_DATA" ] || fail "merge produced empty output"
total_rows=$(wc -l < "$TRAIN_DATA")
log "  train_v17.jsonl: ${total_rows} rows"

# ─── 4. Upload to Modal volume ──────────────────────────────────────────────
log "[4/10] Uploading $TRAIN_DATA to prism-sft-data..."
modal volume put prism-sft-data "$TRAIN_DATA" /train_v17.jsonl --force 2>&1 | tail -3 | tee -a "$LOG"

# ─── 5. Submit training (detached) ──────────────────────────────────────────
log "[5/10] Submitting v17 SFT to Modal H100..."
modal run --detach modal_v17_sft.py::train 2>&1 | tee -a "$LOG" | tail -20
sleep 5
TRAIN_APP=$(modal app list 2>/dev/null | grep "prism-v17-sft" | awk -F'│' '{print $2}' | xargs | head -1)
log "  training app: $TRAIN_APP"
[ -n "$TRAIN_APP" ] || fail "could not determine training app id"

# ─── 6. Wait for training ───────────────────────────────────────────────────
log "[6/10] Waiting for v17 training to complete..."
while :; do
  state=$(modal app list 2>/dev/null | grep "$TRAIN_APP" | awk -F'│' '{print $4}' | xargs)
  step=$(modal app logs "$TRAIN_APP" 2>/dev/null | grep -oE '[0-9]+/1500' | tail -1)
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

# ─── 7. Fetch adapter + gate report ─────────────────────────────────────────
log "[7/10] Fetching adapter..."
mkdir -p "$(dirname "$ADAPTER_LOCAL")"
modal volume get prism-v17 v17_adapter "$(dirname "$ADAPTER_LOCAL")" --force 2>&1 | tail -2 | tee -a "$LOG"
modal volume get prism-v17 v17_gate.json /tmp/v17_gate.json --force 2>&1 | tail -1 | tee -a "$LOG"
[ -d "$ADAPTER_LOCAL" ] || fail "adapter dir missing at $ADAPTER_LOCAL"
gate_pass=$(python3 -c "import json; print(json.load(open('/tmp/v17_gate.json')).get('pass', False))" 2>/dev/null || echo False)
log "  in-container gate: pass=$gate_pass"

# ─── 8. Fuse via PEFT + GGUF + Ollama ───────────────────────────────────────
log "[8/10] Fusing PEFT adapter + converting to GGUF + registering Ollama..."
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "$ADAPTER_LOCAL" \
  "$FUSED_DIR" 2>&1 | tail -25 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || fail "fusion missing config.json"
log "  fused -> $FUSED_DIR"

GGUF_FP16=/tmp/v17_fp16.gguf
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
ollama list | grep "v17" | tee -a "$LOG" || fail "Ollama tag not registered"

# ─── 9. AAC realigned + BFCL official eval ──────────────────────────────────
log "[9/10] Running AAC realigned eval against $OLLAMA_TAG..."
MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tail -40 | tee -a "$LOG" || true
AAC_RESULT=$(ls -t results/aac_realigned_*v17*.json 2>/dev/null | head -1)

log "[10/10] Running BFCL official on Modal..."
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
  echo "# v17 Pipeline — FINISHED"
  echo
  echo "**Started**: $(date -r "$START_TS")"
  echo "**Finished**: $(date)"
  echo "**Elapsed**: ${ELAPSED} min"
  echo
  echo "## Artifacts"
  echo "- Training data: \`$TRAIN_DATA\` ($(wc -l < "$TRAIN_DATA") rows)"
  echo "- Adapter: \`$ADAPTER_LOCAL\`"
  echo "- Fused model: \`$FUSED_DIR\`"
  echo "- GGUF: \`$GGUF_OUT\` ($(du -h "$GGUF_OUT" 2>/dev/null | cut -f1))"
  echo "- Ollama tag: \`$OLLAMA_TAG\`"
  echo
  echo "## In-container reliability gate"
  echo '```json'
  cat /tmp/v17_gate.json 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(no gate.json)"
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
  echo "- Caregiver ≥ 85 %"
  echo "- Tone-switch ≥ 80 %"
  echo
  echo "## Log tail"
  echo '```'; tail -60 "$LOG"; echo '```'
} > "$REPORT"

log "=== v17 pipeline complete; report at $REPORT ==="
