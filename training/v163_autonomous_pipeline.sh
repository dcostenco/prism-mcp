#!/usr/bin/env bash
# v16.3 autonomous pipeline — runs unattended through training completion,
# fusion, GGUF export, Ollama register, AAC eval, BFCL eval, and BFCL
# submission. Logs every step to /tmp/v163_pipeline.log; final report at
# /tmp/v163_pipeline_report.md.
#
# Stops on the first hard failure with a clear message in the report.
# Intended to run while the user sleeps.

set -uo pipefail

LOG=/tmp/v163_pipeline.log
REPORT=/tmp/v163_pipeline_report.md
TRAIN_APP=ap-Y1g04LA0QvSxTKOtf79tqz
ADAPTER_LOCAL=/Users/admin/prism/training/models/v16_3_modal/v16_3_adapter
FUSED_DIR=/Users/admin/prism/training/models/prism-v16-3-fused
GGUF_OUT=/Users/admin/prism/training/models/prism-v16-3-fused-q4km.gguf
MODELFILE=/tmp/Modelfile.v16-3
OLLAMA_TAG=prism-coder:7b-v16-3
LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
START_TS=$(date +%s)

cd "$BASE_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  {
    echo "# v16.3 Pipeline — FAILED"
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
log "=== v16.3 autonomous pipeline starting ==="

# ─── 1. Wait for training to complete ────────────────────────────────────
log "[1/8] Waiting for v16.3 training (app $TRAIN_APP)..."
while :; do
  state=$(modal app list 2>/dev/null | grep "$TRAIN_APP" | awk -F'│' '{print $4}' | xargs)
  step=$(modal app logs "$TRAIN_APP" 2>/dev/null | grep -oE '[0-9]+/1500' | tail -1)
  if echo "$state" | grep -q "stopped"; then
    log "  training complete (state=stopped, last step=$step)"
    break
  fi
  log "  state=$state step=$step — waiting 90s"
  sleep 90
done

# ─── 2. Fetch reliability gate result ────────────────────────────────────
log "[2/8] Fetching v16_3_gate.json..."
modal volume get prism-v16-3 v16_3_gate.json /tmp/v163_gate.json --force 2>&1 | tail -2 | tee -a "$LOG"
gate_pass=$(python3 -c "import json; print(json.load(open('/tmp/v163_gate.json')).get('pass', False))" 2>/dev/null || echo False)
log "  gate.pass = $gate_pass"
if [ "$gate_pass" != "True" ]; then
  log "  WARN: in-container gate failed; proceeding to external eval anyway"
fi

# ─── 3. Fetch adapter ────────────────────────────────────────────────────
log "[3/8] Fetching adapter to $ADAPTER_LOCAL..."
mkdir -p "$(dirname "$ADAPTER_LOCAL")"
modal volume get prism-v16-3 v16_3_adapter "$(dirname "$ADAPTER_LOCAL")" --force 2>&1 | tail -2 | tee -a "$LOG"
[ -d "$ADAPTER_LOCAL" ] || fail "adapter dir missing at $ADAPTER_LOCAL"

# ─── 4. Fuse adapter into base model ─────────────────────────────────────
log "[4/8] Fusing adapter via mlx_lm..."
source venv/bin/activate 2>/dev/null || true
rm -rf "$FUSED_DIR"
mlx_lm.fuse \
  --model "models/prism-v12-fused" \
  --adapter-path "$ADAPTER_LOCAL" \
  --save-path "$FUSED_DIR" \
  --dequantize 2>&1 | tail -25 | tee -a "$LOG"
[ -d "$FUSED_DIR" ] || fail "fusion produced no output dir"
[ -f "$FUSED_DIR/config.json" ] || fail "fusion missing config.json"
log "  fused -> $FUSED_DIR"

# ─── 5. Convert to GGUF Q4_K_M ───────────────────────────────────────────
log "[5/8] Converting to GGUF Q4_K_M..."
[ -d "$LLAMA_CPP" ] || fail "llama.cpp not found at $LLAMA_CPP"
GGUF_FP16=/tmp/v163_fp16.gguf
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -15 | tee -a "$LOG"
[ -f "$GGUF_FP16" ] || fail "f16 GGUF not produced"

QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || QUANTIZE_BIN="$LLAMA_CPP/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || fail "llama-quantize binary not found"
"$QUANTIZE_BIN" "$GGUF_FP16" "$GGUF_OUT" Q4_K_M 2>&1 | tail -10 | tee -a "$LOG"
[ -f "$GGUF_OUT" ] || fail "Q4_K_M GGUF not produced"
log "  GGUF -> $GGUF_OUT ($(du -h "$GGUF_OUT" | cut -f1))"
rm -f "$GGUF_FP16"

# ─── 6. Register Ollama tag ──────────────────────────────────────────────
log "[6/8] Registering Ollama tag $OLLAMA_TAG..."
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
ollama create "$OLLAMA_TAG" -f "$MODELFILE" 2>&1 | tail -10 | tee -a "$LOG"
ollama list | grep "v16-3" | tee -a "$LOG" || fail "Ollama tag not registered"

# ─── 7. AAC realigned eval (local) ───────────────────────────────────────
log "[7/8] Running AAC realigned eval against $OLLAMA_TAG..."
MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tail -40 | tee -a "$LOG" || true
AAC_RESULT=$(ls -t results/aac_realigned_*.json 2>/dev/null | head -1)

# ─── 8. BFCL official eval (Modal) + submission staging ──────────────────
log "[8/8] Running BFCL official on Modal..."
modal run --detach run_bfcl_official.py::main 2>&1 | tail -10 | tee -a "$LOG" || true
log "  BFCL submitted detached. Fetching when ready..."
# Poll for results — BFCL takes 30-60min
for i in $(seq 1 80); do
  if modal volume ls bfcl-results 2>/dev/null | grep -q "scores.txt"; then
    modal run run_bfcl_official.py::fetch 2>&1 | tail -3 | tee -a "$LOG"
    break
  fi
  sleep 60
done

# ─── Final report ────────────────────────────────────────────────────────
END_TS=$(date +%s)
ELAPSED=$(( (END_TS - START_TS) / 60 ))

{
  echo "# v16.3 Pipeline — FINISHED"
  echo
  echo "**Started**: $(date -r "$START_TS")"
  echo "**Finished**: $(date)"
  echo "**Elapsed**: ${ELAPSED} min"
  echo
  echo "## Artifacts"
  echo "- Adapter: \`$ADAPTER_LOCAL\`"
  echo "- Fused model: \`$FUSED_DIR\`"
  echo "- GGUF: \`$GGUF_OUT\` ($(du -h "$GGUF_OUT" 2>/dev/null | cut -f1))"
  echo "- Ollama tag: \`$OLLAMA_TAG\`"
  echo
  echo "## In-container reliability gate"
  echo '```json'
  cat /tmp/v163_gate.json 2>/dev/null | python3 -m json.tool 2>/dev/null || cat /tmp/v163_gate.json 2>/dev/null
  echo '```'
  echo
  echo "## AAC realigned eval"
  if [ -n "${AAC_RESULT:-}" ] && [ -f "$AAC_RESULT" ]; then
    python3 -c "
import json
d = json.load(open('$AAC_RESULT'))
for k,v in d.get('summary', {}).items():
    print(f'- **{k}**: {v}')
" 2>/dev/null || echo "  (raw: $AAC_RESULT)"
  else
    echo "  (no result file found)"
  fi
  echo
  echo "## BFCL official"
  if [ -f bfcl_scores.txt ]; then
    echo '```'
    cat bfcl_scores.txt
    echo '```'
  else
    echo "  (BFCL still running on Modal — check \`modal volume ls bfcl-results\` later)"
  fi
  echo
  echo "## BFCL submission"
  echo "Custom handler: \`bfcl-submission/prism_coder.py\`"
  echo "To submit: copy handler to local bfcl-eval install and run \`bfcl generate --model prism-coder-7b-v16-3-FC\`"
  echo
  echo "## Next steps for human review"
  if [ -f bfcl_scores.txt ] && [ -n "${AAC_RESULT:-}" ]; then
    echo "1. Compare BFCL ≥ 78%, AAC ≥ 89%, emergency = 13/13 against thresholds"
    echo "2. If pass → run \`bash deploy_v16.sh\` (will need v12 → v16-3 retag) to promote"
    echo "3. If any gate fails → keep v12 in production, iterate on training data"
  fi
  echo
  echo "## Log tail"
  echo '```'
  tail -60 "$LOG"
  echo '```'
} > "$REPORT"

log "=== pipeline complete; report at $REPORT ==="
