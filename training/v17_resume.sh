#!/usr/bin/env bash
# v17 resume — skip steps 1-5 (already done): generation, fetch, merge,
# upload, train-submit. Pick up at step 6: poll for the already-running
# training app, then fuse → GGUF → Ollama → eval → report.

set -uo pipefail

LOG=/tmp/v17_pipeline.log
REPORT=/tmp/v17_pipeline_report.md
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
    echo "# v17 Pipeline (resume) — FAILED"
    echo
    echo "**Stopped at**: $(date)"
    echo "**Reason**: $1"
    echo
    echo "## Log tail"
    echo '```'; tail -50 "$LOG"; echo '```'
  } > "$REPORT"
  exit 1
}

log "=== v17 RESUME (skipping gen/merge/upload — train_v17.jsonl already on Modal) ==="

# ─── Find the most recent prism-v17-sft app id ──────────────────────────────
log "[6/10] Finding latest prism-v17-sft training app..."
TRAIN_APP=""
for i in 1 2 3 4 5; do
  TRAIN_APP=$(modal app list 2>/dev/null | grep "prism-v17" | grep -v "stopped\|stopping" | awk -F'│' '{print $2}' | xargs | head -1)
  [ -n "$TRAIN_APP" ] && break
  sleep 5
done
[ -n "$TRAIN_APP" ] || fail "no active prism-v17-sft app found — submit modal_v17_sft.py::train first"
log "  training app: $TRAIN_APP"

# ─── Wait for training to complete ──────────────────────────────────────────
log "[6/10] Waiting for v17 training (3000 steps) to complete..."
while :; do
  state=$(modal app list 2>/dev/null | grep "$TRAIN_APP" | awk -F'│' '{print $4}' | xargs)
  step=$(modal app logs "$TRAIN_APP" 2>/dev/null | grep -oE '[0-9]+/3000' | tail -1)
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

# ─── Fetch adapter + gate report ────────────────────────────────────────────
log "[7/10] Fetching adapter..."
mkdir -p "$(dirname "$ADAPTER_LOCAL")"
modal volume get prism-v17 v17_adapter "$(dirname "$ADAPTER_LOCAL")" --force 2>&1 | tail -2 | tee -a "$LOG"
modal volume get prism-v17 v17_gate.json /tmp/v17_gate.json --force 2>&1 | tail -1 | tee -a "$LOG"
[ -d "$ADAPTER_LOCAL" ] || fail "adapter dir missing at $ADAPTER_LOCAL"
gate_pass=$(python3 -c "import json; print(json.load(open('/tmp/v17_gate.json')).get('pass', False))" 2>/dev/null || echo False)
log "  in-container gate: pass=$gate_pass"

# ─── Fuse via PEFT + GGUF + Ollama ──────────────────────────────────────────
log "[8/10] Fusing PEFT adapter + converting to GGUF + registering Ollama..."
source venv/bin/activate 2>/dev/null || true
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "$ADAPTER_LOCAL" \
  "$FUSED_DIR" 2>&1 | tail -25 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || fail "fusion missing config.json"

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

# ─── AAC realigned + BFCL official eval ─────────────────────────────────────
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
  echo "# v17 Pipeline — FINISHED (3000-step run)"
  echo
  echo "**Resume started**: $(date -r "$START_TS")"
  echo "**Finished**: $(date)"
  echo "**Resume elapsed**: ${ELAPSED} min"
  echo
  echo "## Artifacts"
  echo "- Training data: \`$TRAIN_DATA\` ($(wc -l < "$TRAIN_DATA") rows)"
  echo "- Steps trained: 3000 (~3.5 epochs)"
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

log "=== v17 resume complete; report at $REPORT ==="
