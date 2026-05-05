#!/usr/bin/env bash
# Deploy v18aac-7b-MAX (r=512, 5 epochs) → Ollama tag prism-aac:7b-v18max
# After 51-test smoke passes, promotes to prism-coder:7b (the tag PrismAAC consumes).
# Also surfaces new tag for Synalux portal route-llm config.

set -uo pipefail

BASE_DIR=/Users/admin/prism/training
LLAMA_CPP=/Users/admin/llama.cpp
LOG=/tmp/deploy_v18aac_7b_max.log

ADAPTER_DIR=$BASE_DIR/models/v18aac_7b_max_adapter
FUSED_DIR=$BASE_DIR/models/prism-v18aac-7b-max-fused
GGUF=$BASE_DIR/models/prism-v18aac-7b-max-q4km.gguf
NEW_TAG=prism-aac:7b-v18max
PROD_TAG=prism-coder:7b

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== Fetch final adapter from prism-v18aac-7b-max volume ==="
mkdir -p "$ADAPTER_DIR"
modal volume get prism-v18aac-7b-max final_adapter "$ADAPTER_DIR/" --force 2>&1 | tail -3 | tee -a "$LOG"
[ -f "$ADAPTER_DIR/final_adapter/adapter_config.json" ] || { log "FATAL: adapter not fetched"; exit 1; }

log "=== PEFT merge → BF16 ==="
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py "Qwen/Qwen2.5-Coder-7B-Instruct" "$ADAPTER_DIR/final_adapter" "$FUSED_DIR" 2>&1 | tail -5 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || { log "FATAL: merge incomplete"; exit 1; }

log "=== Convert HF → GGUF F16 ==="
GGUF_FP16=/tmp/v18aac_7b_max_fp16.gguf
PYTHONPATH=$LLAMA_CPP/gguf-py python3 $LLAMA_CPP/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

log "=== Quantize Q4_K_M ==="
QBIN=$LLAMA_CPP/build/bin/llama-quantize
[ -x "$QBIN" ] || QBIN=$LLAMA_CPP/llama-quantize
"$QBIN" "$GGUF_FP16" "$GGUF" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
rm -f "$GGUF_FP16"

log "=== Register Ollama tag $NEW_TAG ==="
cat > /tmp/Modelfile.v18aac-7b-max <<EOF
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
ollama create "$NEW_TAG" -f /tmp/Modelfile.v18aac-7b-max 2>&1 | tail -3 | tee -a "$LOG"

log "=== Run 51-test production smoke ==="
SMOKE_LOG=/tmp/v18aac_7b_max_smoke.log
MODEL="$NEW_TAG" bash /tmp/test_production_v17_2.sh 2>&1 | tee "$SMOKE_LOG" > /dev/null || true
PASS=$(grep -E "^PASS:" "$SMOKE_LOG" | tail -1 | grep -oE "PASS: [0-9]+" | grep -oE "[0-9]+")
FAIL=$(grep -E "^PASS:" "$SMOKE_LOG" | tail -1 | grep -oE "FAIL: [0-9]+" | grep -oE "[0-9]+")
log "  Smoke result: $PASS pass / $FAIL fail"

if [ "${FAIL:-99}" -le 2 ]; then
  log "=== PROMOTE to $PROD_TAG (snapshot first) ==="
  SNAPSHOT_TAG="${PROD_TAG}-prev-$(date +%Y%m%d-%H%M)"
  ollama cp "$PROD_TAG" "$SNAPSHOT_TAG" 2>&1 | tail -2 | tee -a "$LOG"
  ollama create "$PROD_TAG" -f /tmp/Modelfile.v18aac-7b-max 2>&1 | tail -2 | tee -a "$LOG"
  log "Promoted. Previous prod snapshotted as $SNAPSHOT_TAG"
  log ""
  log "=== Synalux portal note ==="
  log "  New canonical AAC tag: $NEW_TAG (also aliased as $PROD_TAG)"
  log "  PrismAAC consumers automatically use $PROD_TAG via Ollama → no client change needed."
  log "  Synalux portal route-llm should map AAC intent → $NEW_TAG once vLLM cloud serving is live."
else
  log "=== HOLD — $FAIL test failures ($PASS passing) ==="
  log "Production tag $PROD_TAG NOT touched. Review $SMOKE_LOG before promoting."
fi

log "=== DONE ==="
ollama list 2>/dev/null | grep -E "v18max|prism-coder:7b|prism-aac" | head -10 | tee -a "$LOG"
