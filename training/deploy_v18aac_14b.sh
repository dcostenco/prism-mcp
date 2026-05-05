#!/usr/bin/env bash
# Deploy v18aac-14b (DoRA r=384, 3 epochs, Qwen2.5-Coder-14B-Instruct base)
# → Ollama tag prism-aac:14b for iPad Pro M4 + desktop offline use

set -uo pipefail

BASE_DIR=/Users/admin/prism/training
LLAMA_CPP=/Users/admin/llama.cpp
LOG=/tmp/deploy_v18aac_14b.log

ADAPTER_DIR=$BASE_DIR/models/v18aac_14b_adapter
FUSED_DIR=$BASE_DIR/models/prism-v18aac-14b-fused
GGUF=$BASE_DIR/models/prism-v18aac-14b-q4km.gguf
OLLAMA_TAG=prism-aac:14b

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== Fetch final adapter from prism-v18aac-14b volume ==="
mkdir -p "$ADAPTER_DIR"
modal volume get prism-v18aac-14b final_adapter "$ADAPTER_DIR/" --force 2>&1 | tail -3 | tee -a "$LOG"
[ -f "$ADAPTER_DIR/final_adapter/adapter_config.json" ] || { log "FATAL: adapter not fetched"; exit 1; }

log "=== PEFT merge (Qwen2.5-Coder-14B-Instruct + DoRA) → BF16 ==="
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py "Qwen/Qwen2.5-Coder-14B-Instruct" "$ADAPTER_DIR/final_adapter" "$FUSED_DIR" 2>&1 | tail -5 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || { log "FATAL: merge incomplete"; exit 1; }

log "=== Convert HF → GGUF F16 ==="
GGUF_FP16=/tmp/v18aac_14b_fp16.gguf
PYTHONPATH=$LLAMA_CPP/gguf-py python3 $LLAMA_CPP/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

log "=== Quantize Q4_K_M ==="
QBIN=$LLAMA_CPP/build/bin/llama-quantize
[ -x "$QBIN" ] || QBIN=$LLAMA_CPP/llama-quantize
"$QBIN" "$GGUF_FP16" "$GGUF" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
rm -f "$GGUF_FP16"

log "=== Register Ollama tag $OLLAMA_TAG ==="
cat > /tmp/Modelfile.v18aac-14b <<EOF
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
ollama create "$OLLAMA_TAG" -f /tmp/Modelfile.v18aac-14b 2>&1 | tail -3 | tee -a "$LOG"

log "=== Smoke test ==="
RESP=$(curl -s http://localhost:11434/api/generate -d "$(jq -n --arg m "$OLLAMA_TAG" '{model:$m,system:"You are Prism AAC Assistant.",prompt:"Map smile to confirm.",stream:false,options:{num_predict:120,temperature:0.0}}')" | jq -r '.response // ""')
echo "  Response: ${RESP:0:300}" | tee -a "$LOG"

log "=== DONE — tag $OLLAMA_TAG ready ==="
ollama list 2>/dev/null | grep -E "prism-aac|prism-coder:7b" | head -10 | tee -a "$LOG"
