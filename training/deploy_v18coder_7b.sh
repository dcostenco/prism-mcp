#!/usr/bin/env bash
# Deploy v18coder-7b (BFCL-optimized) → Ollama tag prism-coder:7b-v18bfcl
# Used by Prism Coder IDE.

set -uo pipefail

BASE_DIR=/Users/admin/prism/training
LLAMA_CPP=/Users/admin/llama.cpp
LOG=/tmp/deploy_v18coder_7b.log

ADAPTER_DIR=$BASE_DIR/models/v18coder_7b_adapter
FUSED_DIR=$BASE_DIR/models/prism-v18coder-7b-fused
GGUF=$BASE_DIR/models/prism-v18coder-7b-q4km.gguf
OLLAMA_TAG=prism-coder:7b-v18bfcl

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== Fetch final adapter from prism-v18coder volume ==="
mkdir -p "$ADAPTER_DIR"
modal volume get prism-v18coder final_adapter "$ADAPTER_DIR/" --force 2>&1 | tail -3 | tee -a "$LOG"
[ -f "$ADAPTER_DIR/final_adapter/adapter_config.json" ] || { log "FATAL: adapter not fetched"; exit 1; }

log "=== PEFT merge → BF16 ==="
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py "Qwen/Qwen2.5-Coder-7B-Instruct" "$ADAPTER_DIR/final_adapter" "$FUSED_DIR" 2>&1 | tail -5 | tee -a "$LOG"
[ -f "$FUSED_DIR/config.json" ] || { log "FATAL: merge incomplete"; exit 1; }

log "=== Convert HF → GGUF F16 ==="
GGUF_FP16=/tmp/v18coder_7b_fp16.gguf
PYTHONPATH=$LLAMA_CPP/gguf-py python3 $LLAMA_CPP/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

log "=== Quantize Q4_K_M ==="
QBIN=$LLAMA_CPP/build/bin/llama-quantize
[ -x "$QBIN" ] || QBIN=$LLAMA_CPP/llama-quantize
"$QBIN" "$GGUF_FP16" "$GGUF" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
rm -f "$GGUF_FP16"

log "=== Register Ollama tag $OLLAMA_TAG ==="
cat > /tmp/Modelfile.v18coder-7b <<EOF
FROM $GGUF

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are Prism Coder, a focused coding assistant. Write clear, correct code. Prefer concise answers; show diffs when modifying files."""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</tool_call>"
EOF
ollama create "$OLLAMA_TAG" -f /tmp/Modelfile.v18coder-7b 2>&1 | tail -3 | tee -a "$LOG"

log "=== Smoke test ==="
RESP=$(curl -s http://localhost:11434/api/generate -d "$(jq -n --arg m "$OLLAMA_TAG" '{model:$m,prompt:"Write a Python one-liner to reverse a string.",stream:false,options:{num_predict:120,temperature:0.0}}')" | jq -r '.response // ""')
echo "  Response: ${RESP:0:300}" | tee -a "$LOG"

log "=== DONE — tag $OLLAMA_TAG ready ==="
ollama list 2>/dev/null | grep -E "v18bfcl|prism-coder:7b" | head -5 | tee -a "$LOG"
