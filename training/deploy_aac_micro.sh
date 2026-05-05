#!/usr/bin/env bash
# Deploy qwen3-8b AAC micro-SFT adapter and run AAC + custom BFCL eval.

set -uo pipefail

LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
SUMMARY=/tmp/aac_micro_eval.md
LOG=/tmp/aac_micro_deploy.log

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VOLUME=prism-v18coder-qwen3-8b-aac-micro
CKPT=final_aac_micro_adapter
LOCAL_SUBDIR=aac-micro-q3-8b
OLLAMA_TAG=prism-coder:8b-aac-micro-q3
BASE_HF=Qwen/Qwen3-8B

ADAPTER_LOCAL="${BASE_DIR}/models/${LOCAL_SUBDIR}/${CKPT}"
FUSED_DIR="${BASE_DIR}/models/${LOCAL_SUBDIR}-fused"
GGUF_OUT="${BASE_DIR}/models/${LOCAL_SUBDIR}-q4km.gguf"
BFCL_LOG="/tmp/bfcl_${OLLAMA_TAG//[\/:]/_}.log"
AAC_LOG="/tmp/aac_${OLLAMA_TAG//[\/:]/_}.log"

if ! ollama list 2>/dev/null | grep -F "$OLLAMA_TAG" > /dev/null; then
  log "[1/5] fetch from $VOLUME..."
  mkdir -p "$(dirname "$ADAPTER_LOCAL")"
  modal volume get "$VOLUME" "$CKPT" "$(dirname "$ADAPTER_LOCAL")/" --force 2>&1 | tail -3 | tee -a "$LOG"
  [ -f "$ADAPTER_LOCAL/adapter_config.json" ] || { log "FAIL — adapter missing"; exit 1; }

  log "[2/5] PEFT merge..."
  rm -rf "$FUSED_DIR"
  python3 v163_peft_merge.py "$BASE_HF" "$ADAPTER_LOCAL" "$FUSED_DIR" 2>&1 | tail -10 | tee -a "$LOG"
  [ -f "$FUSED_DIR/config.json" ] || { log "FAIL — merge"; exit 1; }

  log "[3/5] convert to GGUF Q4_K_M..."
  GGUF_FP16=/tmp/${LOCAL_SUBDIR}_fp16.gguf
  PYTHONPATH="$LLAMA_CPP/gguf-py" python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -5 | tee -a "$LOG"
  QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"
  [ -x "$QUANT_BIN" ] || QUANT_BIN="$LLAMA_CPP/llama-quantize"
  "$QUANT_BIN" "$GGUF_FP16" "$GGUF_OUT" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
  rm -f "$GGUF_FP16"
  [ -f "$GGUF_OUT" ] || { log "FAIL — quantize"; exit 1; }

  log "[4/5] register Ollama tag $OLLAMA_TAG..."
  cat > /tmp/Modelfile.${LOCAL_SUBDIR} <<EOF
FROM $GGUF_OUT

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are Prism, a helpful AI assistant. Answer the user's question directly and concisely. If a question is conversational or factual, respond in plain text. Do not invent tools or function calls — only call functions when the user explicitly provides function signatures in <tools></tools> tags."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</tool_call>"
EOF
  ollama create "$OLLAMA_TAG" -f /tmp/Modelfile.${LOCAL_SUBDIR} 2>&1 | tail -3 | tee -a "$LOG"
else
  log "[1-4/5] SKIP — Ollama tag $OLLAMA_TAG already exists"
fi

log "[5/5] eval BFCL + AAC..."
python3 bfcl_eval.py --model "$OLLAMA_TAG" 2>&1 | tee "$BFCL_LOG" > /dev/null
BFCL_SCORE=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$BFCL_LOG" | tail -1 | grep -oE "[0-9.]+" | head -1)

MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tee "$AAC_LOG" > /dev/null
AAC_RESULTS=$(grep -oE "(ask_ai|caregiver|emergency_qa|text_correct|translate)[[:space:]]+[0-9]+/[[:space:]]*[0-9]+" "$AAC_LOG" | sed -E 's/[[:space:]]+/ /g; s|/ |/|' | tr '\n' ' ')

{
  echo "# qwen3-8b AAC micro-SFT eval"
  echo
  echo "**Tag**: \`$OLLAMA_TAG\`"
  echo "**BFCL**: ${BFCL_SCORE}%"
  echo "**AAC**: $AAC_RESULTS"
  echo
  echo "## Comparison"
  echo "| Model | BFCL | caregiver | emergency | text_correct | translate | ask_ai |"
  echo "|---|---|---|---|---|---|---|"
  echo "| qwen3-8b base | 45.3% | 2/7 | 8/13 | 7/15 | 4/8 | 3/5 |"
  echo "| qwen3-8b polish (broken) | 35.0% | 0/7 | 7/13 | 1/15 | 1/8 | 4/5 |"
  echo "| **qwen3-8b AAC micro** | **${BFCL_SCORE}%** | (see AAC) | | | | |"
  echo "| v18-clean epoch_0 (winner) | 88.1% | 6/7 | 13/13 | 15/15 | 8/8 | 5/5 |"
} > "$SUMMARY"

log "=== DONE — see $SUMMARY ==="
cat "$SUMMARY"
