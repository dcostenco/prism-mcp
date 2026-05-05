#!/usr/bin/env bash
# Deploy + evaluate the v18 polish runs (qwen3-8b + 14b).
#
# Idempotent: skips fetch/merge/quantize/register if Ollama tag exists,
# skips eval if logs already exist with valid scores.
#
# Both polishes load final_polish_adapter from their respective volumes.
# Production prism-coder:7b is NOT touched.

set -uo pipefail

LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
SUMMARY=/tmp/v18_polish_comparison.md
LOG=/tmp/v18_polish_deploy.log
SCORES_DIR=/tmp/v18_polish_scores
mkdir -p "$SCORES_DIR"

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

deploy_and_eval_one() {
  local volume_name="$1"
  local base_model_hf="$2"
  local ckpt_name="$3"
  local local_subdir="$4"
  local ollama_tag="$5"
  local bfcl_log="/tmp/bfcl_${ollama_tag//[\/:]/_}.log"
  local aac_log="/tmp/aac_${ollama_tag//[\/:]/_}.log"

  log "========== $volume_name/$ckpt_name → $ollama_tag =========="

  local adapter_local="${BASE_DIR}/models/${local_subdir}/${ckpt_name}"
  local fused_dir="${BASE_DIR}/models/${local_subdir}-fused"
  local gguf_out="${BASE_DIR}/models/${local_subdir}-q4km.gguf"

  if ! ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null; then
    log "[1/5] fetch from Modal volume $volume_name..."
    mkdir -p "$(dirname "$adapter_local")"
    modal volume get "$volume_name" "$ckpt_name" "$(dirname "$adapter_local")/" --force 2>&1 | tail -3 | tee -a "$LOG"
    [ -f "$adapter_local/adapter_config.json" ] || { log "  FAIL — adapter_config.json missing (training may not be done yet)"; return 1; }
    [ -f "$adapter_local/adapter_model.safetensors" ] || { log "  FAIL — safetensors missing"; return 1; }

    log "[2/5] PEFT merge from base $base_model_hf..."
    rm -rf "$fused_dir"
    python3 v163_peft_merge.py "$base_model_hf" "$adapter_local" "$fused_dir" 2>&1 | tail -10 | tee -a "$LOG"
    [ -f "$fused_dir/config.json" ] || { log "  FAIL — merge incomplete"; return 1; }

    log "[3/5] convert to GGUF Q4_K_M..."
    local gguf_fp16="/tmp/${local_subdir}_fp16.gguf"
    PYTHONPATH="$LLAMA_CPP/gguf-py" python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
      "$fused_dir" --outfile "$gguf_fp16" --outtype f16 2>&1 | tail -5 | tee -a "$LOG"
    local quantize_bin="$LLAMA_CPP/build/bin/llama-quantize"
    [ -x "$quantize_bin" ] || quantize_bin="$LLAMA_CPP/llama-quantize"
    "$quantize_bin" "$gguf_fp16" "$gguf_out" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
    rm -f "$gguf_fp16"
    [ -f "$gguf_out" ] || { log "  FAIL — quantize"; return 1; }

    log "[4/5] register Ollama tag $ollama_tag..."
    cat > /tmp/Modelfile.${local_subdir} <<EOF
FROM $gguf_out

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
    ollama create "$ollama_tag" -f /tmp/Modelfile.${local_subdir} 2>&1 | tail -3 | tee -a "$LOG"
    ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null || { log "  FAIL — ollama tag missing"; return 1; }
  else
    log "[1-4/5] SKIP — Ollama tag $ollama_tag already registered"
  fi

  log "[5/5] eval — custom BFCL + AAC realigned..."
  local bfcl_score=""
  if [ -s "$bfcl_log" ]; then
    bfcl_score=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$bfcl_log" | tail -1 | grep -oE "[0-9.]+" | head -1)
  fi
  if [ -z "$bfcl_score" ]; then
    python3 bfcl_eval.py --model "$ollama_tag" 2>&1 | tee "$bfcl_log" > /dev/null
    bfcl_score=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$bfcl_log" | tail -1 | grep -oE "[0-9.]+" | head -1)
  else
    log "  BFCL cached"
  fi
  log "  BFCL: ${bfcl_score}%"

  if [ ! -s "$aac_log" ] || ! grep -q "OVERALL" "$aac_log"; then
    MODEL="$ollama_tag" python3 run_aac_realigned.py 2>&1 | tee "$aac_log" > /dev/null
  else
    log "  AAC cached"
  fi
  local aac_results=$(grep -oE "(ask_ai|caregiver|emergency_qa|text_correct|translate)[[:space:]]+[0-9]+/[[:space:]]*[0-9]+" "$aac_log" | sed -E 's/[[:space:]]+/ /g; s|/ |/|' | tr '\n' ' ')

  printf 'BFCL=%s | AAC: %s\n' "$bfcl_score" "$aac_results" \
    > "${SCORES_DIR}/${ollama_tag//[\/:]/_}.txt"
  log "  $ollama_tag done"
  log ""
}

log "=== v18 polish deploy + eval ==="

# qwen3-8b polish
deploy_and_eval_one "prism-v18coder-qwen3-8b-polish" "Qwen/Qwen3-8B" \
  "final_polish_adapter" "v18polish-q3-8b" "prism-coder:8b-v18polish-q3" || true

# 14b polish
deploy_and_eval_one "prism-v18coder-14b-polish" "Qwen/Qwen2.5-Coder-14B-Instruct" \
  "final_polish_adapter" "v18polish-14b" "prism-coder:14b-v18polish" || true

# Build comparison report
{
  echo "# v18 Polish Deploy Comparison"
  echo
  echo "**Generated**: $(date)"
  echo
  echo "| Tag | Result |"
  echo "|---|---|"
  for tag in "prism-coder:8b-v18polish-q3" "prism-coder:14b-v18polish"; do
    key="${tag//[\/:]/_}"
    if [ -f "${SCORES_DIR}/${key}.txt" ]; then
      score=$(cat "${SCORES_DIR}/${key}.txt")
    else
      score="NOT EVALUATED (training may not be complete)"
    fi
    echo "| \`${tag}\` | $score |"
  done
  echo
  echo "## Reference baselines"
  echo "- Current prod (\`prism-coder:7b\` = v18max): BFCL 47.2%, AAC ask_ai 5/5 caregiver 6/7 emergency 13/13 text_correct 15/15 translate 8/8"
  echo "- Top candidate (\`prism-coder:7b-v18clean-epoch0\`): BFCL 88.1%, AAC same as prod + targeted 20/20"
} > "$SUMMARY"

log "=== ALL DONE — see $SUMMARY ==="
echo
cat "$SUMMARY"
