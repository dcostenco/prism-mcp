#!/usr/bin/env bash
# Deploy + evaluate ALL v18-clean checkpoints — pick the best.
#
# v18-clean saves 4 adapters: epoch_1, epoch_2, epoch_3, final.
# This script fetches each, merges to BF16, GGUF Q4_K_M, Ollama-tags it,
# runs the full eval suite, and produces a comparison table so we can
# pick the best checkpoint to promote.
#
# Each checkpoint is preserved as prism-coder:7b-v18clean-epoch{N}.
# Production tag (prism-coder:7b) is NOT touched by this script.

set -uo pipefail

LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
SUMMARY=/tmp/v18_clean_checkpoint_comparison.md
LOG=/tmp/v18_clean_deploy_all.log
SCORES_DIR=/tmp/v18clean_scores
mkdir -p "$SCORES_DIR"

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  exit 1
}

deploy_and_eval_one() {
  local ckpt_name="$1"   # e.g. "epoch_1_adapter" or "final_adapter"
  local tag_suffix="$2"  # e.g. "v18clean-epoch1" or "v18clean-final"
  local ollama_tag="prism-coder:7b-${tag_suffix}"

  log "========== $ckpt_name → $ollama_tag =========="

  local adapter_local="${BASE_DIR}/models/v18_clean/${ckpt_name}"
  local fused_dir="${BASE_DIR}/models/prism-${tag_suffix}-fused"
  local gguf_out="${BASE_DIR}/models/prism-${tag_suffix}-fused-q4km.gguf"
  local bfcl_log=/tmp/bfcl_v18clean_${tag_suffix}.log
  local aac_log=/tmp/aac_v18clean_${tag_suffix}.log

  # Idempotent fast-path: if Ollama tag and both eval logs already exist with valid
  # scores, just record them and return. Lets reruns finish in seconds.
  local existing_bfcl=""
  if ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null; then
    if [ -s "$bfcl_log" ]; then
      existing_bfcl=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$bfcl_log" | tail -1 | grep -oE "[0-9.]+" | head -1)
    fi
    if [ -n "$existing_bfcl" ] && [ -s "$aac_log" ] && grep -q "OVERALL" "$aac_log"; then
      log "  CACHED — Ollama tag + eval logs already present (BFCL ${existing_bfcl}%)"
      local cached_aac=$(grep -oE "(ask_ai|caregiver|emergency_qa|text_correct|translate)[[:space:]]+[0-9]+/[[:space:]]*[0-9]+" "$aac_log" | sed -E 's/[[:space:]]+/ /g; s|/ |/|' | tr '\n' ' ')
      printf 'BFCL=%s|AAC: %s\n' "$existing_bfcl" "$cached_aac" > "${SCORES_DIR}/${tag_suffix}.txt"
      return 0
    fi
  fi

  log "[1/5] fetch from Modal..."
  mkdir -p "$(dirname "$adapter_local")"
  modal volume get prism-v18-clean "$ckpt_name" "$(dirname "$adapter_local")/" --force 2>&1 | tail -3 | tee -a "$LOG"
  [ -f "$adapter_local/adapter_config.json" ] || { log "  SKIP — adapter not found"; return 1; }
  [ -f "$adapter_local/adapter_model.safetensors" ] || { log "  SKIP — safetensors not found"; return 1; }

  if ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null; then
    log "[2-4/5] SKIP merge/quantize/register — Ollama tag $ollama_tag already exists"
  else
    log "[2/5] PEFT merge..."
    rm -rf "$fused_dir"
    python3 v163_peft_merge.py "Qwen/Qwen2.5-Coder-7B-Instruct" "$adapter_local" "$fused_dir" 2>&1 | tail -10 | tee -a "$LOG"
    [ -f "$fused_dir/config.json" ] || { log "  FAIL — merge incomplete"; return 1; }

    log "[3/5] convert to GGUF Q4_K_M..."
    local gguf_fp16=/tmp/v18clean_${tag_suffix}_fp16.gguf
    PYTHONPATH="$LLAMA_CPP/gguf-py" python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
      "$fused_dir" --outfile "$gguf_fp16" --outtype f16 2>&1 | tail -5 | tee -a "$LOG"
    local quantize_bin="$LLAMA_CPP/build/bin/llama-quantize"
    [ -x "$quantize_bin" ] || quantize_bin="$LLAMA_CPP/llama-quantize"
    "$quantize_bin" "$gguf_fp16" "$gguf_out" Q4_K_M 2>&1 | tail -3 | tee -a "$LOG"
    rm -f "$gguf_fp16"
    [ -f "$gguf_out" ] || { log "  FAIL — quantize"; return 1; }

    log "[4/5] register Ollama tag $ollama_tag..."
    cat > /tmp/Modelfile.${tag_suffix} <<EOF
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
    ollama create "$ollama_tag" -f /tmp/Modelfile.${tag_suffix} 2>&1 | tail -3 | tee -a "$LOG"
    ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null || { log "  FAIL — ollama tag not registered"; return 1; }
  fi

  log "[5/5] eval — single BFCL run + AAC realigned..."
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

  printf 'BFCL=%s|AAC: %s\n' "$bfcl_score" "$aac_results" > "${SCORES_DIR}/${tag_suffix}.txt"
  log "  $tag_suffix done"
  log ""
}

log "=== v18-clean multi-checkpoint deploy + eval ==="
log "Production tag (prism-coder:7b) will NOT be touched."
log ""

# NOTE: epoch naming is off-by-one due to int(state.epoch) flooring in the
# training callback. Modal volume contents:
#   epoch_0_adapter = after actual epoch 1 (step 964,  loss 0.0341)
#   epoch_1_adapter = after actual epoch 2 (step 1928, loss 0.0339)
#   epoch_2_adapter = after actual epoch 3 (step 2892, loss 0.0335)
#   final_adapter   = end-of-training save
# Tag suffixes preserved for continuity (epoch1 already deployed/evaluated).
for spec in "epoch_0_adapter:v18clean-epoch0" "epoch_1_adapter:v18clean-epoch1" "epoch_2_adapter:v18clean-epoch2" "final_adapter:v18clean-final"; do
  ckpt_name="${spec%:*}"
  tag_suffix="${spec#*:}"
  deploy_and_eval_one "$ckpt_name" "$tag_suffix" || log "  ${ckpt_name} skipped"
done

# Build comparison report
{
  echo "# v18-clean Checkpoint Comparison"
  echo
  echo "**Generated**: $(date)"
  echo
  echo "| Checkpoint | Result |"
  echo "|---|---|"
  for tag in "v18clean-epoch0" "v18clean-epoch1" "v18clean-epoch2" "v18clean-final"; do
    if [ -f "${SCORES_DIR}/${tag}.txt" ]; then
      score=$(cat "${SCORES_DIR}/${tag}.txt")
    else
      score="NOT EVALUATED"
    fi
    echo "| \`prism-coder:7b-${tag}\` | $score |"
  done
  echo
  echo "## v17.2 baseline (current production)"
  echo "BFCL=82.2% | AAC: ask_ai 5/5, caregiver 6/7+20/20, emergency_qa 12/13, text_correct 15/15, translate 7/8"
  echo
  echo "## Promotion guidance"
  echo "1. Identify the checkpoint where BFCL is highest AND no AAC gate regresses"
  echo "2. Run \`bash run_full_eval_v17_2.sh MODEL_TAG=prism-coder:7b-<best>\` for 3-run statistical confidence"
  echo "3. If gates pass: \`ollama cp prism-coder:7b-<best> prism-coder:7b\`"
  echo "4. Rollback always available: \`ollama cp snap-prism-coder_7b-v17-2-20260503-0043:latest prism-coder:7b\`"
} > "$SUMMARY"

log "=== ALL DONE — see $SUMMARY ==="
echo
cat "$SUMMARY"
