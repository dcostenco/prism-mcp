#!/usr/bin/env bash
# v18coder A/B deploy + eval pipeline.
#
# Polls Modal for two parallel training runs to finish, then fetches the
# adapter from each, PEFT-merges to BF16, GGUF-quantizes Q4_K_M, registers
# Ollama tags, and runs Berkeley BFCL V4 + custom 64-test bfcl_eval.py +
# AAC realigned for both — produces a comparison report.
#
# Runs being compared:
#   A) prism-v18coder-14b-sft         → tag prism-coder:14b-v18coder
#   B) prism-v18coder-qwen3-8b-sft    → tag prism-coder:8b-v18coder-q3
#
# The winner gets promoted to prism-coder:7b-coder per HYBRID_RESEARCH.md
# (note: a 14B winner would be tagged prism-coder:14b-coder instead).

set -uo pipefail

LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
SUMMARY=/tmp/v18coder_ab_comparison.md
LOG=/tmp/v18coder_ab_deploy.log
SCORES_DIR=/tmp/v18coder_ab_scores
mkdir -p "$SCORES_DIR"

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ─── Wait for both Modal apps to finish ──────────────────────────────────
wait_for_app() {
  local app_name="$1"
  while true; do
    local state=$(COLUMNS=200 modal app list 2>/dev/null | grep -F "$app_name" | head -1 | awk -F'│' '{print $4}' | xargs)
    log "[$app_name] state: ${state:-unknown}"
    case "$state" in
      "stopped"|"") return 0 ;;
      *) sleep 120 ;;
    esac
  done
}

deploy_and_eval_one() {
  local volume_name="$1"     # Modal volume to fetch from
  local base_model_hf="$2"   # HF base model for PEFT merge
  local ckpt_name="$3"       # adapter dir name on volume
  local local_subdir="$4"    # local models/<subdir> destination
  local ollama_tag="$5"      # ollama tag to register
  local bfcl_log="/tmp/bfcl_${ollama_tag//[\/:]/_}.log"
  local aac_log="/tmp/aac_${ollama_tag//[\/:]/_}.log"
  local berkeley_log="/tmp/berkeley_v4_${ollama_tag//[\/:]/_}.log"

  log "========== $volume_name/$ckpt_name → $ollama_tag =========="

  local adapter_local="${BASE_DIR}/models/${local_subdir}/${ckpt_name}"
  local fused_dir="${BASE_DIR}/models/${local_subdir}-fused"
  local gguf_out="${BASE_DIR}/models/${local_subdir}-q4km.gguf"

  if ! ollama list 2>/dev/null | grep -F "$ollama_tag" > /dev/null; then
    log "[1/5] fetch from Modal volume $volume_name..."
    mkdir -p "$(dirname "$adapter_local")"
    modal volume get "$volume_name" "$ckpt_name" "$(dirname "$adapter_local")/" --force 2>&1 | tail -3 | tee -a "$LOG"
    [ -f "$adapter_local/adapter_config.json" ] || { log "  FAIL — adapter_config.json missing"; return 1; }
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
    log "  custom BFCL cached"
  fi
  log "  custom BFCL (Prism 64-test): ${bfcl_score}%"

  if [ ! -s "$aac_log" ] || ! grep -q "OVERALL" "$aac_log"; then
    MODEL="$ollama_tag" python3 run_aac_realigned.py 2>&1 | tee "$aac_log" > /dev/null
  else
    log "  AAC cached"
  fi
  local aac_results=$(grep -oE "(ask_ai|caregiver|emergency_qa|text_correct|translate)[[:space:]]+[0-9]+/[[:space:]]*[0-9]+" "$aac_log" | sed -E 's/[[:space:]]+/ /g; s|/ |/|' | tr '\n' ' ')

  # Berkeley V4: only attempt if a runner script exists; otherwise skip and note.
  local berkeley_score="N/A (no runner)"
  if [ -x "${BASE_DIR}/run_berkeley_v4.sh" ]; then
    if [ ! -s "$berkeley_log" ]; then
      MODEL="$ollama_tag" bash "${BASE_DIR}/run_berkeley_v4.sh" 2>&1 | tee "$berkeley_log" > /dev/null || true
    fi
    berkeley_score=$(grep -oE "Overall.*: [0-9.]+%" "$berkeley_log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1)
    [ -z "$berkeley_score" ] && berkeley_score="N/A"
  fi

  printf 'CustomBFCL=%s | BerkeleyV4=%s | AAC: %s\n' "$bfcl_score" "$berkeley_score" "$aac_results" \
    > "${SCORES_DIR}/${ollama_tag//[\/:]/_}.txt"
  log "  $ollama_tag done"
  log ""
}

log "=== v18coder A/B deploy + eval — waiting for Modal jobs ==="

wait_for_app "prism-v18coder-14b-sft"
wait_for_app "prism-v18coder-qwen3-8b-sft"

log "=== Both training jobs complete — beginning deploy + eval ==="

# 14B run — Qwen2.5-Coder-14B-Instruct base
deploy_and_eval_one "prism-v18coder-14b" "Qwen/Qwen2.5-Coder-14B-Instruct" \
  "epoch_1_adapter" "v18coder-14b-epoch1" "prism-coder:14b-v18coder-epoch1" || true
deploy_and_eval_one "prism-v18coder-14b" "Qwen/Qwen2.5-Coder-14B-Instruct" \
  "final_adapter"  "v18coder-14b-final"  "prism-coder:14b-v18coder-final" || true

# Qwen3-8B run — Qwen3-8B base
deploy_and_eval_one "prism-v18coder-qwen3-8b" "Qwen/Qwen3-8B" \
  "epoch_1_adapter" "v18coder-q3-8b-epoch1" "prism-coder:8b-v18coder-q3-epoch1" || true
deploy_and_eval_one "prism-v18coder-qwen3-8b" "Qwen/Qwen3-8B" \
  "final_adapter"  "v18coder-q3-8b-final"  "prism-coder:8b-v18coder-q3-final" || true

# ─── Build comparison report ──────────────────────────────────────────────
{
  echo "# v18coder A/B Comparison: 14B (Qwen2.5-Coder) vs 8B (Qwen3)"
  echo
  echo "**Generated**: $(date)"
  echo
  echo "| Checkpoint | Result |"
  echo "|---|---|"
  for tag in "prism-coder:14b-v18coder-epoch1" "prism-coder:14b-v18coder-final" "prism-coder:8b-v18coder-q3-epoch1" "prism-coder:8b-v18coder-q3-final"; do
    key="${tag//[\/:]/_}"
    if [ -f "${SCORES_DIR}/${key}.txt" ]; then
      score=$(cat "${SCORES_DIR}/${key}.txt")
    else
      score="NOT EVALUATED"
    fi
    echo "| \`${tag}\` | $score |"
  done
  echo
  echo "## v17.4 baseline (current AAC production)"
  echo "Custom-BFCL=79.7% | AAC: ask_ai 5/5, caregiver 5/7+20/20, emergency_qa 13/13, text_correct 15/15, translate 7/8"
  echo
  echo "## Public BFCL V4 7-9B leaderboard (snapshot 2026-04-12)"
  echo "- xLAM-2-8b-fc-r: 46.68% Overall"
  echo "- BitAgent-Bounty-8B: 46.23%"
  echo "- Qwen3-8B (FC): 42.57% (the base of run B)"
  echo "- ToolACE-2-8B (FC): 42.44%"
  echo
  echo "## Promotion guidance"
  echo "1. Pick the checkpoint with the highest Berkeley V4 OR custom BFCL where AAC gates don't regress."
  echo "2. If 8B wins: \`ollama cp <best-8b-tag> prism-coder:7b-coder\`"
  echo "3. If 14B wins: \`ollama cp <best-14b-tag> prism-coder:14b-coder\`  (different size class)"
  echo "4. Production prism-coder:7b (= v18aac) is NOT touched by this script."
} > "$SUMMARY"

log "=== ALL DONE — see $SUMMARY ==="
echo
cat "$SUMMARY"
