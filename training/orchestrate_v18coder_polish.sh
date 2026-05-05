#!/usr/bin/env bash
# v18coder Variant-E orchestrator: base → polish → deploy + eval.
#
# Pipeline:
#   1. Wait for prism-v18coder-14b-sft to stop, then launch 14b polish.
#   2. Wait for prism-v18coder-qwen3-8b-sft to stop, then launch 8b polish.
#   3. Wait for both polish runs to stop.
#   4. Deploy + eval all 4 adapters: base+polish × {14B, 8B}.
#   5. Write comparison report to /tmp/v18coder_polish_comparison.md.

set -uo pipefail

LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
SUMMARY=/tmp/v18coder_polish_comparison.md
LOG=/tmp/v18coder_polish_orchestrator.log
SCORES_DIR=/tmp/v18coder_polish_scores
mkdir -p "$SCORES_DIR"

cd "$BASE_DIR"
source venv/bin/activate 2>/dev/null || true
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_app_stop() {
  # Returns when the named Modal app reaches a terminal state.
  # Treats: "stopped" → done. "absent" (not in list) → done (Modal trims old apps).
  # Anything else (ephemeral/running/deployed/...) → sleep and re-check.
  local app_name="$1"
  while true; do
    local row=$(COLUMNS=200 modal app list 2>/dev/null | grep -F "$app_name" | head -1)
    if [ -z "$row" ]; then
      log "[$app_name] state: absent (terminal)"
      return 0
    fi
    local state=$(echo "$row" | awk -F'│' '{print $4}' | xargs)
    log "[$app_name] state: ${state:-unknown}"
    case "$state" in
      "stopped") return 0 ;;
      *) sleep 120 ;;
    esac
  done
}

launch_polish() {
  local script="$1"
  local app_name="$2"
  if COLUMNS=200 modal app list 2>/dev/null | grep -F "$app_name" | grep -qE "ephemeral|running"; then
    log "[$app_name] already running — skip launch"
    return 0
  fi
  log "[$app_name] launching $script (detached)..."
  ( cd "$BASE_DIR" && modal run --detach "$script" >> "$LOG" 2>&1 ) &
  sleep 30
}

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
  fi
  log "  custom BFCL: ${bfcl_score}%"

  if [ ! -s "$aac_log" ] || ! grep -q "OVERALL" "$aac_log"; then
    MODEL="$ollama_tag" python3 run_aac_realigned.py 2>&1 | tee "$aac_log" > /dev/null
  fi
  local aac_results=$(grep -oE "(ask_ai|caregiver|emergency_qa|text_correct|translate)[[:space:]]+[0-9]+/[[:space:]]*[0-9]+" "$aac_log" | sed -E 's/[[:space:]]+/ /g; s|/ |/|' | tr '\n' ' ')

  printf 'CustomBFCL=%s | AAC: %s\n' "$bfcl_score" "$aac_results" > "${SCORES_DIR}/${ollama_tag//[\/:]/_}.txt"
  log "  $ollama_tag done"
  log ""
}

# ─── Stage 1: parallel — each base watcher launches its polish independently ─
log "=== Stage 1 — parallel base watch + polish launch ==="

(
  wait_for_app_stop "prism-v18coder-14b-sft"
  launch_polish "modal_v18coder_14b_polish_sft.py" "prism-v18coder-14b-polish"
) &
PID_14B_BRANCH=$!

(
  wait_for_app_stop "prism-v18coder-qwen3-8b-sft"
  launch_polish "modal_v18coder_qwen3_8b_polish_sft.py" "prism-v18coder-qwen3-8b-polish"
) &
PID_8B_BRANCH=$!

wait $PID_14B_BRANCH $PID_8B_BRANCH
log "=== Stage 1 done — both polish runs launched ==="

# ─── Stage 2: wait for polish runs ───────────────────────────────────────
log "=== Stage 2 — wait for polish runs to finish ==="
wait_for_app_stop "prism-v18coder-14b-polish"
wait_for_app_stop "prism-v18coder-qwen3-8b-polish"

# ─── Stage 3: deploy + eval all 4 adapters ───────────────────────────────
log "=== Stage 3 — deploy + eval (4 adapters) ==="

# 14B base timed out at Modal 10h cap; using last saved checkpoint (step 5000, 67% trained)
deploy_and_eval_one "prism-v18coder-14b" "Qwen/Qwen2.5-Coder-14B-Instruct" \
  "v18coder_14b_run/checkpoint-5000" "v18coder-14b-base" "prism-coder:14b-v18coder-base" || true

deploy_and_eval_one "prism-v18coder-14b-polish" "Qwen/Qwen2.5-Coder-14B-Instruct" \
  "final_polish_adapter" "v18coder-14b-polish" "prism-coder:14b-v18coder-polish" || true

deploy_and_eval_one "prism-v18coder-qwen3-8b" "Qwen/Qwen3-8B" \
  "final_adapter" "v18coder-q3-8b-base" "prism-coder:8b-v18coder-q3-base" || true

deploy_and_eval_one "prism-v18coder-qwen3-8b-polish" "Qwen/Qwen3-8B" \
  "final_polish_adapter" "v18coder-q3-8b-polish" "prism-coder:8b-v18coder-q3-polish" || true

# ─── Stage 4: comparison report ──────────────────────────────────────────
{
  echo "# v18coder Polish Comparison: 14B vs 8B, base vs polish"
  echo
  echo "**Generated**: $(date)"
  echo
  echo "| Variant | Result |"
  echo "|---|---|"
  for tag in "prism-coder:14b-v18coder-base" "prism-coder:14b-v18coder-polish" "prism-coder:8b-v18coder-q3-base" "prism-coder:8b-v18coder-q3-polish"; do
    key="${tag//[\/:]/_}"
    if [ -f "${SCORES_DIR}/${key}.txt" ]; then
      score=$(cat "${SCORES_DIR}/${key}.txt")
    else
      score="NOT EVALUATED"
    fi
    echo "| \`${tag}\` | $score |"
  done
  echo
  echo "## v17.4 baseline"
  echo "Custom-BFCL=79.7% | AAC: ask_ai 5/5, caregiver 5/7+20/20, emergency_qa 13/13, text_correct 15/15, translate 7/8"
  echo
  echo "## Promotion guidance"
  echo "Pick the highest custom-BFCL where AAC gates do not regress."
  echo "- 8B winner → \`ollama cp <best> prism-coder:7b-coder\`"
  echo "- 14B winner → \`ollama cp <best> prism-coder:14b-coder\`"
  echo "- Production prism-coder:7b (= v18aac) is NOT touched."
} > "$SUMMARY"

log "=== ALL DONE — see $SUMMARY ==="
echo
cat "$SUMMARY"
