#!/usr/bin/env bash
# Auto-deploy v17.1 surgical adapter: fetch → merge → GGUF → Ollama tag → eval.
#
# Usage:
#   bash deploy_v17_1.sh
#
# Pre-conditions:
#   - Modal training app `prism-v17-1-surgical` has completed successfully
#   - prism-v17-1 Modal volume contains /v17_1_adapter
#
# Post-conditions:
#   - Local: /Users/admin/prism/training/models/v17_1_modal/v17_1_adapter
#   - Local: /Users/admin/prism/training/models/prism-v17-1-fused (BF16 HF)
#   - Local: /Users/admin/prism/training/models/prism-v17-1-fused-q4km.gguf
#   - Ollama: prism-coder:7b-v17-1 tag registered
#   - Eval report: /tmp/v17_1_deploy_report.md
#
# DOES NOT touch the production prism-coder:7b tag. Promotion is a separate
# decision step after eval results are reviewed.

set -uo pipefail

LOG=/tmp/v17_1_deploy.log
REPORT=/tmp/v17_1_deploy_report.md
ADAPTER_LOCAL=/Users/admin/prism/training/models/v17_1_modal/v17_1_adapter
FUSED_DIR=/Users/admin/prism/training/models/prism-v17-1-fused
GGUF_OUT=/Users/admin/prism/training/models/prism-v17-1-fused-q4km.gguf
MODELFILE=/tmp/Modelfile.v17-1
OLLAMA_TAG=prism-coder:7b-v17-1
LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
START_TS=$(date +%s)

cd "$BASE_DIR"
: > "$LOG"

# Activate the training venv so python3 has peft, transformers, gguf, etc.
source "$BASE_DIR/venv/bin/activate" 2>/dev/null || true

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  {
    echo "# v17.1 Deploy — FAILED"
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

verify() {
  # Mandatory verification per command_verification skill.
  local what="$1"
  local cmd="$2"
  log "  verify: $what"
  if ! eval "$cmd" >> "$LOG" 2>&1; then
    fail "verification failed: $what"
  fi
}

log "=== v17.1 deploy pipeline starting ==="

# ─── 1. Fetch adapter from Modal ────────────────────────────────────────────
log "[1/6] Fetching adapter from prism-v17-1 volume..."
mkdir -p "$(dirname "$ADAPTER_LOCAL")"
modal volume get prism-v17-1 v17_1_adapter "$(dirname "$ADAPTER_LOCAL")/" --force 2>&1 | tail -10 | tee -a "$LOG"
verify "adapter directory exists" "[ -d \"$ADAPTER_LOCAL\" ]"
verify "adapter_config.json exists" "[ -f \"$ADAPTER_LOCAL/adapter_config.json\" ]"
verify "adapter_model.safetensors exists" "[ -f \"$ADAPTER_LOCAL/adapter_model.safetensors\" ]"
log "  adapter -> $ADAPTER_LOCAL"

# Also fetch meta + smoke results
modal volume get prism-v17-1 v17_1_meta.json /tmp/v17_1_meta.json --force 2>&1 | tail -3 | tee -a "$LOG" || log "WARN: v17_1_meta.json not fetched"

# ─── 2. PEFT merge: v17 adapter + v17.1 delta on top of clean Qwen base ─────
log "[2/6] PEFT merge into BF16 fused model..."
rm -rf "$FUSED_DIR"
python3 v163_peft_merge.py \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "$ADAPTER_LOCAL" \
  "$FUSED_DIR" 2>&1 | tail -20 | tee -a "$LOG"
verify "fused model has config.json" "[ -f \"$FUSED_DIR/config.json\" ]"
verify "fused model has safetensors" "ls $FUSED_DIR/*.safetensors 2>/dev/null | head -1"
log "  fused -> $FUSED_DIR"

# ─── 3. Convert HF → GGUF F16 ───────────────────────────────────────────────
log "[3/6] Converting HF → GGUF F16..."
GGUF_FP16=/tmp/v17_1_fp16.gguf
PYTHONPATH="$LLAMA_CPP/gguf-py" python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -10 | tee -a "$LOG"
verify "F16 GGUF produced" "[ -f \"$GGUF_FP16\" ]"
verify "F16 GGUF non-trivial size (>5GB)" "[ \$(stat -f%z \"$GGUF_FP16\" 2>/dev/null || stat -c%s \"$GGUF_FP16\") -gt 5000000000 ]"

# ─── 4. Quantize Q4_K_M ─────────────────────────────────────────────────────
log "[4/6] Quantizing F16 → Q4_K_M..."
QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || QUANTIZE_BIN="$LLAMA_CPP/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || fail "llama-quantize not found"
"$QUANTIZE_BIN" "$GGUF_FP16" "$GGUF_OUT" Q4_K_M 2>&1 | tail -8 | tee -a "$LOG"
verify "Q4_K_M GGUF produced" "[ -f \"$GGUF_OUT\" ]"
rm -f "$GGUF_FP16"
GGUF_SIZE=$(du -h "$GGUF_OUT" | cut -f1)
log "  GGUF -> $GGUF_OUT ($GGUF_SIZE)"

# ─── 5. Register with Ollama ────────────────────────────────────────────────
log "[5/6] Registering Ollama tag $OLLAMA_TAG..."
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
PARAMETER stop "</tool_call>"
EOF
ollama create "$OLLAMA_TAG" -f "$MODELFILE" 2>&1 | tail -5 | tee -a "$LOG"
verify "Ollama tag registered" "ollama list 2>/dev/null | grep -F '$OLLAMA_TAG' > /dev/null"
log "  Ollama tag -> $OLLAMA_TAG"

# ─── 6. Build deploy report (eval is run separately) ────────────────────────
ELAPSED=$(( $(date +%s) - START_TS ))
log "[6/6] Building deploy report..."
{
  echo "# v17.1 Deploy — SUCCESS"
  echo
  echo "**Completed at**: $(date)"
  echo "**Wall time**: ${ELAPSED}s"
  echo "**Ollama tag**: $OLLAMA_TAG"
  echo "**GGUF size**: $GGUF_SIZE"
  echo "**Fused HF**: $FUSED_DIR"
  echo
  echo "## Modal training meta"
  echo '```json'
  cat /tmp/v17_1_meta.json 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(no meta)"
  echo '```'
  echo
  echo "## Next step"
  echo "Run \`bash run_full_eval_v17_1.sh\` to evaluate."
  echo
  echo "## Promotion (after eval gates pass)"
  echo "\`\`\`"
  echo "ollama cp $OLLAMA_TAG prism-coder:7b   # ⚠️  PRODUCTION CHANGE"
  echo "\`\`\`"
} > "$REPORT"
log "=== deploy complete in ${ELAPSED}s — see $REPORT ==="
