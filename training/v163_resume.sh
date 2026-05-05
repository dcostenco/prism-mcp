#!/usr/bin/env bash
# Resume v16.3 pipeline from step 4 (fusion). Adapter is already on disk.
set -uo pipefail

LOG=/tmp/v163_pipeline.log
REPORT=/tmp/v163_pipeline_report.md
ADAPTER_LOCAL=/Users/admin/prism/training/models/v16_3_modal/v16_3_adapter
FUSED_DIR=/Users/admin/prism/training/models/prism-v16-3-fused
GGUF_OUT=/Users/admin/prism/training/models/prism-v16-3-fused-q4km.gguf
MODELFILE=/tmp/Modelfile.v16-3
OLLAMA_TAG=prism-coder:7b-v16-3
LLAMA_CPP=/Users/admin/llama.cpp
BASE_DIR=/Users/admin/prism/training
START_TS=$(date +%s)

cd "$BASE_DIR"
log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() {
  log "FATAL: $*"
  {
    echo "# v16.3 Pipeline — FAILED at $(date)"
    echo
    echo "**Reason**: $1"
    echo
    echo "## Log tail"
    echo '```'
    tail -50 "$LOG"
    echo '```'
  } > "$REPORT"
  exit 1
}

log "=== v16.3 RESUME starting (skipping training/fetch — adapter already local) ==="
[ -d "$ADAPTER_LOCAL" ] || fail "adapter dir missing at $ADAPTER_LOCAL — full run required"

# ─── 4. Fuse via PEFT merge_and_unload (adapter is HF/PEFT format) ──────
log "[4/8] Fusing PEFT adapter into base via transformers + peft..."
source venv/bin/activate 2>/dev/null || true
rm -rf "$FUSED_DIR"
# Use the HF base model id from adapter_config.json's base_model_name_or_path
# (Qwen/Qwen2.5-Coder-7B-Instruct) — that's what the adapter was trained
# against. mlx_lm.fuse failed because the adapter is PEFT-format, not MLX.
python3 v163_peft_merge.py \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "$ADAPTER_LOCAL" \
  "$FUSED_DIR" 2>&1 | tail -25 | tee -a "$LOG"
[ -d "$FUSED_DIR" ] || fail "fusion produced no output dir"
[ -f "$FUSED_DIR/config.json" ] || fail "fusion missing config.json"
log "  fused -> $FUSED_DIR ($(du -sh "$FUSED_DIR" | cut -f1))"

# ─── 5. GGUF ─────────────────────────────────────────────────────────────
log "[5/8] Converting to GGUF Q4_K_M..."
[ -d "$LLAMA_CPP" ] || fail "llama.cpp not found at $LLAMA_CPP"
GGUF_FP16=/tmp/v163_fp16.gguf
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$FUSED_DIR" --outfile "$GGUF_FP16" --outtype f16 2>&1 | tail -15 | tee -a "$LOG"
[ -f "$GGUF_FP16" ] || fail "f16 GGUF not produced"
QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || QUANTIZE_BIN="$LLAMA_CPP/llama-quantize"
[ -x "$QUANTIZE_BIN" ] || fail "llama-quantize not found"
"$QUANTIZE_BIN" "$GGUF_FP16" "$GGUF_OUT" Q4_K_M 2>&1 | tail -10 | tee -a "$LOG"
[ -f "$GGUF_OUT" ] || fail "Q4_K_M GGUF not produced"
log "  GGUF -> $GGUF_OUT ($(du -h "$GGUF_OUT" | cut -f1))"
rm -f "$GGUF_FP16"

# ─── 6. Ollama register ─────────────────────────────────────────────────
log "[6/8] Registering Ollama tag $OLLAMA_TAG..."
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
ollama create "$OLLAMA_TAG" -f "$MODELFILE" 2>&1 | tail -10 | tee -a "$LOG"
ollama list | grep "v16-3" | tee -a "$LOG" || fail "Ollama tag not registered"

# ─── 7. AAC realigned ───────────────────────────────────────────────────
log "[7/8] Running AAC realigned eval against $OLLAMA_TAG..."
MODEL="$OLLAMA_TAG" python3 run_aac_realigned.py 2>&1 | tail -40 | tee -a "$LOG" || true
AAC_RESULT=$(ls -t results/aac_realigned_*.json 2>/dev/null | head -1)

# ─── 8. BFCL official ───────────────────────────────────────────────────
log "[8/8] Running BFCL official on Modal..."
modal run --detach run_bfcl_official.py::main 2>&1 | tail -10 | tee -a "$LOG" || true
log "  BFCL submitted detached. Polling for results..."
for i in $(seq 1 80); do
  if modal volume ls bfcl-results 2>/dev/null | grep -q "scores.txt"; then
    modal run run_bfcl_official.py::fetch 2>&1 | tail -3 | tee -a "$LOG"
    break
  fi
  sleep 60
done

# ─── Report ─────────────────────────────────────────────────────────────
END_TS=$(date +%s)
ELAPSED=$(( (END_TS - START_TS) / 60 ))
{
  echo "# v16.3 Pipeline — FINISHED"
  echo
  echo "**Started (resume)**: $(date -r "$START_TS")"
  echo "**Finished**: $(date)"
  echo "**Resume elapsed**: ${ELAPSED} min"
  echo
  echo "## Artifacts"
  echo "- Adapter: \`$ADAPTER_LOCAL\`"
  echo "- Fused model: \`$FUSED_DIR\`"
  echo "- GGUF: \`$GGUF_OUT\` ($(du -h "$GGUF_OUT" 2>/dev/null | cut -f1))"
  echo "- Ollama tag: \`$OLLAMA_TAG\`"
  echo
  echo "## In-container reliability gate (advisory only)"
  echo '```json'
  cat /tmp/v163_gate.json 2>/dev/null | python3 -m json.tool 2>/dev/null || cat /tmp/v163_gate.json 2>/dev/null
  echo '```'
  echo
  echo "## AAC realigned eval"
  if [ -n "${AAC_RESULT:-}" ] && [ -f "$AAC_RESULT" ]; then
    echo "Result file: \`$AAC_RESULT\`"
    python3 -c "
import json
d = json.load(open('$AAC_RESULT'))
s = d.get('summary', d)
for k,v in s.items():
    if isinstance(v, dict): print(f'- **{k}**: {v}')
    else: print(f'- **{k}**: {v}')
" 2>/dev/null
  fi
  echo
  echo "## BFCL official"
  if [ -f bfcl_scores.txt ]; then
    echo '```'; cat bfcl_scores.txt; echo '```'
  else
    echo "  (BFCL still running — \`modal volume ls bfcl-results\`)"
  fi
  echo
  echo "## BFCL submission"
  echo "Custom handler: \`bfcl-submission/prism_coder.py\`"
  echo "To submit: copy handler to local bfcl-eval install and run"
  echo "  \`bfcl generate --model prism-coder-7b-v16-3-FC\`"
  echo
  echo "## Log tail"
  echo '```'; tail -60 "$LOG"; echo '```'
} > "$REPORT"

log "=== resume complete; report at $REPORT ==="
