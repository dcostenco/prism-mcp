#!/bin/bash
# Deploy v16 — fuse adapter, convert to GGUF, register in Ollama with rollback safety.
#
# Usage:
#   bash deploy_v16.sh                  # full deploy (must pass gates first)
#   bash deploy_v16.sh --rollback       # restore prism-coder:7b -> v12
#   bash deploy_v16.sh --skip-validation # bypass gates (USE WITH EXTREME CAUTION)
#
# Reliability: prism-coder:7b currently points at v12 (the production model).
# This script:
#   1. Validates v16 passes the synalux + AAC + emergency gates
#   2. Creates a new ollama tag :7b-v16
#   3. Tags v12 as :7b-v12 (backup)
#   4. Repoints :7b -> v16
#   5. On any failure, repoints :7b -> v12 (rollback)
set -uo pipefail
cd "$(dirname "$0")"

ROLLBACK=0
SKIP_VALIDATION=0
for arg in "$@"; do
  case "$arg" in
    --rollback) ROLLBACK=1 ;;
    --skip-validation) SKIP_VALIDATION=1 ;;
  esac
done

# ── Rollback path ───────────────────────────────────────────────────────────
if [[ $ROLLBACK -eq 1 ]]; then
  echo "=== ROLLBACK: prism-coder:7b -> v12 ==="
  if ! ollama list | grep -q "prism-coder:7b-v12"; then
    echo "ERROR: prism-coder:7b-v12 tag not found — cannot rollback safely"
    exit 1
  fi
  ollama cp prism-coder:7b-v12 prism-coder:7b
  echo "✅ rollback complete; prism-coder:7b now points at v12"
  exit 0
fi

# ── Pre-flight: verify v16 artifacts exist ──────────────────────────────────
ADAPTER_DIR="models/v16_modal/v16_adapter"
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: $ADAPTER_DIR not found. Run 'modal run modal_v16_sft.py::fetch' first."
  exit 1
fi

# ── Step 1: Fuse adapter into a full Qwen2.5-Coder-7B model ────────────────
echo "[1/6] Fusing v16 adapter with mlx_lm..."
source venv/bin/activate
mlx_lm.fuse \
  --model "models/prism-v12-fused" \
  --adapter-path "$ADAPTER_DIR" \
  --save-path "models/prism-v16-fused" \
  --de-quantize 2>&1 | tail -20

if [[ ! -d "models/prism-v16-fused" ]]; then
  echo "ERROR: fuse failed"; exit 1
fi
echo "  ✓ fused -> models/prism-v16-fused"

# ── Step 2: Convert to GGUF for Ollama ─────────────────────────────────────
echo "[2/6] Converting to GGUF Q4_K_M via llama.cpp..."
LLAMA_CPP_DIR="${HOME}/llama.cpp"
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "ERROR: $LLAMA_CPP_DIR not found"; exit 1
fi
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  models/prism-v16-fused \
  --outfile models/prism-v16-fused.gguf \
  --outtype f16
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  models/prism-v16-fused.gguf \
  models/prism-v16-fused-q4km.gguf \
  Q4_K_M
echo "  ✓ q4_k_m gguf -> models/prism-v16-fused-q4km.gguf"

# ── Step 3: Register as ollama tag prism-coder:7b-v16 ──────────────────────
echo "[3/6] Registering ollama tag prism-coder:7b-v16..."
cat > /tmp/Modelfile.v16 <<'EOF'
FROM ./models/prism-v16-fused-q4km.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "</|tool_call|>"
PARAMETER stop "</|synalux_answer|>"

SYSTEM """You are Prism, an AI coding assistant with persistent memory across sessions and an AAC helper for users with motor and communication impairments.
Use <|synalux_think|> for reasoning before tool calls. Wrap tool calls in <|tool_call|>...</|tool_call|>. Wrap direct text answers in <|synalux_answer|>...</|synalux_answer|>.
For AAC tasks (text correction, translation, kid Q&A, caregiver notes, emergency Q&A), respond directly without tool calls."""
EOF
ollama create prism-coder:7b-v16 -f /tmp/Modelfile.v16
echo "  ✓ tag created: prism-coder:7b-v16"

# ── Step 4: Reliability gates ──────────────────────────────────────────────
if [[ $SKIP_VALIDATION -ne 1 ]]; then
  echo "[4/6] Running reliability gates against prism-coder:7b-v16..."

  # Synalux benchmark
  echo "  → synalux benchmark..."
  python3 benchmark.py --model prism-coder:7b-v16 2>&1 | tail -10
  SYN_RESULT=$(grep -oE "Tool-Call Accuracy:\s*[0-9.]+%" results/benchmark_report.md | head -1 | grep -oE "[0-9.]+")
  echo "  synalux: ${SYN_RESULT}%"

  # AAC realigned eval
  echo "  → AAC realigned eval..."
  MODEL=prism-coder:7b-v16 python3 run_aac_realigned.py 2>&1 | tail -10
  AAC_RESULT=$(grep -oE "OVERALL\s+[0-9]+/[0-9]+\s*=\s*[0-9.]+%" "results/aac_realigned_prism-coder_7b-v16.json" 2>/dev/null | head -1 | grep -oE "[0-9.]+%" | tail -1 | tr -d '%')
  AAC_RESULT=${AAC_RESULT:-0}
  echo "  AAC: ${AAC_RESULT}%"

  # Gate decision
  GATES_PASS=1
  python3 -c "
import sys
syn = float('${SYN_RESULT}' or 0)
aac = float('${AAC_RESULT}' or 0)
if syn < 95.0:
    print(f'  ❌ synalux gate FAILED: {syn}% < 95.0%'); sys.exit(1)
if aac < 89.0:
    print(f'  ❌ AAC gate FAILED: {aac}% < 89.0% (v12 baseline)'); sys.exit(1)
print(f'  ✅ all gates passed: synalux={syn}% AAC={aac}%')
" || GATES_PASS=0

  if [[ $GATES_PASS -ne 1 ]]; then
    echo ""
    echo "⚠️  v16 did not meet reliability gates."
    echo "    Tag prism-coder:7b-v16 is created but :7b NOT promoted."
    echo "    Re-run with --skip-validation to override (NOT RECOMMENDED)."
    exit 1
  fi
fi

# ── Step 5: Save current :7b as :7b-v12 backup, then promote :7b-v16 ────────
echo "[5/6] Backing up current :7b as :7b-v12..."
ollama cp prism-coder:7b prism-coder:7b-v12
echo "  ✓ backup: prism-coder:7b-v12"

echo "[6/6] Promoting v16 to production (:7b)..."
ollama cp prism-coder:7b-v16 prism-coder:7b
echo "  ✓ prism-coder:7b now points at v16"

echo ""
echo "==================================================="
echo "✅ v16 deployed. Rollback at any time with:"
echo "   bash deploy_v16.sh --rollback"
echo "==================================================="
