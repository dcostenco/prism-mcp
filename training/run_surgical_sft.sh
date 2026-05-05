#!/bin/bash
# Surgical SFT pipeline with OOM protection
# Usage: ./run_surgical_sft.sh [fix_version] [base_model]
# Example: ./run_surgical_sft.sh v8 prism-v7-fused

set -euo pipefail
cd "$(dirname "$0")"

FIX_VERSION="${1:-v8}"
BASE_MODEL="${2:-prism-v7-fused}"
BASE_MODEL_PATH="models/${BASE_MODEL}"
ADAPTER_PATH="models/prism-fix-${FIX_VERSION}-lora"
FUSED_PATH="models/prism-${FIX_VERSION}-fused"
ITERS="${ITERS:-300}"
LR="${LR:-5e-6}"
BATCH="${BATCH:-2}"
MAX_SEQ="${MAX_SEQ:-2048}"

echo "=== Surgical SFT Pipeline ==="
echo "Fix version: ${FIX_VERSION}"
echo "Base model:  ${BASE_MODEL_PATH}"
echo "Adapter out: ${ADAPTER_PATH}"
echo "Fused out:   ${FUSED_PATH}"
echo "Iters: ${ITERS}, LR: ${LR}, Batch: ${BATCH}"
echo ""

# Step 0: Memory guard — kill any loaded Ollama models
echo "[0/4] Memory guard: unloading Ollama models..."
curl -s http://localhost:11434/api/generate -d '{"model":"_","keep_alive":0}' >/dev/null 2>&1 || true
# Force GC on any stale Python MLX processes
pkill -f "mlx_lm" 2>/dev/null || true
sleep 1

# Check available memory (need ~12GB for 7B LoRA training)
FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
PAGE_SIZE=16384
FREE_GB=$(echo "scale=1; ($FREE_PAGES + $INACTIVE_PAGES) * $PAGE_SIZE / 1073741824" | bc)
echo "  Available memory: ~${FREE_GB}GB"
MIN_GB=10
if (( $(echo "$FREE_GB < $MIN_GB" | bc -l) )); then
    echo "❌ ABORT: Only ${FREE_GB}GB free, need at least ${MIN_GB}GB for training"
    exit 1
fi
echo "  ✅ Sufficient memory"

# Step 1: Verify data exists
echo ""
echo "[1/4] Verifying training data..."
TRAIN_FILE="data/train.jsonl"
VALID_FILE="data/valid.jsonl"
if [[ ! -f "$TRAIN_FILE" ]] || [[ ! -f "$VALID_FILE" ]]; then
    echo "❌ ABORT: Missing train.jsonl or valid.jsonl"
    exit 1
fi
TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
VALID_COUNT=$(wc -l < "$VALID_FILE")
echo "  Train: ${TRAIN_COUNT} examples, Valid: ${VALID_COUNT} examples"

# Step 2: Run surgical LoRA SFT
echo ""
echo "[2/4] Running surgical LoRA SFT..."
source venv/bin/activate

python3 -m mlx_lm.lora \
    --model "$BASE_MODEL_PATH" \
    --train \
    --data data \
    --adapter-path "$ADAPTER_PATH" \
    --num-layers "${NUM_LAYERS:-16}" \
    --batch-size "$BATCH" \
    --iters "$ITERS" \
    --max-seq-length "$MAX_SEQ" \
    --learning-rate "$LR" \
    --steps-per-report 25 \
    --save-every 100 2>&1 | tail -20

echo "  ✅ SFT complete: ${ADAPTER_PATH}"

# Step 3: Fuse adapter into base model
echo ""
echo "[3/4] Fusing adapter into base model..."
python3 -m mlx_lm.fuse \
    --model "$BASE_MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --save-path "$FUSED_PATH" 2>&1 | tail -5

echo "  ✅ Fused model: ${FUSED_PATH}"

# Step 4: Quick sanity check (3 prompts)
echo ""
echo "[4/4] Quick sanity check..."
python3 -c "
from mlx_lm import load, generate
import gc, json
model, tok = load('${FUSED_PATH}')
tests = [
    'Run a health check on the memory backend.',
    'Export all memory to /tmp/export in JSON format.',
    'What is the difference between gRPC and REST?',
]
for t in tests:
    prompt = f'<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n'
    r = generate(model, tok, prompt=prompt, max_tokens=150)
    tool = 'NO_TOOL'
    if 'tool_call' in r:
        try:
            import re
            m = re.search(r'\"name\":\s*\"([^\"]+)\"', r)
            if m: tool = m.group(1)
        except: pass
    print(f'  {t[:50]:50s} → {tool}')
del model, tok
gc.collect()
"

echo ""
echo "=== Pipeline complete ==="
echo "Next: run bfcl_eval.py --model ${FUSED_PATH} --quiet --cleanup"
