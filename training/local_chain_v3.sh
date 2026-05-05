#!/bin/bash
# v3 chain — Phase 1: high-capacity DoRA to override entrenched priors.
# Config: rank=128, all layers, all 7 projections, fine_tune_type=dora
set -uo pipefail
cd "$(dirname "$0")"

LOG="local_chain_v3.log"
exec > "$LOG" 2>&1

FIX_VERSION="v15_dora_hi"
BASE_MODEL="prism-v12-fused"
ADAPTER_PATH="models/prism-fix-${FIX_VERSION}-lora"
FUSED_PATH="models/prism-${FIX_VERSION}-fused"
NEW_REPORT="results/benchmark_${FIX_VERSION}.md"
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p results data

echo "=== local_chain_v3 (DoRA hi-rank) start $(date) ==="
echo "FIX=${FIX_VERSION}  BASE=${BASE_MODEL}"

# Reuse normalized data from v2 chain (already on disk)
TRAIN_N=$(wc -l < data/train.jsonl | tr -d ' ')
VALID_N=$(wc -l < data/valid.jsonl | tr -d ' ')
echo "Reusing normalized data — train:${TRAIN_N} valid:${VALID_N}"

# ─── STEP 1: Memory guard ─────────────────────────────────────────────────
echo "[1/3] Memory guard..."
curl -s http://localhost:11434/api/generate -d '{"model":"_","keep_alive":0}' >/dev/null 2>&1 || true
pkill -f "mlx_lm" 2>/dev/null || true
sleep 1
FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
PAGE_SIZE=16384
FREE_GB=$(echo "scale=1; ($FREE_PAGES + $INACTIVE_PAGES) * $PAGE_SIZE / 1073741824" | bc)
echo "  free: ~${FREE_GB}GB"

# ─── STEP 2: DoRA training ────────────────────────────────────────────────
echo ""
echo "[2/3] High-rank DoRA training..."
source venv/bin/activate

python3 -m mlx_lm.lora \
    --model "models/${BASE_MODEL}" \
    --train \
    --data data \
    --adapter-path "$ADAPTER_PATH" \
    -c dora_config.yaml \
    --batch-size 1 \
    --iters 600 \
    --max-seq-length 2048 \
    --learning-rate 5e-5 \
    --grad-accumulation-steps 4 \
    --grad-checkpoint \
    --steps-per-report 25 \
    --save-every 100 2>&1 | tail -50
SFT_RC=${PIPESTATUS[0]}
if [[ $SFT_RC -ne 0 ]]; then
    echo "[2/3] DoRA FAILED rc=${SFT_RC}"
    exit $SFT_RC
fi

echo ""
echo "[2/3b] Fusing DoRA adapter into base..."
python3 -m mlx_lm.fuse \
    --model "models/${BASE_MODEL}" \
    --adapter-path "$ADAPTER_PATH" \
    --save-path "$FUSED_PATH" 2>&1 | tail -5
echo "  fused: ${FUSED_PATH}"

# ─── STEP 3: Benchmark ───────────────────────────────────────────────────
echo ""
echo "[3/3] Benchmark on ${FUSED_PATH}..."
PRISM_MODEL_PATH="$(pwd)/${FUSED_PATH}" \
PRISM_ADAPTER_PATH="/tmp/no_adapter_${TS}" \
PRISM_SFT_ADAPTER="/tmp/no_adapter_${TS}" \
    python3 benchmark.py --adapter "/tmp/no_adapter_${TS}" 2>&1 | tee "${NEW_REPORT}.raw"
[[ -f results/benchmark_report.md ]] && cp results/benchmark_report.md "${NEW_REPORT}"
echo "[3/3] report → ${NEW_REPORT}"

# ─── 3-way comparison ─────────────────────────────────────────────────────
echo ""
echo "=== 3-WAY COMPARISON ==="
printf "%-20s %-12s %-12s %-12s %s\n" "Metric" "v12_base" "v13_contrast" "v14_aligned" "v15_dora"
for metric in "Tool-Call Accuracy" "Parameter Accuracy"; do
  v12=$(grep "$metric" results/baseline_prism-v12-fused.md 2>/dev/null | head -1 | sed 's/.*| //;s/ |.*//')
  v13=$(grep "$metric" results/benchmark_v13_contrastive.md 2>/dev/null | head -1 | sed 's/.*| //;s/ |.*//')
  v14=$(grep "$metric" results/benchmark_v14_aligned.md 2>/dev/null | head -1 | sed 's/.*| //;s/ |.*//')
  v15=$(grep "$metric" results/benchmark_v15_dora_hi.md 2>/dev/null | head -1 | sed 's/.*| //;s/ |.*//')
  printf "%-20s %-12s %-12s %-12s %s\n" "$metric" "$v12" "$v13" "$v14" "$v15"
done
echo ""
echo "v15 failures:"
grep "❌" "${NEW_REPORT}.raw" 2>/dev/null | head -10

echo ""
echo "=== local_chain_v3 done $(date) ==="
