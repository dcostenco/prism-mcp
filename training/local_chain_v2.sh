#!/bin/bash
# v2 chain — fix train/inference distribution mismatch.
# 1. Normalize training data: system prompt + canonical tokens
# 2. Oversample contrastive 4x
# 3. SFT with bumped hyperparams (LR=1e-5, ITERS=800)
# 4. Benchmark
set -uo pipefail
cd "$(dirname "$0")"

LOG="local_chain_v2.log"
exec > "$LOG" 2>&1

FIX_VERSION="v14_aligned"
BASE_MODEL="prism-v12-fused"
NEW_REPORT="results/benchmark_${FIX_VERSION}.md"
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p results data

# Stronger hyperparams. MAX_SEQ=2048 with outlier filtering,
# 8 LoRA layers to halve memory footprint.
export ITERS=800
export LR=1e-5
export BATCH=1
export MAX_SEQ=2048
export NUM_LAYERS=8

echo "=== local_chain_v2 start $(date) ==="
echo "FIX=${FIX_VERSION}  BASE=${BASE_MODEL}"
echo "Hyperparams: ITERS=${ITERS} LR=${LR} BATCH=${BATCH}"

# ─── STEP 1: Normalize + merge ────────────────────────────────────────────
echo ""
echo "[1/3] Normalizing training data and merging oversampled contrastive..."
# Backup current train/valid (they were modified by v1 chain)
[[ -f data/train.jsonl ]] && cp data/train.jsonl "data/train.jsonl.bak.${TS}"
[[ -f data/valid.jsonl ]] && cp data/valid.jsonl "data/valid.jsonl.bak.${TS}"

# Restore from oldest backup as the true pre-contrastive baseline
OLDEST_TRAIN_BAK=$(ls data/train.jsonl.bak.* 2>/dev/null | sort | head -1)
OLDEST_VALID_BAK=$(ls data/valid.jsonl.bak.* 2>/dev/null | sort | head -1)
if [[ -z "$OLDEST_TRAIN_BAK" ]]; then
    echo "ABORT: no train.jsonl.bak.* found"
    exit 1
fi
echo "  Restoring base from ${OLDEST_TRAIN_BAK}"
cp "$OLDEST_TRAIN_BAK" data/train.jsonl
cp "$OLDEST_VALID_BAK" data/valid.jsonl

source venv/bin/activate
python3 normalize_and_merge.py
NORM_RC=$?
if [[ $NORM_RC -ne 0 ]]; then
    echo "[1/3] normalize failed rc=${NORM_RC}"
    exit $NORM_RC
fi

# ─── STEP 2: Surgical SFT with bumped hyperparams ─────────────────────────
echo ""
echo "[2/3] Running surgical SFT: ${FIX_VERSION} on ${BASE_MODEL}..."
./run_surgical_sft.sh "${FIX_VERSION}" "${BASE_MODEL}"
SFT_RC=$?
if [[ $SFT_RC -ne 0 ]]; then
    echo "[2/3] SFT FAILED rc=${SFT_RC}"
    exit $SFT_RC
fi
echo "[2/3] SFT done"

# ─── STEP 3: Post-train benchmark ─────────────────────────────────────────
echo ""
echo "[3/3] Post-train benchmark on prism-${FIX_VERSION}-fused..."
PRISM_MODEL_PATH="$(pwd)/models/prism-${FIX_VERSION}-fused" \
PRISM_ADAPTER_PATH="/tmp/no_adapter_${TS}" \
PRISM_SFT_ADAPTER="/tmp/no_adapter_${TS}" \
    python3 benchmark.py --adapter "/tmp/no_adapter_${TS}" 2>&1 | tee "${NEW_REPORT}.raw"
[[ -f results/benchmark_report.md ]] && cp results/benchmark_report.md "${NEW_REPORT}"
echo "[3/3] Post-train report → ${NEW_REPORT}"

echo ""
echo "=== local_chain_v2 done $(date) ==="
echo "Reports:"
echo "  baseline_prism-v12-fused.md  (v12 raw)"
echo "  benchmark_v13_contrastive.md (v1 attempt — same as baseline)"
echo "  ${NEW_REPORT} (v2)"
