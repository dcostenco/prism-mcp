#!/bin/bash
# Local pipeline: baseline benchmark → merge contrastive → SFT → post-train benchmark.
# No Modal, no remote GPU. Runs on Mac with MLX.
set -uo pipefail
cd "$(dirname "$0")"

LOG="local_chain.log"
exec > "$LOG" 2>&1

FIX_VERSION="v13_contrastive"
BASE_MODEL="prism-v12-fused"
BASELINE_REPORT="results/baseline_${BASE_MODEL}.md"
NEW_REPORT="results/benchmark_${FIX_VERSION}.md"
TS=$(date +%Y%m%d-%H%M%S)

mkdir -p results data

echo "=== local_chain start $(date) ==="
echo "FIX=${FIX_VERSION}  BASE=${BASE_MODEL}"

# ─── STEP 1: Baseline benchmark on prism-v12-fused (no adapter) ────────────
echo ""
echo "[1/4] BASELINE benchmark on ${BASE_MODEL}..."
PRISM_MODEL_PATH="$(pwd)/models/${BASE_MODEL}" \
PRISM_ADAPTER_PATH="/tmp/no_adapter_${TS}" \
PRISM_SFT_ADAPTER="/tmp/no_adapter_${TS}" \
    python3 benchmark.py --adapter "/tmp/no_adapter_${TS}" 2>&1 | tee "${BASELINE_REPORT}.raw"
[[ -f results/benchmark_report.md ]] && cp results/benchmark_report.md "${BASELINE_REPORT}"
echo "[1/4] Baseline report → ${BASELINE_REPORT}"

# ─── STEP 2: Merge contrastive into train.jsonl ───────────────────────────
echo ""
echo "[2/4] Merging contrastive_sft.jsonl into train.jsonl..."
[[ -f data/train.jsonl ]] && cp data/train.jsonl "data/train.jsonl.bak.${TS}"
[[ -f data/valid.jsonl ]] && cp data/valid.jsonl "data/valid.jsonl.bak.${TS}"
TRAIN_BEFORE=$(wc -l < data/train.jsonl 2>/dev/null | tr -d ' ' || echo 0)
CONTRAST=$(wc -l < data/contrastive_sft.jsonl | tr -d ' ')
echo "  train.jsonl before: ${TRAIN_BEFORE} | contrastive: ${CONTRAST}"

# 90/10 split of contrastive into train/valid, then concat + shuffle
python3 - <<'PY'
import json, random
from pathlib import Path
random.seed(42)
data_dir = Path("data")
existing_train = [l for l in (data_dir/"train.jsonl").read_text().splitlines() if l.strip()]
existing_valid = [l for l in (data_dir/"valid.jsonl").read_text().splitlines() if l.strip()]
contrast = [l for l in (data_dir/"contrastive_sft.jsonl").read_text().splitlines() if l.strip()]
random.shuffle(contrast)
split = int(len(contrast) * 0.9)
new_train = existing_train + contrast[:split]
new_valid = existing_valid + contrast[split:]
random.shuffle(new_train); random.shuffle(new_valid)
(data_dir/"train.jsonl").write_text("\n".join(new_train) + "\n")
(data_dir/"valid.jsonl").write_text("\n".join(new_valid) + "\n")
print(f"  merged train: {len(new_train)}  valid: {len(new_valid)}")
PY

TRAIN_AFTER=$(wc -l < data/train.jsonl | tr -d ' ')
VALID_AFTER=$(wc -l < data/valid.jsonl | tr -d ' ')
echo "  train.jsonl after: ${TRAIN_AFTER} | valid.jsonl: ${VALID_AFTER}"

# ─── STEP 3: Surgical SFT ─────────────────────────────────────────────────
echo ""
echo "[3/4] Running surgical SFT: ${FIX_VERSION} on ${BASE_MODEL}..."
./run_surgical_sft.sh "${FIX_VERSION}" "${BASE_MODEL}"
SFT_RC=$?
if [[ $SFT_RC -ne 0 ]]; then
    echo "[3/4] SFT FAILED rc=${SFT_RC} — aborting"
    exit $SFT_RC
fi
echo "[3/4] SFT done"

# ─── STEP 4: Post-train benchmark on new fused model ──────────────────────
echo ""
echo "[4/4] Post-train benchmark on prism-${FIX_VERSION}-fused..."
PRISM_MODEL_PATH="$(pwd)/models/prism-${FIX_VERSION}-fused" \
PRISM_ADAPTER_PATH="/tmp/no_adapter_${TS}" \
PRISM_SFT_ADAPTER="/tmp/no_adapter_${TS}" \
    python3 benchmark.py --adapter "/tmp/no_adapter_${TS}" 2>&1 | tee "${NEW_REPORT}.raw"
[[ -f results/benchmark_report.md ]] && cp results/benchmark_report.md "${NEW_REPORT}"
echo "[4/4] Post-train report → ${NEW_REPORT}"

echo ""
echo "=== local_chain done $(date) ==="
echo "Baseline:  ${BASELINE_REPORT}"
echo "Post-SFT:  ${NEW_REPORT}"
echo "Compare with: diff ${BASELINE_REPORT} ${NEW_REPORT}"
