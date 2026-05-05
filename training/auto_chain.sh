#!/bin/bash
# Orchestrate: wait for contrastive gen + Modal BFCL eval, fetch scores,
# then launch surgical SFT on the new contrastive data.
set -uo pipefail
cd "$(dirname "$0")"

LOG="auto_chain.log"
exec >"$LOG" 2>&1

GEN_PID=$(cat contrastive_gen.pid 2>/dev/null || echo "")
APP_ID="ap-lCAcSFcjYTTRB0MSeHM53F"
BASE_MODEL="prism-v12-fused"
FIX_VERSION="v13_contrastive"

echo "=== auto_chain start $(date) ==="
echo "GEN_PID=$GEN_PID  APP_ID=$APP_ID  BASE=$BASE_MODEL  FIX=$FIX_VERSION"

# 1) Wait for contrastive generation to finish
if [[ -n "$GEN_PID" ]]; then
    echo "[1] Waiting on contrastive gen PID $GEN_PID..."
    while kill -0 "$GEN_PID" 2>/dev/null; do
        sleep 30
        N=$(wc -l < data/contrastive_sft.jsonl 2>/dev/null | tr -d ' ' || echo 0)
        echo "  [gen] $N examples written, still running..."
    done
fi
N=$(wc -l < data/contrastive_sft.jsonl 2>/dev/null | tr -d ' ' || echo 0)
echo "[1] Generation finished: $N examples in data/contrastive_sft.jsonl"

# 2) Wait for Modal eval to finish
echo "[2] Polling Modal app $APP_ID..."
while modal app list 2>/dev/null | grep -q "$APP_ID"; do
    sleep 60
    echo "  [modal] still alive..."
done
echo "[2] Modal app no longer in live list."

# 3) Fetch BFCL results to local
echo "[3] Fetching BFCL results from Volume..."
modal run run_bfcl_official.py::fetch || echo "WARN: fetch failed"
if [[ -f bfcl_scores.txt ]]; then
    echo "=== BFCL SCORES ==="
    cat bfcl_scores.txt
    echo "==================="
else
    echo "WARN: bfcl_scores.txt not present"
fi

# 4) Launch surgical SFT with new contrastive data merged in
if [[ "$N" -lt 100 ]]; then
    echo "[4] ABORT: only $N contrastive examples — too few. Skipping retrain."
    exit 1
fi

echo "[4] Launching surgical SFT: ${FIX_VERSION} on ${BASE_MODEL}"
./run_surgical_sft.sh "$FIX_VERSION" "$BASE_MODEL"
RC=$?
echo "=== auto_chain done $(date) rc=$RC ==="
exit $RC
