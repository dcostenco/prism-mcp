#!/usr/bin/env bash
# Auto-trigger script — polls Modal until v18-clean completes, then runs
# deploy_v18_clean_checkpoints.sh, then picks the best checkpoint vs v17.4.
#
# Run this in a separate terminal or via:
#   bash auto_pick_best_v18clean.sh > /tmp/auto_pick_v18clean.log 2>&1 &
#
# It will:
#   1. Poll modal app list every 60s for prism-v18-clean state
#   2. When state=stopped, run deploy_v18_clean_checkpoints.sh
#   3. Parse comparison report, identify best checkpoint
#   4. Compare best vs v17.4 production scores
#   5. Write final recommendation to /tmp/v18clean_pick_best.md

set -uo pipefail

BASE_DIR=/Users/admin/prism/training
APP_NAME="prism-v18-clean-sft"
LOG=/tmp/auto_pick_v18clean.log
RECOMMENDATION=/tmp/v18clean_pick_best.md
COMPARISON_REPORT=/tmp/v18_clean_checkpoint_comparison.md

cd "$BASE_DIR"
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== Auto-pick best v18-clean checkpoint — polling started ==="

# ─── Poll until training completes ────────────────────────────────────────
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  STATE=$(COLUMNS=200 modal app list 2>/dev/null | grep "$APP_NAME" | head -1 | awk -F'│' '{print $4}' | xargs)
  log "[poll $ATTEMPT] $APP_NAME state: ${STATE:-unknown}"

  if [ "$STATE" = "stopped" ]; then
    log "Training complete! Proceeding to deploy."
    break
  elif [ -z "$STATE" ]; then
    log "WARN: app not found — may have already cleaned up. Trying deploy anyway."
    break
  fi

  sleep 120
done

# ─── Run multi-checkpoint deploy + eval ────────────────────────────────────
log "=== Running deploy_v18_clean_checkpoints.sh ==="
bash deploy_v18_clean_checkpoints.sh 2>&1 | tee -a "$LOG"

# ─── Parse comparison report and pick best ────────────────────────────────
if [ ! -f "$COMPARISON_REPORT" ]; then
  log "FATAL: $COMPARISON_REPORT not produced"
  exit 1
fi

# Reference v17.4 (current production)
V174_BFCL=79.7
V174_CAREGIVER=5
V174_EMERGENCY=13
V174_TEXT_CORRECT=15
V174_ASKAI=5
V174_TRANSLATE=7

# Parse each checkpoint's BFCL score
log "=== Picking best checkpoint vs v17.4 production ==="
BEST_TAG=""
BEST_BFCL=0
declare -a CHECKPOINT_REPORTS=()

for tag in v18clean-epoch0 v18clean-epoch1 v18clean-epoch2 v18clean-final; do
  bfcl_log="/tmp/bfcl_v18clean_${tag}.log"
  if [ ! -f "$bfcl_log" ]; then
    log "  $tag: skipped (no BFCL log)"
    continue
  fi
  bfcl_score=$(grep -oE "Overall Accuracy.*: [0-9.]+%" "$bfcl_log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1)
  log "  $tag: BFCL=${bfcl_score:-N/A}%"

  # Track highest BFCL that doesn't regress AAC critical gates
  aac_log="/tmp/aac_v18clean_${tag}.log"
  if [ -f "$aac_log" ]; then
    emergency=$(grep -oE "emergency_qa.*[0-9]+/[0-9]+" "$aac_log" | head -1 | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    askai=$(grep -oE "ask_ai.*[0-9]+/[0-9]+" "$aac_log" | head -1 | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    text_correct=$(grep -oE "text_correct.*[0-9]+/[0-9]+" "$aac_log" | head -1 | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)

    # Hard gate enforcement
    if [ "${emergency:-0}" -lt 12 ]; then
      log "    SKIP: emergency $emergency/13 < 12 (HARD gate)"
      continue
    fi
    if [ "${askai:-0}" -lt 5 ]; then
      log "    SKIP: ask_ai $askai/5 < 5 (HARD gate)"
      continue
    fi
    if [ "${text_correct:-0}" -lt 13 ]; then
      log "    SKIP: text_correct $text_correct/15 < 13"
      continue
    fi
  fi

  # Float comparison via awk
  is_better=$(awk -v new="${bfcl_score:-0}" -v cur="$BEST_BFCL" 'BEGIN { print (new > cur) }')
  if [ "$is_better" = "1" ]; then
    BEST_BFCL="$bfcl_score"
    BEST_TAG="prism-coder:7b-${tag}"
  fi
done

# ─── Write recommendation ─────────────────────────────────────────────────
{
  echo "# v18-clean Best Checkpoint Recommendation"
  echo
  echo "**Generated**: $(date)"
  echo
  echo "## Production baseline (v17.4)"
  echo "- BFCL: ${V174_BFCL}%"
  echo "- caregiver: ${V174_CAREGIVER}/7 (+ targeted 20/20)"
  echo "- emergency_qa: ${V174_EMERGENCY}/13 PERFECT"
  echo "- text_correct: ${V174_TEXT_CORRECT}/15"
  echo "- ask_ai: ${V174_ASKAI}/5"
  echo "- translate: ${V174_TRANSLATE}/8"
  echo
  echo "## Best v18-clean checkpoint passing all hard gates"
  if [ -z "$BEST_TAG" ]; then
    echo
    echo "**❌ NO v18-clean checkpoint passes all hard gates.**"
    echo "Production stays on v17.4. v18-clean experiment did not produce a winner."
  else
    awk -v best_bfcl="$BEST_BFCL" -v v174_bfcl="$V174_BFCL" 'BEGIN {
      delta = best_bfcl - v174_bfcl
      printf "BFCL: %s%% (vs v17.4 %s%% — Δ %+.1f pp)\n", best_bfcl, v174_bfcl, delta
    }'
    echo "Tag: \`$BEST_TAG\`"
    echo
    awk -v new="$BEST_BFCL" -v cur="$V174_BFCL" 'BEGIN {
      if (new > cur + 1.0) {
        print "**✅ STRONG WIN — recommend promotion**"
        printf "Promote: `ollama cp %s prism-coder:7b`\n", "PLACEHOLDER"
      } else if (new > cur - 1.0) {
        print "**~ Comparable — DO NOT PROMOTE (no clear win, regression risk)**"
      } else {
        print "**❌ Worse than v17.4 — DO NOT PROMOTE**"
      }
    }'
    echo
    echo "Promotion command (review first!):"
    echo "\`\`\`bash"
    echo "# Snapshot current first"
    echo "ollama cp prism-coder:7b prism-coder:7b-prev-\$(date +%Y%m%d-%H%M)"
    echo "# Run full Modelfile re-create with SYSTEM directive (do NOT use ollama cp directly — loses SYSTEM)"
    echo "# See deploy_v17_4.sh pattern for re-creation with SYSTEM"
    echo "\`\`\`"
  fi
  echo
  echo "## Full comparison table"
  echo "See: $COMPARISON_REPORT"
} > "$RECOMMENDATION"

log "=== DONE — recommendation at $RECOMMENDATION ==="
echo
cat "$RECOMMENDATION"
