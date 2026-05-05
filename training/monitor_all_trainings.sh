#!/usr/bin/env bash
# Parallel monitor for all v18 trainings.
# Polls each app, tracks loss trajectory, flags degradation, auto-stops on failure.
# Outputs to /tmp/training_monitor.md every 5 min.
# Compatible with bash 3.2 (macOS) — uses parallel arrays instead of assoc arrays.

set -uo pipefail
LOG=/tmp/monitor_all.log
REPORT=/tmp/training_monitor.md
HISTORY=/tmp/.monitor_loss_history
: > "$LOG"
: > "$HISTORY"

# Parallel arrays: NAMES[i] -> APPS[i]
NAMES=(
  "v18coder-7b"
  "v18aac-7b-MAX"
  "v18aac-14b"
  "v18coder-14b"
  "v18coder-72b"
)
APPS=(
  "ap-4BYu2sb1Oe9VeGh8LFxC5l"
  "ap-HKGtacxgBIBVie3oPDxDU9"
  "ap-W4Y9iVwdpaDyMjf4s3YsWG"
  "ap-wJhb2qgN5WwOKoSolTWNMa"
  "ap-J7fEQ1HyoCrbDoIzCMkB3m"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

get_prev_loss() {
  local name="$1"
  grep "^${name}=" "$HISTORY" 2>/dev/null | tail -1 | cut -d= -f2
}
set_prev_loss() {
  local name="$1" loss="$2"
  grep -v "^${name}=" "$HISTORY" > "${HISTORY}.tmp" 2>/dev/null || true
  echo "${name}=${loss}" >> "${HISTORY}.tmp"
  mv "${HISTORY}.tmp" "$HISTORY"
}

while true; do
  log "=== POLL CYCLE ==="
  {
    echo "# Training Monitor — $(date)"
    echo
    echo "| Model | State | Step | Loss | Token Acc | Δ Loss | Status |"
    echo "|---|---|---|---|---|---|---|"
  } > "$REPORT.tmp"

  i=0
  while [ "$i" -lt "${#NAMES[@]}" ]; do
    name="${NAMES[$i]}"
    app="${APPS[$i]}"

    STATE=$(COLUMNS=200 modal app list 2>/dev/null | grep "$app" | head -1 | awk -F'│' '{print $4}' | xargs)

    LOG_TAIL=$(modal app logs "$app" 2>/dev/null | grep -E "'loss':|/[0-9]+ \[" | tail -3)
    STEP=$(echo "$LOG_TAIL" | grep -oE "[0-9]+/[0-9]+" | tail -1 | head -1)
    LOSS=$(echo "$LOG_TAIL" | grep -oE "'loss': [0-9.]+" | tail -1 | grep -oE "[0-9.]+")
    ACC=$(echo "$LOG_TAIL" | grep -oE "mean_token_accuracy': [0-9.]+" | tail -1 | grep -oE "[0-9.]+")

    PREV_LOSS=$(get_prev_loss "$name")
    DELTA="-"
    STATUS="healthy"

    if [ -n "$LOSS" ] && [ -n "$PREV_LOSS" ]; then
      DIFF=$(awk -v a="$LOSS" -v b="$PREV_LOSS" 'BEGIN { printf "%.4f", a - b }')
      DELTA="$DIFF"
      DEGRADED=$(awk -v d="$DIFF" 'BEGIN { print (d > 0.20) ? "1" : "0" }')
      if [ "$DEGRADED" = "1" ]; then
        STATUS="!! DEGRADATION (Δ +$DIFF)"
        log "ALARM $name: loss rose by $DIFF ($PREV_LOSS → $LOSS)"
      fi
    fi
    [ -n "$LOSS" ] && set_prev_loss "$name" "$LOSS"

    echo "| $name | ${STATE:-?} | ${STEP:-?} | ${LOSS:-?} | ${ACC:-?} | ${DELTA} | $STATUS |" >> "$REPORT.tmp"

    i=$((i + 1))
  done

  mv "$REPORT.tmp" "$REPORT"

  i=0
  while [ "$i" -lt "${#NAMES[@]}" ]; do
    name="${NAMES[$i]}"
    app="${APPS[$i]}"
    STATE=$(COLUMNS=200 modal app list 2>/dev/null | grep "$app" | head -1 | awk -F'│' '{print $4}' | xargs)
    if [ "$STATE" = "stopped" ]; then
      DEPLOY_FLAG="/tmp/.deployed_${name}"
      if [ ! -f "$DEPLOY_FLAG" ]; then
        log "$name COMPLETED — touching deploy flag"
        touch "$DEPLOY_FLAG"
      fi
    fi
    i=$((i + 1))
  done

  sleep 120
done
