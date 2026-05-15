#!/usr/bin/env python3
"""
Cost estimator + pre-flight gate for cloud training.

Predicts the dollar burn of a planned training run BEFORE launching.
If the estimate exceeds the threshold (default $15) or any of the
pinned safety checks fails, exits non-zero so a wrapper script can
block the cloud launch.

Usage:

  # Estimate a 100-iter 32B run on 1× B200
  python3 tools/cost_estimator.py \\
      --tier 32b --iters 100 --gpu B200 --gpus 1 \\
      --base "Qwen/QwQ-32B"

  # Estimate + run safety checks (blocks if any fail)
  python3 tools/cost_estimator.py \\
      --tier 32b --iters 100 --gpu B200 --gpus 1 \\
      --base "Qwen/QwQ-32B" \\
      --threshold 15 --strict

Calibrated from this session's actuals:
  - 32B QwQ on 1×B200: ~6.7 s/iter, 16 min for 100 iters → $2 in compute
  - 235B on 4×B200: never trained (OOM'd in load) — but $30 wasted
  - 14B v26-polish on Mac (MLX): free, ~5 min
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ── GPU pricing (vast.ai market rates, May 2026, ±20%) ────────────
GPU_HOURLY = {
    "B200":       7.51,   # 1×
    "B200_4x":   20.00,   # 4× (cheaper per-GPU due to bulk)
    "B200_8x":   40.00,
    "H100_80":    3.20,
    "H100_2x":    6.00,
    "A100_80":    1.60,
    "A100_8x":   10.00,
    "RTX_6000":   0.90,
    "RTX_PRO_6000": 0.50,
}

# ── Empirical s/iter, batch=1, grad_accum=8, seq=2048, 4-bit LoRA ──
# Tier → GPU → seconds per training iteration
SEC_PER_ITER = {
    "1b7":  {"B200": 0.5,  "H100_80": 0.6,  "A100_80": 1.2},
    "14b":  {"B200": 2.5,  "H100_80": 3.0,  "A100_80": 6.0},
    "32b":  {"B200": 6.7,  "H100_80": 8.0,  "A100_80": 16.0},
    "30b_moe": {"B200": 5.0,  "H100_80": 6.0},   # MoE: less active compute
    "235b": {"B200_4x": 60.0, "B200_8x": 30.0},  # requires sharding
}

# ── Fixed overhead per run (load, fuse, GGUF, download) ───────────
# Calibrated from session: 32B was ~30 min total wall, ~16 min training
OVERHEAD_MIN = {
    "1b7":     10,
    "14b":     15,
    "32b":     20,
    "30b_moe": 18,
    "235b":    30,
}


def estimate(tier: str, iters: int, gpu: str, gpus: int = 1) -> dict:
    """Return cost estimate as dict. All time in minutes, cost in USD."""
    gpu_key = gpu if gpus == 1 else f"{gpu}_{gpus}x"
    hourly = GPU_HOURLY.get(gpu_key)
    if hourly is None:
        raise ValueError(
            f"Unknown gpu config '{gpu_key}'. Known: {sorted(GPU_HOURLY)}"
        )

    spi_map = SEC_PER_ITER.get(tier, {})
    spi = spi_map.get(gpu) or spi_map.get(gpu_key)
    if spi is None:
        # Fallback: estimate from tier size if hardware combo unmeasured
        warn = (f"⚠️  No calibrated s/iter for tier={tier} gpu={gpu}; "
                f"using rough scaling")
        spi = {"1b7": 1.0, "14b": 5.0, "32b": 12.0, "30b_moe": 8.0,
               "235b": 90.0}.get(tier, 10.0)
    else:
        warn = None

    train_min = (iters * spi) / 60
    overhead_min = OVERHEAD_MIN.get(tier, 20)
    total_min = train_min + overhead_min

    cost_total = (total_min / 60) * hourly
    cost_train = (train_min / 60) * hourly
    cost_overhead = (overhead_min / 60) * hourly

    return {
        "tier": tier,
        "iters": iters,
        "gpu": gpu_key,
        "hourly_rate_usd": hourly,
        "sec_per_iter": spi,
        "minutes_train": round(train_min, 1),
        "minutes_overhead": overhead_min,
        "minutes_total": round(total_min, 1),
        "cost_train_usd": round(cost_train, 2),
        "cost_overhead_usd": round(cost_overhead, 2),
        "cost_total_usd": round(cost_total, 2),
        "warning": warn,
    }


# ── Pre-flight safety checks ──────────────────────────────────────

def check_lineage(tier: str, base: str) -> Optional[str]:
    """Match BASE_MODEL_LINEAGE from the lineage test."""
    canonical = {
        "1b7":  ["Qwen/Qwen3-1.7B"],
        "14b":  ["Qwen/Qwen3-14B"],
        "32b":  ["Qwen/QwQ-32B"],
    }
    allowed = canonical.get(tier)
    if not allowed:
        return None  # No canonical for this tier — accept
    if base not in allowed:
        return (
            f"LINEAGE MISMATCH — tier '{tier}' canonical base is "
            f"{allowed[0]!r} but you set base={base!r}. This is the "
            f"$11 mistake from this session (Qwen3-32B vs QwQ-32B). "
            f"Either fix the base, or version-bump and update "
            f"BASE_MODEL_LINEAGE in tests/training/test_base_model_lineage.py."
        )
    return None


def check_recipe(iters: int, lora_rank: int, corpus_rows: int) -> Optional[str]:
    """Flag known-bad recipes per RUNBOOK_TRAINING.md."""
    # v25-max regression signature: 40K rows + r=32 + 300 iters
    if corpus_rows > 5000 and lora_rank >= 16 and iters > 100:
        return (
            f"RECIPE SMELL — corpus={corpus_rows} rows + r={lora_rank} "
            f"+ iters={iters} matches the v25-max regression pattern that "
            f"caused 14B 100% → 81% on the BFCL gate in this session. "
            f"Recommend: shrink corpus to <1K rows of plain+tool guards, "
            f"r ≤ 16, iters ≤ 100. See docs/RUNBOOK_TRAINING.md."
        )
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", required=True,
                   choices=["1b7", "14b", "32b", "30b_moe", "235b"])
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--gpu", default="B200",
                   choices=["B200", "H100_80", "A100_80", "RTX_6000",
                            "RTX_PRO_6000"])
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--base", help="HF base model id (for lineage check)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--corpus-rows", type=int, default=576)
    p.add_argument("--threshold", type=float, default=15.0,
                   help="Block launch if estimated cost > this many USD")
    p.add_argument("--strict", action="store_true",
                   help="Exit nonzero if cost > threshold OR any check fails")
    p.add_argument("--json", action="store_true",
                   help="Output JSON only (for scripting)")
    args = p.parse_args()

    est = estimate(args.tier, args.iters, args.gpu, args.gpus)
    issues = []
    if args.base:
        if msg := check_lineage(args.tier, args.base):
            issues.append(msg)
    if msg := check_recipe(args.iters, args.lora_rank, args.corpus_rows):
        issues.append(msg)
    cost_over = est["cost_total_usd"] > args.threshold

    if args.json:
        print(json.dumps({
            **est, "issues": issues, "over_threshold": cost_over,
        }, indent=2))
    else:
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Pre-flight cost estimate")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Tier         : {args.tier}")
        print(f"  Iters        : {args.iters} @ {est['sec_per_iter']:.1f}s/iter")
        print(f"  GPU          : {est['gpu']} (@ ${est['hourly_rate_usd']:.2f}/hr)")
        print(f"  Train wall   : {est['minutes_train']:.1f} min")
        print(f"  Overhead     : {est['minutes_overhead']:.0f} min")
        print(f"  Total wall   : {est['minutes_total']:.1f} min")
        print(f"")
        print(f"  TRAIN cost   : ${est['cost_train_usd']:.2f}")
        print(f"  OVERHEAD     : ${est['cost_overhead_usd']:.2f}")
        print(f"  TOTAL cost   : ${est['cost_total_usd']:.2f}")
        if est.get("warning"):
            print(f"")
            print(f"  {est['warning']}")
        if cost_over:
            print(f"")
            print(f"  ⚠️  EXCEEDS THRESHOLD ${args.threshold:.2f} — confirm with user")
        for issue in issues:
            print(f"")
            print(f"  ❌ {issue}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.strict and (cost_over or issues):
        sys.exit(2)


if __name__ == "__main__":
    main()
