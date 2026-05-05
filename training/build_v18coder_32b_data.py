"""Build train_v18coder_32b.jsonl — balanced mix for Prism Coder 32B SFT.

Composition (target ~46K rows):
  - 4,517 train_v18aac.jsonl       (caregiver + text_correct + emergency)
  - 1,191 train_aac_micro.jsonl    (translate + ask_ai synthesized)
  - 5,721 train_v18_synalux/...    (Synalux/Prism Memory asset — Phase 0 output)
  - 30,000 train_v18coder.jsonl    (BFCL backbone — subsampled from 189,710)
  - 5,000 train_v18coder.jsonl     (additional generalization slice, separate seed)

Each row has shape: {"text": "<|im_start|>system\\n...<|im_end|>..."}.
Max seq length: rows >2K tokens (~8K chars) are dropped to fit Qwen 2048 ctx.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path("/Users/admin/prism/training/data")
SYNALUX_PHASE0 = Path("/tmp/v18_synalux/train.jsonl")
OUT = DATA / "train_v18coder_32b.jsonl"

MAX_CHARS = 8000  # ~2K tokens at 4 chars/token


def load_jsonl(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if isinstance(text, str) and 50 <= len(text) <= MAX_CHARS:
                    rows.append(text)
            except json.JSONDecodeError:
                continue
    return rows


def main():
    random.seed(42)

    print("=== loading sources ===")
    aac = load_jsonl(DATA / "train_v18aac.jsonl")
    aac_micro = load_jsonl(DATA / "train_aac_micro.jsonl")
    synalux = load_jsonl(SYNALUX_PHASE0)
    coder = load_jsonl(DATA / "train_v18coder.jsonl")

    print(f"  v18aac:        {len(aac):>7,}")
    print(f"  aac_micro:     {len(aac_micro):>7,}")
    print(f"  synalux phase0:{len(synalux):>7,}")
    print(f"  v18coder pool: {len(coder):>7,}")

    # Subsample BFCL backbone (30K) + generalization slice (5K), separate seeds
    coder_pool = list(coder)
    random.Random(42).shuffle(coder_pool)
    coder_main = coder_pool[:30_000]
    coder_gen = coder_pool[30_000:35_000]

    out_rows: list[str] = []
    out_rows.extend(aac)
    out_rows.extend(aac_micro)
    out_rows.extend(synalux)
    out_rows.extend(coder_main)
    out_rows.extend(coder_gen)

    # Final shuffle (mix curriculum)
    random.Random(7).shuffle(out_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for text in out_rows:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    # Stats
    n_total = len(out_rows)
    avg_chars = sum(len(t) for t in out_rows) // max(1, n_total)
    by_size = {"<1K": 0, "1K-4K": 0, "4K-8K": 0}
    for t in out_rows:
        if len(t) < 1000:
            by_size["<1K"] += 1
        elif len(t) < 4000:
            by_size["1K-4K"] += 1
        else:
            by_size["4K-8K"] += 1

    print(f"\n=== output ===")
    print(f"  rows:        {n_total:,}")
    print(f"  avg chars:   {avg_chars:,}")
    print(f"  size dist:   {by_size}")
    print(f"  approx tokens (1 tok=4ch): {sum(len(t) for t in out_rows) // 4:,}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
