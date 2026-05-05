#!/usr/bin/env python3
"""Normalize training data to inference format and merge contrastive (oversampled).

Why: Training data was using <think>/<|tool_call_end|> but inference uses
<|synalux_think|>/</|tool_call|> because the system prompt forces it. The
model learned one format and was evaluated on another. This script aligns
training to inference distribution.

Steps:
  1. Read backup train.jsonl + valid.jsonl (pre-contrastive originals)
  2. Read contrastive_sft.jsonl
  3. Normalize each example: wrap with system prompt, fix tokens
  4. Oversample contrastive 4x so it dominates the gradient signal
  5. Write new train.jsonl / valid.jsonl with 90/10 split
"""
import json, random, sys, glob
from pathlib import Path
from config import format_system_prompt

random.seed(42)

DATA = Path(__file__).parent / "data"
TOOL_SCHEMA_PATH = DATA / "tool_schema.json"
CONTRASTIVE_OVERSAMPLE = 4

# Use a TRIMMED system prompt for training (format rules + tool NAMES only).
# Full XML schemas at inference are 27k chars — too long for training seqs.
# The base v12 model already knows tool semantics; we just need the format
# wrapper so the model learns canonical tokens (<|synalux_think|>, etc.) in
# the same context structure it sees at inference.
with open(TOOL_SCHEMA_PATH) as f:
    tools = json.load(f).get("tools", [])
tool_names = [t["name"] for t in tools]
_full = format_system_prompt(tools=None)  # rules only, no tool block
SYSTEM = _full + "\n\n# Available tools\n" + ", ".join(tool_names)
print(f"  trimmed system prompt: {len(SYSTEM)} chars (was {len(format_system_prompt(tools=tools))} with full schemas)")

TOKEN_REPLACEMENTS = [
    ("<think>", "<|synalux_think|>"),
    ("</think>", "</|synalux_think|>"),
    ("<|tool_call_end|>", "</|tool_call|>"),
]


def normalize_example(text: str) -> str:
    """Apply token canonicalization and prepend system block."""
    for old, new in TOKEN_REPLACEMENTS:
        text = text.replace(old, new)
    if text.startswith("<|im_start|>user"):
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n{text}"
    return text  # already has a system block, leave alone


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def find_latest_backup() -> tuple[Path, Path]:
    train_baks = sorted(glob.glob(str(DATA / "train.jsonl.bak.*")))
    valid_baks = sorted(glob.glob(str(DATA / "valid.jsonl.bak.*")))
    if not train_baks:
        sys.exit("ERROR: no train.jsonl.bak.* found — run local_chain.sh first to create one")
    return Path(train_baks[-1]), Path(valid_baks[-1])


def main():
    train_bak, valid_bak = find_latest_backup()
    print(f"Loading backup: {train_bak.name}, {valid_bak.name}")

    base_train = load_jsonl(train_bak)
    base_valid = load_jsonl(valid_bak)
    contrast = load_jsonl(DATA / "contrastive_sft.jsonl")
    print(f"  base train: {len(base_train)}  base valid: {len(base_valid)}  contrastive: {len(contrast)}")

    # Normalize all
    for pool in (base_train, base_valid, contrast):
        for ex in pool:
            ex["text"] = normalize_example(ex["text"])

    # Drop outliers > 2048 tokens (rough char-based estimate: 1 tok ~ 3.5 chars)
    MAX_CHARS = 2048 * 3
    def _filter(pool, name):
        kept = [e for e in pool if len(e["text"]) <= MAX_CHARS]
        dropped = len(pool) - len(kept)
        if dropped:
            print(f"  filtered {name}: dropped {dropped} long examples (>{MAX_CHARS} chars)")
        return kept
    base_train = _filter(base_train, "base_train")
    base_valid = _filter(base_valid, "base_valid")
    contrast = _filter(contrast, "contrast")

    # Oversample contrastive
    contrast_oversampled = contrast * CONTRASTIVE_OVERSAMPLE
    random.shuffle(contrast_oversampled)
    print(f"  contrastive oversampled {CONTRASTIVE_OVERSAMPLE}x: {len(contrast_oversampled)}")

    # 90/10 split of oversampled contrastive
    split = int(len(contrast_oversampled) * 0.9)
    new_train = base_train + contrast_oversampled[:split]
    new_valid = base_valid + contrast_oversampled[split:]
    random.shuffle(new_train)
    random.shuffle(new_valid)

    (DATA / "train.jsonl").write_text("\n".join(json.dumps(e) for e in new_train) + "\n")
    (DATA / "valid.jsonl").write_text("\n".join(json.dumps(e) for e in new_valid) + "\n")
    print(f"  wrote train: {len(new_train)}  valid: {len(new_valid)}")

    # Sanity check first example has system block + canonical tokens
    sample = new_train[0]["text"]
    assert sample.startswith("<|im_start|>system"), "missing system block"
    assert "<think>" not in sample, "stale <think> token leaked"
    assert "<|tool_call_end|>" not in sample, "stale tool_call_end token leaked"
    print("  ✅ format check passed")


if __name__ == "__main__":
    main()
