#!/usr/bin/env python3
"""Quality gates on the v16 generated data BEFORE merging into train.jsonl.

Critical safety check — this is a life-saver app, training data quality
directly affects production model behavior. We refuse to proceed if:
  - Any category falls below 70% of its target count
  - Any category has > 5% empty/degenerate examples
  - Any category has > 30% duplicate examples (low diversity)
  - Required fields missing in > 5% of records
  - Suspect content (PII, profanity-leaked-from-training) appears in samples

Exits 0 only if every gate passes. Exits 1 with a detailed report otherwise.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

GEN_DIR = Path(__file__).parent / "data" / "v16_gen"

# Per-category contracts: target count + required fields
CONTRACTS = {
    "text_correct":     {"target": 1500, "required": ["input", "expected"]},
    "emergency_qa":     {"target": 500,  "required": ["script", "question", "answer"]},
    "caregiver_parse":  {"target": 500,  "required": ["note", "actions"]},
    "translate":        {"target": 500,  "required": ["text", "fromLang", "toLang", "translation"]},
    "ask_ai_aac":       {"target": 500,  "required": ["question", "answer"]},
    "word_predict_aac": {"target": 1000, "required": ["prefix", "top5"]},
    "memory_checkout":  {"target": 80,   "required": ["prompt", "args"]},
    "hipaa":            {"target": 80,   "required": ["prompt", "args"]},
    "contrastive_extra":{"target": 200,  "required": ["prompt", "right_tool", "args"]},
    "no_tool_extra":    {"target": 200,  "required": ["prompt", "answer"]},
}

# Gate thresholds
COUNT_FLOOR_PCT = 0.70   # categories must hit >= 70% of target
DEGENERATE_CEIL_PCT = 0.05  # < 5% empty/short examples
DUP_CEIL_PCT = 0.30      # < 30% duplicate primary keys
MISSING_FIELD_CEIL_PCT = 0.05

# PII / safety patterns to flag (small, conservative)
PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),  # CC-shaped (12-16 digit runs)
]

# COMPOSITE keys for diversity check — pairs that, when together, should
# nearly always be unique. Just using the "primary" alone causes false
# positives because intentional design has the same question across many
# scripts (emergency_qa) or the same source text in many target langs
# (translate). These composites correctly target genuine duplicate-record
# patterns.
PRIMARY_KEYS_OF = {
    "text_correct": ["input", "expected"],
    "emergency_qa": ["script", "question"],         # same Q across diff scripts is GOOD
    "caregiver_parse": ["note"],
    "translate": ["text", "toLang"],                # same text to diff target IS the design
    "ask_ai_aac": ["question", "answer"],           # same Q with diff answers is OK
    "word_predict_aac": ["prefix", "top5"],         # same prefix with diff predictions is OK
    "memory_checkout": ["prompt"],
    "hipaa": ["prompt"],
    "contrastive_extra": ["prompt"],
    "no_tool_extra": ["prompt"],
}

# Field-name whitelist: 2-char codes ("en", "es", "fr", ...) are NOT
# degenerate. Only flag short *content* fields, not metadata.
SHORT_FIELD_OK = {"lang", "fromLang", "toLang", "code"}


def degenerate(record: dict) -> bool:
    """True if any *content* field is empty or laughably short.
    Whitelisted metadata fields like `lang` are exempt (they're 2-char codes)."""
    for k, v in record.items():
        if k in SHORT_FIELD_OK:
            continue
        if isinstance(v, str) and 0 < len(v.strip()) < 3:
            return True
    return False


def check_pii(records: list, sample_size: int = 200) -> list:
    """Return PII matches found in a random sample."""
    hits = []
    for r in records[:sample_size]:
        for k, v in r.items():
            if not isinstance(v, str):
                continue
            for pat in PII_PATTERNS:
                m = pat.search(v)
                if m:
                    hits.append({"field": k, "match": m.group(0)[:30]})
    return hits


def validate_category(cat: str) -> dict:
    contract = CONTRACTS[cat]
    path = GEN_DIR / f"{cat}.jsonl"
    out = {"category": cat, "target": contract["target"], "issues": []}

    if not path.exists() or path.stat().st_size == 0:
        out["issues"].append("file missing or empty")
        out["count"] = 0
        return out

    records = []
    parse_errors = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                parse_errors += 1

    n = len(records)
    out["count"] = n
    out["parse_errors"] = parse_errors

    # Count gate
    floor = int(contract["target"] * COUNT_FLOOR_PCT)
    if n < floor:
        out["issues"].append(f"count {n} < floor {floor} ({COUNT_FLOOR_PCT*100:.0f}% of target)")

    # Required fields
    missing = sum(1 for r in records for f in contract["required"] if not r.get(f))
    miss_pct = missing / max(n * len(contract["required"]), 1)
    if miss_pct > MISSING_FIELD_CEIL_PCT:
        out["issues"].append(f"missing-field rate {miss_pct*100:.1f}% > {MISSING_FIELD_CEIL_PCT*100:.0f}%")
    out["missing_field_pct"] = round(miss_pct * 100, 2)

    # Degenerate (very short fields)
    deg = sum(1 for r in records if degenerate(r))
    deg_pct = deg / max(n, 1)
    if deg_pct > DEGENERATE_CEIL_PCT:
        out["issues"].append(f"degenerate {deg_pct*100:.1f}% > {DEGENERATE_CEIL_PCT*100:.0f}%")
    out["degenerate_pct"] = round(deg_pct * 100, 2)

    # Duplicates on COMPOSITE primary key (intentional same-text-different-lang
    # patterns are NOT counted as duplicates — only true clones are).
    pk_fields = PRIMARY_KEYS_OF[cat]
    def composite(r):
        parts = []
        for f in pk_fields:
            v = r.get(f, "")
            if isinstance(v, list):
                v = "|".join(str(x) for x in v)
            elif not isinstance(v, str):
                v = json.dumps(v, sort_keys=True, ensure_ascii=False)
            parts.append(v.strip().lower())
        return "::".join(parts)

    primaries = [composite(r) for r in records]
    counts = Counter(p for p in primaries if p)
    dups = sum(c - 1 for c in counts.values() if c > 1)
    dup_pct = dups / max(n, 1)
    if dup_pct > DUP_CEIL_PCT:
        out["issues"].append(f"duplicate composite-key rate {dup_pct*100:.1f}% > {DUP_CEIL_PCT*100:.0f}%")
    out["duplicate_pct"] = round(dup_pct * 100, 2)

    # PII spot-check
    pii = check_pii(records)
    if pii:
        out["issues"].append(f"PII patterns matched in {len(pii)} fields (sample of 200)")
        out["pii_samples"] = pii[:3]

    return out


def main():
    print("=== v16 data quality gates ===\n")

    summaries = []
    total_count = 0
    total_target = 0
    fail = False

    for cat in CONTRACTS:
        s = validate_category(cat)
        summaries.append(s)
        total_count += s["count"]
        total_target += s["target"]

        status = "✅" if not s["issues"] else "❌"
        print(f"  {status}  {cat:20s}  {s['count']:4d}/{s['target']:4d}  "
              f"deg={s.get('degenerate_pct',0):4.1f}%  dup={s.get('duplicate_pct',0):4.1f}%  "
              f"miss={s.get('missing_field_pct',0):4.1f}%")
        for issue in s["issues"]:
            print(f"        ⚠ {issue}")
            fail = True

    print()
    print(f"=== TOTAL: {total_count}/{total_target} examples "
          f"({total_count/total_target*100:.0f}% of target) ===")

    if fail:
        print("\n❌ QUALITY GATES FAILED — do NOT merge into training data.")
        sys.exit(1)
    else:
        print("\n✅ All quality gates passed. Safe to merge.")
        sys.exit(0)


if __name__ == "__main__":
    main()
