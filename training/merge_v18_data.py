#!/usr/bin/env python3
"""v18 training data merge.

Composes the v18 SFT corpus from three sources, deduplicating by SHA-256 of the
rendered chatml text. Output: data/train_v18.jsonl.

Sources:
  1. data/train_v17.jsonl              — full v17 corpus (~13.7K rows): bfcl_local
                                         + v16 deltas (tone_switch, caregiver_parse_extra,
                                         bfcl_irrelevance) + AAC text_correct + emergency
  2. data/v18_external_train.jsonl     — Apache-2.0 external (~20.7K rows):
                                         bfcl_local (overlaps with #1) + glaive 6K
                                         + hermes 4K. The dedup pass strips the
                                         overlap so we don't duplicate bfcl rows.
  3. data/v16_gen_72b/caregiver_parse_extra.jsonl — boost (~1.5K rows × 2x weight)
     to address the v17 caregiver regression (57.1% on held-out vs 85% target).

Why this composition:
  - v17 had 5 gates (BFCL, text_correct, emergency, caregiver, tone). Caregiver
    was the one that regressed. We can't lose AAC-specific behavior, so we keep
    all of train_v17 verbatim.
  - External glaive + hermes adds ~10K function-calling examples that broaden
    BFCL generalization — addresses the disambiguation/multi_turn weakness
    flagged in handoff TODOs.
  - The caregiver boost duplicates caregiver_parse_extra so it gets ~3000
    effective forwards (≈ 6% of the corpus) instead of ~1.5K (≈ 3%). Mirrors
    the v17 strategy that boosted irrelevance to fix the v16 BFCL gap.

Run:
  cd /Users/admin/prism/training
  source venv/bin/activate
  python3 merge_v18_data.py

Output:
  data/train_v18.jsonl   — shuffled, deduped, ready to upload to Modal.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Re-use v16 renderers so caregiver boost rows produce identical chatml as v17.
sys.path.insert(0, str(Path(__file__).parent))
from merge_v16_data import render_caregiver_parse  # type: ignore  # noqa: E402

ROOT = Path(__file__).parent
TRAIN_V17 = ROOT / "data" / "train_v17.jsonl"
EXTERNAL = ROOT / "data" / "v18_external_train.jsonl"
CAREGIVER_BOOST = ROOT / "data" / "v16_gen_72b" / "caregiver_parse_extra.jsonl"
PLATFORM_DIR = ROOT / "data" / "v18_platform"
PLATFORM_CATEGORIES = [
    "video_script_gen",
    "tts_ssml_control",
    "voice_persona_pick",
    "word_predict_aac",
    "multimodal_tool_route",
]
DEFAULT_OUT = ROOT / "data" / "train_v18.jsonl"
CAREGIVER_BOOST_FACTOR = 2  # duplicate caregiver_extra rows N times


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chatml_jsonl(path: Path, label: str) -> list[dict]:
    """Load already-rendered chatml rows ({"text": ...})."""
    if not path.exists():
        print(f"  WARN: {label} missing at {path} — skipping", flush=True)
        return []
    rows = []
    bad = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            text = obj.get("text")
            if not isinstance(text, str) or len(text) < 80:
                bad += 1
                continue
            rows.append({"text": text})
    print(f"  {label:40s} loaded={len(rows):6d}  rejected={bad}")
    return rows


def load_caregiver_boost(path: Path, factor: int) -> list[dict]:
    """Render caregiver_parse_extra rows via v16's renderer, duplicate by factor."""
    if not path.exists():
        print(f"  WARN: caregiver boost missing at {path} — skipping")
        return []
    rendered = []
    bad = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            text = render_caregiver_parse(obj)
            if not text or len(text) < 80:
                bad += 1
                continue
            rendered.append({"text": text})
    rows = rendered * factor
    print(
        f"  caregiver_parse_extra (boost ×{factor})  unique={len(rendered):6d}"
        f"  emitted={len(rows):6d}  rejected={bad}"
    )
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    """Keep first occurrence of each unique text."""
    seen = set()
    kept = []
    for r in rows:
        h = sha(r["text"])
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caregiver-boost",
        type=int,
        default=CAREGIVER_BOOST_FACTOR,
        help="Duplicate caregiver_parse_extra rows N times (boost weight). 0 disables.",
    )
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== v18 merge starting ===\n")
    print("[A] v17 corpus")
    v17_rows = load_chatml_jsonl(TRAIN_V17, "train_v17.jsonl")
    print("\n[B] external Apache-2.0 (glaive + hermes + bfcl)")
    ext_rows = load_chatml_jsonl(EXTERNAL, "v18_external_train.jsonl")
    print("\n[C] caregiver boost")
    boost_rows = (
        load_caregiver_boost(CAREGIVER_BOOST, args.caregiver_boost)
        if args.caregiver_boost > 0
        else []
    )

    print("\n[D] platform categories (video / TTS / persona / word_predict / multimodal)")
    platform_rows: list[dict] = []
    for cat in PLATFORM_CATEGORIES:
        path = PLATFORM_DIR / f"{cat}.jsonl"
        rows = load_chatml_jsonl(path, f"v18_platform/{cat}.jsonl")
        platform_rows.extend(rows)
    print(f"  platform total: {len(platform_rows):6d} rows")

    # Order matters for first-keep dedup — v17 wins over external on collisions
    # so AAC-specific renderings of bfcl_local survive any tiny formatting drift
    # from the external generator. Platform rows come last so they're kept in
    # full (no overlap with v17 or external).
    print("\n[E] dedup by SHA-256(text), v17 wins ties")
    combined = v17_rows + ext_rows + boost_rows + platform_rows
    deduped = dedup(combined)
    n_dropped = len(combined) - len(deduped)
    print(f"  combined={len(combined):6d}  deduped={len(deduped):6d}  dropped={n_dropped}")

    if not args.no_shuffle:
        random.shuffle(deduped)
        print(f"  shuffled (seed={args.seed})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {len(deduped)} rows ({size_mb:.1f} MB) -> {out_path}")
    print("\nNext steps:")
    print(f"  modal volume put prism-sft-data {out_path} /train_v18.jsonl --force")
    print("  modal run --detach modal_v18_sft.py::train")


if __name__ == "__main__":
    main()
