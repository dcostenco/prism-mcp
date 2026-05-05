#!/usr/bin/env python3
"""v18 external-data preparation.

Pulls free, Apache-2.0-licensed BFCL-style training data and renders it to
the chatml `{"text": "<|im_start|>...<|im_end|>"}` format the SFT trainer
expects. Output: `data/v18_external_train.jsonl`.

Sources (Apache-2.0 only — skips xlam/APIGen which are CC-BY-NC):
  1. ./data/bfcl/*.jsonl                  ~13.7K rows  (already on disk)
  2. glaiveai/glaive-function-calling-v2   sample 6K of 113K  (HF)
  3. NousResearch/hermes-function-calling-v1  sample 4K of ~10K  (HF)

Format normalization:
  - bfcl/*.jsonl have `{messages:[{role, content}]}` — render directly
  - glaive has `{system, chat}` where chat is a multi-turn string — split + render
  - hermes has `{conversations:[{from, value}]}` ShareGPT format — convert

License gate: every row passes through a license check based on the source
file path. Non-Apache-2.0 sources are refused at parse time.

Run:
  cd /Users/admin/prism/training
  source venv/bin/activate
  python3 prepare_v18_external.py
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BFCL_DIR = ROOT / "data" / "bfcl"
OUT = ROOT / "data" / "v18_external_train.jsonl"

# License allowlist — refuse anything else even if a config slips through.
APACHE_OK = {
    "bfcl_local",                              # synthesized internally
    "glaiveai/glaive-function-calling-v2",     # Apache-2.0
    "NousResearch/hermes-function-calling-v1", # Apache-2.0
}


def chatml_render(system: str, user: str, assistant: str) -> str:
    """Render a 3-turn dialog as chatml text."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def chatml_render_messages(msgs: list) -> str | None:
    """Render an OpenAI-style `messages` list directly to chatml. Filters out
    rows that don't have at least one user + one assistant turn."""
    parts = []
    has_user = has_assistant = False
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if not (role and isinstance(content, str)):
            continue
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            has_user = True
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            has_assistant = True
        # ignore tool/observation roles for SFT (we only train assistant outputs)
    if not (has_user and has_assistant):
        return None
    return "\n".join(parts)


# ── Source 1: existing BFCL data on disk ─────────────────────────────────────

def load_bfcl_local():
    """Read every bfcl/*.jsonl, render to chatml text format."""
    if not BFCL_DIR.exists():
        return []
    rendered = []
    for path in sorted(BFCL_DIR.glob("*.jsonl")):
        # skip non-SFT files (DPO pairs, validation set)
        if path.name in ("grpo_pairs.jsonl", "valid.jsonl"):
            continue
        n_in = n_out = 0
        with path.open() as f:
            for line in f:
                n_in += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    continue
                text = chatml_render_messages(msgs)
                if text and len(text) >= 80:
                    rendered.append({"text": text, "_src": f"bfcl_local/{path.name}"})
                    n_out += 1
        print(f"  bfcl_local/{path.name:40s}  in={n_in:5d}  rendered={n_out:5d}", flush=True)
    return rendered


# ── Source 2: glaive function calling v2 (Apache-2.0) ────────────────────────

def load_glaive(target_n: int = 6000):
    """Glaive rows look like:
        {"system": "...", "chat": "USER: ...\nASSISTANT: <functioncall>{}"}
    The chat field is a single string with USER:/ASSISTANT: turn markers.
    """
    print(f"\n[glaive] loading glaiveai/glaive-function-calling-v2 (target {target_n} rows)...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  ERROR: datasets package not installed — skipping glaive")
        return []
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading glaive: {e}")
        return []

    rendered = []
    seen = 0
    user_re = re.compile(r"\bUSER:\s*(.+?)(?=\n\s*ASSISTANT:|\nUSER:|\Z)", re.DOTALL)
    assistant_re = re.compile(r"\bASSISTANT:\s*(.+?)(?=\n\s*USER:|\n\s*ASSISTANT:|\Z)", re.DOTALL)

    for row in ds:
        seen += 1
        if len(rendered) >= target_n:
            break
        sys_p = (row.get("system") or "").strip()
        chat = (row.get("chat") or "").strip()
        if not chat:
            continue
        # Take the FIRST user → assistant exchange. Multi-turn chains are
        # noisy; first-turn rows are cleaner and more BFCL-like.
        u = user_re.search(chat)
        a = assistant_re.search(chat)
        if not (u and a):
            continue
        user = u.group(1).strip()
        asst = a.group(1).strip()
        if len(user) < 10 or len(asst) < 10:
            continue
        # Glaive uses <functioncall> {json}; rewrite to BFCL <tool_call> tag
        # so the model trains on the canonical format the eval harness reads.
        asst = re.sub(
            r"<functioncall>\s*(\{.*?\})\s*<\|endoftext\|>",
            lambda m: f"<tool_call>\n{m.group(1).strip()}\n</tool_call>",
            asst, flags=re.DOTALL,
        ).replace("<|endoftext|>", "").strip()

        text = chatml_render(sys_p, user, asst)
        if len(text) >= 100:
            rendered.append({"text": text, "_src": "glaiveai/glaive-function-calling-v2"})
        if seen % 5000 == 0:
            print(f"  ...scanned {seen}, kept {len(rendered)}", flush=True)
    print(f"  glaive scanned={seen} kept={len(rendered)}")
    return rendered


# ── Source 3: hermes function calling v1 (Apache-2.0) ────────────────────────

def load_hermes(target_n: int = 4000):
    """Hermes rows are ShareGPT-flavored:
        {"conversations": [{"from": "system", "value": ...}, {"from": "human"|"gpt", "value": ...}]}
    """
    print(f"\n[hermes] loading NousResearch/hermes-function-calling-v1 (target {target_n} rows)...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  ERROR: datasets package not installed — skipping hermes")
        return []
    try:
        # The repo has multiple subset configs; the function-calling one is "func_calling_singleturn"
        # but the default config also works for the parent train split.
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading hermes (default split): {e}")
        try:
            ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train", streaming=True)
        except Exception as e2:
            print(f"  ERROR retry: {e2}")
            return []

    rendered = []
    seen = 0
    role_map = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

    for row in ds:
        seen += 1
        if len(rendered) >= target_n:
            break
        convs = row.get("conversations")
        if not isinstance(convs, list):
            continue
        msgs = []
        for c in convs:
            r = role_map.get(c.get("from"))
            v = c.get("value")
            if r and isinstance(v, str):
                msgs.append({"role": r, "content": v})
        text = chatml_render_messages(msgs)
        if text and len(text) >= 100:
            rendered.append({"text": text, "_src": "NousResearch/hermes-function-calling-v1"})
        if seen % 2000 == 0:
            print(f"  ...scanned {seen}, kept {len(rendered)}", flush=True)
    print(f"  hermes scanned={seen} kept={len(rendered)}")
    return rendered


def main():
    all_rows = []
    print("=== Step A1: bfcl/*.jsonl on disk ===")
    all_rows += load_bfcl_local()

    print("\n=== Step A2: external Apache-2.0 datasets ===")
    all_rows += load_glaive(target_n=6000)
    all_rows += load_hermes(target_n=4000)

    # License gate (defensive)
    filtered = [r for r in all_rows if any(r["_src"].startswith(k) or r["_src"] == k or r["_src"].startswith("bfcl_local") for k in APACHE_OK)]
    print(f"\nLicense-gate kept {len(filtered)}/{len(all_rows)} rows")

    # Strip _src before writing
    out_rows = [{"text": r["text"]} for r in filtered]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} rows to {OUT}")
    print(f"  size: {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
