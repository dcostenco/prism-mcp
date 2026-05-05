#!/usr/bin/env python3
"""Patch OFFLINE_DICT_1 in prism-aac/constants/offlineDictionary.ts.

Reads /tmp/offline_dict_translations.json (output from modal_translate_dict.py)
and splices the 8 missing locales (pt, uk, zh-Hans, zh-Hant, zh-HK, ja, ko, ar)
into the existing _OFFLINE_DICT_1 object literal alongside the 6 existing
locales (en, es, fr, de, ru, ro), preserving the visual layout (5 chunks of
100 entries per language, separated by newlines and 4-space indentation).

Usage:
  python3 patch_offline_dict.py
"""
import json
import re
import sys
from pathlib import Path

DICT_TS = Path("/Users/admin/prism-aac/constants/offlineDictionary.ts")
TRANS = Path("/tmp/offline_dict_translations.json")

# JS-string-safe encoding using single quotes — escape backslash, single-quote.
def js_quote(s: str) -> str:
    s = (s or "").replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def render_lang_block(lang: str, words: list[str]) -> str:
    """5 chunks of 100 entries each, joined like the existing en/es/fr blocks."""
    if len(words) != 500:
        raise ValueError(f"{lang}: expected 500 entries, got {len(words)}")
    chunks = []
    for i in range(0, 500, 100):
        chunks.append(",".join(js_quote(w) for w in words[i:i + 100]))
    body = ",\n    ".join(chunks)
    return f"  {js_quote(lang)[1:-1] if False else lang}: [\n    {body}\n  ],\n"


def render_lang_block_keyed(lang: str, words: list[str]) -> str:
    """Render with the lang as a JS object key. CJK locales need quoting since
    they contain hyphens."""
    if len(words) != 500:
        raise ValueError(f"{lang}: expected 500 entries, got {len(words)}")
    chunks = []
    for i in range(0, 500, 100):
        chunks.append(",".join(js_quote(w) for w in words[i:i + 100]))
    body = ",\n    ".join(chunks)
    # Keys with hyphens (zh-Hans, etc.) need quoting; bare identifiers don't.
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lang):
        key = lang
    else:
        key = f"'{lang}'"
    return f"  {key}: [\n    {body}\n  ],\n"


def main():
    if not TRANS.exists():
        sys.exit(f"ERROR: {TRANS} not found. Fetch from Modal first:\n  modal run modal_translate_dict.py::fetch")
    if not DICT_TS.exists():
        sys.exit(f"ERROR: {DICT_TS} not found")

    translations = json.loads(TRANS.read_text())
    expected_langs = {"pt", "uk", "zh-Hans", "zh-Hant", "zh-HK", "ja", "ko", "ar"}
    missing = expected_langs - set(translations.keys())
    if missing:
        sys.exit(f"ERROR: translations missing {missing}")

    # Sanity check entry counts
    for lang in expected_langs:
        n = len(translations[lang])
        if n != 500:
            sys.exit(f"ERROR: {lang} has {n} entries, expected 500")

    content = DICT_TS.read_text()

    # Find the closing `};` of the `_OFFLINE_DICT_1` object literal.
    # Pattern: a trailing `  ],\n};` (last lang block close + obj close)
    m = re.search(r"(\n)(\};\s*\n\s*export const OFFLINE_DICT_1)", content)
    if not m:
        sys.exit("ERROR: could not find _OFFLINE_DICT_1 closing brace")
    insert_at = m.start(2)  # before `};`

    # Render new lang blocks in canonical order
    order = ["pt", "uk", "zh-Hans", "zh-Hant", "zh-HK", "ja", "ko", "ar"]
    new_blocks = "".join(render_lang_block_keyed(lang, translations[lang]) for lang in order)

    # Splice in
    patched = content[:insert_at] + new_blocks + content[insert_at:]
    DICT_TS.write_text(patched)

    print(f"✅ patched {DICT_TS}")
    print(f"   added 8 locales × 500 entries = 4000 new translation pairs")
    sz_kb = DICT_TS.stat().st_size / 1024
    print(f"   file size: {sz_kb:.0f} KB")


if __name__ == "__main__":
    main()
