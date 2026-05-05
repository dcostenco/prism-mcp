#!/usr/bin/env python3
"""Generate OFFLINE_DICT_1 entries for the 8 missing locales.

For each missing lang (pt, uk, ja, ko, zh-Hans, zh-Hant, zh-HK, ar), translates
the 500 English entries (5 categories × 100 each: nouns, verbs, adjectives,
phrases, time/numbers/connectors) using local qwen-coder:32b.

Outputs JSON files to data/offline_dict/<lang>.json that can be patched into
constants/offlineDictionary.ts.
"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "qwen2.5-coder:32b"
OUT_DIR = Path(__file__).parent / "data" / "offline_dict"
SRC_TS = Path("/Users/admin/prism-aac/constants/offlineDictionary.ts")

# Language metadata for the teacher prompt
LANG_META = {
    "pt":      ("Portuguese (Brazilian)",          "Use natural Brazilian Portuguese spelling and vocabulary."),
    "uk":      ("Ukrainian",                        "Use modern Ukrainian (not Russian-influenced surzhyk). Use proper Cyrillic."),
    "ja":      ("Japanese",                         "Use kanji where appropriate, hiragana/katakana otherwise. Common everyday vocabulary."),
    "ko":      ("Korean",                           "Use Hangul. Standard Seoul Korean."),
    "zh-Hans": ("Chinese Simplified (Mandarin)",    "Use Mainland Mandarin with Simplified characters (简体中文)."),
    "zh-Hant": ("Chinese Traditional (Taiwan)",     "Use Taiwanese Mandarin with Traditional characters (繁體中文). Note Taiwan-specific vocabulary differs from HK Cantonese."),
    "zh-HK":   ("Cantonese (Hong Kong)",            "Use Traditional characters with Hong Kong Cantonese vocabulary (廣東話/粵語). Words may differ from Taiwan Mandarin."),
    "ar":      ("Arabic (Modern Standard)",         "Use Modern Standard Arabic (MSA, فصحى). Right-to-left script."),
}


def extract_en_list() -> list:
    """Pull the en: [...] array out of offlineDictionary.ts as a flat list."""
    src = SRC_TS.read_text()
    m = re.search(r"\ben:\s*\[(.*?)^\s*\],", src, re.MULTILINE | re.DOTALL)
    if not m:
        sys.exit("ERROR: en: [...] block not found in offlineDictionary.ts")
    inner = m.group(1)
    words = re.findall(r"'([^']+)'", inner)
    if len(words) < 400:
        sys.exit(f"ERROR: extracted only {len(words)} en words; expected ~500")
    return words


def translate_batch(en_words: list, lang_code: str, lang_name: str, lang_note: str) -> list:
    """Translate a batch of English words to the target language. Returns list
    of strings same length as en_words. Falls back to en if teacher fails."""
    payload = json.dumps({
        "model": TEACHER,
        "system": (
            f"You translate English vocabulary into {lang_name}. "
            f"{lang_note} "
            "Translate ONE-TO-ONE — output exactly the same number of items, "
            "in the same order, with NO additional commentary. Every entry MUST "
            "translate (no English passthrough, no '<same>' or '?'). "
            "Output as a JSON array of strings, e.g. [\"...\", \"...\", ...]."
        ),
        "prompt": (
            f"Translate these {len(en_words)} English entries into {lang_name}. "
            "Return ONLY a JSON array — no explanations, no fences, no markdown.\n\n"
            f"Input: {json.dumps(en_words, ensure_ascii=False)}"
        ),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192},
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = (data.get("response") or "").strip()
            text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
            if "[" in text and "]" in text:
                text = text[text.index("["): text.rindex("]") + 1]
            arr = json.loads(text)
            if isinstance(arr, list) and len(arr) == len(en_words):
                return [str(x).strip() for x in arr]
            print(f"  warn: got {len(arr) if isinstance(arr, list) else '?'} items, expected {len(en_words)} — retry", file=sys.stderr)
        except Exception as e:
            print(f"  warn: attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2)
    # Failure fallback — return en words unchanged so structure is preserved
    print(f"  ! batch failed after 3 attempts; falling back to en for {len(en_words)} entries", file=sys.stderr)
    return list(en_words)


def translate_lang(lang_code: str, en_words: list, batch_size: int = 50) -> list:
    """Translate the full 500-word list for one language."""
    name, note = LANG_META[lang_code][0], LANG_META[lang_code][1]
    out = []
    n_batches = (len(en_words) + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(n_batches):
        chunk = en_words[i * batch_size:(i + 1) * batch_size]
        translated = translate_batch(chunk, lang_code, name, note)
        out.extend(translated)
        elapsed = time.time() - t0
        print(f"  [{lang_code}] batch {i+1}/{n_batches}  ({len(out)}/{len(en_words)} words, {elapsed:.0f}s)", flush=True)
    print(f"[{lang_code}] done: {len(out)} words in {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    en_words = extract_en_list()
    print(f"=== extracted {len(en_words)} English words from offlineDictionary.ts ===\n")

    # Run all 8 langs in parallel — Ollama serializes anyway but parallel
    # workers reduce idle time between batches
    langs = list(LANG_META.keys())
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(translate_lang, code, en_words): code for code in langs}
        for f in as_completed(futures):
            code = futures[f]
            try:
                out = f.result()
                (OUT_DIR / f"{code}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
                print(f"  wrote {code}.json ({len(out)} words)")
            except Exception as e:
                print(f"  ! {code} failed: {e}", file=sys.stderr)

    print(f"\nAll dictionaries in {OUT_DIR}")


if __name__ == "__main__":
    main()
