#!/usr/bin/env python3
"""Extract per-locale n-gram tables from the offline phrase corpus.

Emits per-locale TS files so Next.js can code-split — each language only
loads its own ~330 KB seed bundle, not all 14 locales.

Outputs:
  /Users/admin/prism-aac/constants/predictionSeeds/index.ts
    - loadPredictionSeed(lang) -> Promise<{ wordFreq, bigrams, trigrams }>
    - SUPPORTED_SEED_LANGS: string[]
  /Users/admin/prism-aac/constants/predictionSeeds/<lang>.ts
    - default export: { wordFreq, bigrams, trigrams }

Format matches predictionEngine.ts exactly:
  bigram key:  "{prev}|{next}"        (no trailing pipe — matches recordBigram)
  trigram key: "{prev2}|{prev1}|{next}"

Tokenization:
  - Latin / Cyrillic: split on whitespace, strip punctuation
  - CJK (zh-*, ja): char-level n-grams (no word boundaries)
  - Others: whitespace tokenize
"""
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

PHRASE_DIR = Path("/Users/admin/prism/training/data/offline_phrases")
OUT_DIR = Path("/Users/admin/prism-aac/constants/predictionSeeds")
LEGACY_FILE = Path("/Users/admin/prism-aac/constants/predictionSeeds.ts")

CHAR_LEVEL = {"zh", "zh-Hans", "zh-Hant", "zh-HK", "ja"}

TOP_N_UNIGRAMS = 1500
TOP_N_BIGRAMS = 3000
TOP_N_TRIGRAMS = 2500

SEED_LAST_USED = 0

PUNCT_RE = re.compile(r"[ -⁯⸀-⹿\\'!\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~。、，！？；：「」『』（）《》〈〉【】…—،؛؟٭]")


def tokenize_word(text: str) -> list[str]:
    text = PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t]


def tokenize_char(text: str) -> list[str]:
    out = []
    for c in text:
        if c.isspace():
            continue
        cat = unicodedata.category(c)
        if cat.startswith("P") or cat.startswith("Z") or cat == "Cc":
            continue
        out.append(c)
    return out


def tokenize(text: str, lang: str) -> list[str]:
    if lang in CHAR_LEVEL:
        return tokenize_char(text)
    return tokenize_word(text)


def topn(counter: Counter, n: int) -> list[tuple[str, int]]:
    return counter.most_common(n)


def extract_for_lang(lang: str, by_cat: dict[str, list[str]]) -> dict:
    uni: Counter = Counter()
    bi: Counter = Counter()
    tri: Counter = Counter()
    n_phrases = 0
    for cat, phrases in by_cat.items():
        for p in phrases:
            if not p:
                continue
            n_phrases += 1
            toks = tokenize(p, lang)
            if not toks:
                continue
            for t in toks:
                uni[t] += 1
            for i in range(len(toks) - 1):
                bi[f"{toks[i]}|{toks[i+1]}"] += 1
            for i in range(len(toks) - 2):
                tri[f"{toks[i]}|{toks[i+1]}|{toks[i+2]}"] += 1
    return {
        "n_phrases": n_phrases,
        "unigrams": dict(topn(uni, TOP_N_UNIGRAMS)),
        "bigrams": dict(topn(bi, TOP_N_BIGRAMS)),
        "trigrams": dict(topn(tri, TOP_N_TRIGRAMS)),
    }


def render_lang_file(lang: str, data: dict, timestamp: str) -> str:
    """One TS file per locale. Default-exports the seed object."""
    def render_record(name: str, table: dict) -> str:
        lines = [f"const {name}: Record<string, WordFreqEntry> = {{"]
        for k, v in table.items():
            ks = json.dumps(k, ensure_ascii=False)
            lines.append(f"  {ks}: {{ count: {v}, lastUsed: {SEED_LAST_USED} }},")
        lines.append("};")
        return "\n".join(lines)

    parts = [
        f"// Auto-generated for locale '{lang}' on {timestamp}.",
        "// DO NOT edit — regenerate via training/build_prediction_seeds.py.",
        f"// {data['n_phrases']} source phrases · "
        f"{len(data['unigrams'])} uni · {len(data['bigrams'])} bi · "
        f"{len(data['trigrams'])} tri",
        "import { WordFreqEntry } from '@/types';",
        "",
        render_record("WORD_FREQ", data["unigrams"]),
        "",
        render_record("BIGRAMS", data["bigrams"]),
        "",
        render_record("TRIGRAMS", data["trigrams"]),
        "",
        "const seed = { wordFreq: WORD_FREQ, bigrams: BIGRAMS, trigrams: TRIGRAMS };",
        "export default seed;",
        "",
    ]
    return "\n".join(parts)


def render_index(langs: list[str], timestamp: str) -> str:
    """index.ts exposes loadPredictionSeed(lang) with dynamic per-locale import.

    Each lang resolves to its own webpack chunk so users only download seeds
    for their currently-selected locale.
    """
    lines = [
        f"// Auto-generated on {timestamp}. DO NOT edit by hand.",
        "// regenerate via training/build_prediction_seeds.py",
        "import { WordFreqEntry } from '@/types';",
        "",
        "export interface PredictionSeed {",
        "  wordFreq: Record<string, WordFreqEntry>;",
        "  bigrams: Record<string, WordFreqEntry>;",
        "  trigrams: Record<string, WordFreqEntry>;",
        "}",
        "",
        f"export const SUPPORTED_SEED_LANGS = {json.dumps(langs)} as const;",
        "export type SeedLang = (typeof SUPPORTED_SEED_LANGS)[number];",
        "",
        "const cache = new Map<string, PredictionSeed>();",
        "const inflight = new Map<string, Promise<PredictionSeed>>();",
        "",
        "export function getCachedPredictionSeed(lang: string): PredictionSeed | null {",
        "  return cache.get(lang) ?? null;",
        "}",
        "",
        "export async function loadPredictionSeed(lang: string): Promise<PredictionSeed> {",
        "  const cached = cache.get(lang);",
        "  if (cached) return cached;",
        "  const pending = inflight.get(lang);",
        "  if (pending) return pending;",
        "  const p = (async () => {",
        "    try {",
        "      const mod = await loadByLang(lang);",
        "      cache.set(lang, mod);",
        "      return mod;",
        "    } catch {",
        "      const empty: PredictionSeed = { wordFreq: {}, bigrams: {}, trigrams: {} };",
        "      cache.set(lang, empty);",
        "      return empty;",
        "    } finally {",
        "      inflight.delete(lang);",
        "    }",
        "  })();",
        "  inflight.set(lang, p);",
        "  return p;",
        "}",
        "",
        "async function loadByLang(lang: string): Promise<PredictionSeed> {",
        "  switch (lang) {",
    ]
    for lg in langs:
        lines.append(f"    case {json.dumps(lg)}: return (await import({json.dumps('./' + lg)})).default;")
    lines.append("    default: return { wordFreq: {}, bigrams: {}, trigrams: {} };")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main():
    if not PHRASE_DIR.exists():
        sys.exit(f"ERROR: {PHRASE_DIR} not found. Run modal_offline_phrases.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    extracted: dict[str, dict] = {}
    for path in sorted(PHRASE_DIR.glob("*.json")):
        lang = path.stem
        try:
            by_cat = json.loads(path.read_text())
        except Exception as e:
            print(f"  ! {lang}: parse failed ({e})", file=sys.stderr)
            continue
        data = extract_for_lang(lang, by_cat)
        extracted[lang] = data
        print(f"  {lang:10s}  {data['n_phrases']:4d} phrases  "
              f"{len(data['unigrams']):4d} uni  {len(data['bigrams']):4d} bi  {len(data['trigrams']):4d} tri",
              flush=True)

    if not extracted:
        sys.exit("ERROR: no language files were extracted")

    # Per-locale TS files
    total_kb = 0
    for lang, data in extracted.items():
        out = OUT_DIR / f"{lang}.ts"
        out.write_text(render_lang_file(lang, data, timestamp))
        total_kb += out.stat().st_size / 1024

    # Index file with dynamic import switch
    langs = sorted(extracted.keys())
    (OUT_DIR / "index.ts").write_text(render_index(langs, timestamp))

    # Remove the legacy single-file output if present (replaced by directory)
    if LEGACY_FILE.exists():
        LEGACY_FILE.unlink()
        print(f"  removed legacy {LEGACY_FILE.name}")

    print(f"\nWrote {OUT_DIR}/  ({total_kb:.0f} KB total across {len(langs)} locales)")
    print(f"  index.ts  +  {len(langs)} per-locale chunks (~{total_kb/len(langs):.0f} KB each)")


if __name__ == "__main__":
    main()
