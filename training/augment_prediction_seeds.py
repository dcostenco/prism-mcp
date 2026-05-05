#!/usr/bin/env python3
"""Augment per-locale prediction seed files with top-5K common-word vocabulary.

The existing seeds were built from a small AAC phrase corpus (~3900 phrases,
1500 unigrams). Common English words like "reason", "actually", "because",
"however" never made it in. Users typing those words see 0 prefix matches.

This script keeps the existing curated seed entries (high counts from the AAC
corpus) and PREPENDS the top 5K wordfreq entries at lower baseline counts.
The merge order (curated > common-word baseline) preserves AAC priority while
ensuring every common word is present for prefix matching.

Run:
    .venv-tools/bin/python3 training/augment_prediction_seeds.py

Output:
    Overwrites /Users/admin/prism-aac/constants/predictionSeeds/<lang>.ts
    in-place. Existing bigrams/trigrams are preserved unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

from wordfreq import top_n_list, word_frequency

SEED_DIR = Path("/Users/admin/prism-aac/constants/predictionSeeds")
# Bumped from 5000 to 20000 after a tester reported that AAC-essential
# concrete nouns like Russian "дуб" (oak), "ёж" (hedgehog), or English
# "antelope", "anchovy" were missing. Rank ~17K is needed to cover the
# tail of common concrete vocabulary that wordfreq under-weights vs.
# abstract / social-media web text. Per-locale seed file goes from ~330 KB
# to ~1.1 MB but is code-split per language (only one is shipped to a
# given user) so the runtime cost is bounded.
TOP_N_COMMON = 20000

# Map our locale codes to wordfreq language codes.
LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "pt": "pt",
    "ru": "ru", "uk": "uk", "ar": "ar", "ja": "ja", "ko": "ko",
    "ro": "ro", "zh-Hans": "zh", "zh-Hant": "zh", "zh-HK": "zh",
}

# Allowed Unicode-script regex per language. wordfreq's lists contain loanwords
# from other scripts (e.g. English "i" appears in the Russian list at low rank
# because of mixed-script web text). Without this filter those loanwords pollute
# the corpus and lead to wrong-language suggestions (e.g. "I" leaking into
# Russian predictions). Each pattern matches strings made entirely of letters
# valid for the target language plus optional apostrophe / hyphen.
SCRIPT_FILTER = {
    "en": re.compile(r"^[a-z'\-]+$"),
    "es": re.compile(r"^[a-zñáéíóúü'\-]+$"),
    "fr": re.compile(r"^[a-zàâäçéèêëîïôœùûüÿ'\-]+$"),
    "de": re.compile(r"^[a-zäöüß'\-]+$"),
    "pt": re.compile(r"^[a-záàâãçéêíóôõú'\-]+$"),
    "ro": re.compile(r"^[a-zăâîșțşţ'\-]+$"),
    "ru": re.compile(r"^[а-яё'\-]+$"),
    "uk": re.compile(r"^[а-яєіїґ'\-]+$"),
    "ar": re.compile(r"^[؀-ۿݐ-ݿ'\-]+$"),
    "ja": re.compile(r"^[぀-ゟ゠-ヿ一-鿿]+$"),
    "ko": re.compile(r"^[가-힯ᄀ-ᇿ㄰-㆏]+$"),
    "zh-Hans": re.compile(r"^[一-鿿]+$"),
    "zh-Hant": re.compile(r"^[一-鿿]+$"),
    "zh-HK": re.compile(r"^[一-鿿]+$"),
}

import re as _re_check  # ensure re is in scope (already imported above)


def parse_existing_word_freq(content: str) -> dict[str, int]:
    """Extract WORD_FREQ entries from an existing seed TS file."""
    out: dict[str, int] = {}
    block = re.search(
        r"const WORD_FREQ:[^=]*=\s*\{(.*?)\n\};",
        content,
        flags=re.DOTALL,
    )
    if not block:
        return out
    body = block.group(1)
    for m in re.finditer(r'"([^"]+)":\s*\{\s*count:\s*(\d+)', body):
        out[m.group(1)] = int(m.group(2))
    return out


def augment_word_freq(existing: dict[str, int], lang: str, wordfreq_lang: str) -> dict[str, int]:
    """Add top-N common words to existing dict; counts scale with frequency rank.

    Baseline count formula: rank-1 → 50, rank-5000 → 1. This keeps very common
    words (the, and, of, reason, because...) competitive with AAC-curated
    entries while letting the rarest of the top-5K stay as low-priority
    fallback completions.

    Words that don't match the language's script filter (loanwords from other
    scripts) are excluded — they would produce wrong-language predictions.
    The existing curated entries are also re-filtered the same way to clean
    out any prior pollution.
    """
    script = SCRIPT_FILTER.get(lang)
    common = top_n_list(wordfreq_lang, TOP_N_COMMON)
    out: dict[str, int] = {}
    for rank, word in enumerate(common):
        clean = word.lower().strip()
        if not clean or any(c.isspace() for c in clean):
            continue
        if script and not script.match(clean):
            continue  # loanword from another script — skip
        # Linear decay from 50 (rank 0) to 1 (rank TOP_N_COMMON-1). Words
        # past rank 5K still get baseline 1 — enough to surface in prefix
        # matching but never to outrank curated AAC vocabulary.
        baseline = max(1, 50 - int((rank / TOP_N_COMMON) * 49))
        out[clean] = max(out.get(clean, 0), baseline)
    # Existing AAC-curated counts win when they're already higher — but ALSO
    # re-filter them so any previously polluted seed gets cleaned this run.
    for word, count in existing.items():
        if script and not script.match(word):
            continue
        out[word] = max(count, out.get(word, 0))
    return out


def render_word_freq_block(wf: dict[str, int]) -> str:
    """Render the WORD_FREQ TS literal sorted by count desc."""
    lines = ["const WORD_FREQ: Record<string, WordFreqEntry> = {"]
    for word, count in sorted(wf.items(), key=lambda kv: (-kv[1], kv[0])):
        # Escape any " or \ in the word.
        safe = word.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{safe}": {{ count: {count}, lastUsed: 0 }},')
    lines.append("};")
    return "\n".join(lines)


def replace_word_freq_block(content: str, new_block: str) -> str:
    return re.sub(
        r"const WORD_FREQ:[^=]*=\s*\{.*?\n\};",
        new_block,
        content,
        count=1,
        flags=re.DOTALL,
    )


def update_header_comment(content: str, new_uni_count: int) -> str:
    """Update the auto-generated header so the unigram count reflects the
    augmented seed size."""
    return re.sub(
        r"// (\d+) source phrases · \d+ uni · (\d+) bi · (\d+) tri",
        rf"// \1 source phrases · {new_uni_count} uni (incl. wordfreq top-{TOP_N_COMMON}) · \2 bi · \3 tri",
        content,
        count=1,
    )


def process_lang(lang: str) -> tuple[int, int]:
    seed_path = SEED_DIR / f"{lang}.ts"
    if not seed_path.exists():
        print(f"  skip {lang}: {seed_path} not found")
        return (0, 0)
    wordfreq_lang = LANG_MAP.get(lang)
    if not wordfreq_lang:
        print(f"  skip {lang}: no wordfreq mapping")
        return (0, 0)
    content = seed_path.read_text()
    existing = parse_existing_word_freq(content)
    augmented = augment_word_freq(existing, lang, wordfreq_lang)
    new_block = render_word_freq_block(augmented)
    new_content = replace_word_freq_block(content, new_block)
    new_content = update_header_comment(new_content, len(augmented))
    seed_path.write_text(new_content)
    return (len(existing), len(augmented))


def main():
    print(f"Augmenting prediction seeds in {SEED_DIR}")
    print(f"Adding top-{TOP_N_COMMON} wordfreq entries per locale at baseline count=5")
    print()
    for lang in sorted(LANG_MAP.keys()):
        before, after = process_lang(lang)
        if before or after:
            added = after - before
            print(f"  {lang:8s}  before: {before:5d}  after: {after:5d}  (+{added:5d})")


if __name__ == "__main__":
    main()
