"""Modal H100 job — translate the OFFLINE_DICT_1 EN canonical 500-word list
to 8 missing locales using the Qwen2.5-72B-Instruct teacher. Output is
strict-JSON 500-element arrays per lang, index-aligned to the EN source so
the existing `OFFLINE_DICT_1[fromLang][i] -> OFFLINE_DICT_1[toLang][i]`
lookup in translateService.ts continues to work.

Run:
  cd /Users/admin/prism/training
  modal volume put prism-offline-phrases /tmp/offline_dict_en.json /offline_dict_en.json --force
  modal run --detach modal_translate_dict.py::translate_all
  modal volume get prism-offline-phrases offline_dict_translations.json /tmp/offline_dict_translations.json --force
"""
import json
from pathlib import Path

import modal

app = modal.App("prism-offline-dict")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub",
    )
)

vol = modal.Volume.from_name("prism-offline-phrases", create_if_missing=True)

TARGET_LANGS = {
    "pt": "Portuguese (Brazilian)",
    "uk": "Ukrainian",
    "zh-Hans": "Simplified Chinese (Mandarin, mainland China)",
    "zh-Hant": "Traditional Chinese (Mandarin, Taiwan)",
    "zh-HK": "Cantonese (Traditional script, Hong Kong colloquial)",
    "ja": "Japanese (use kanji+hiragana, common everyday register)",
    "ko": "Korean (use hangul, common everyday register)",
    "ar": "Modern Standard Arabic",
}

# Index ranges describe the section semantics so the teacher gets the right
# part-of-speech for each slot. Matches comments in offlineDictionary.ts.
SECTIONS = [
    (0, 100, "common nouns"),
    (100, 200, "common verbs (use the most natural everyday infinitive form)"),
    (200, 300, "common adjectives"),
    (300, 400, "everyday phrases (translate idiomatically, not word-for-word)"),
    (400, 500, "time words, numbers, connectors, function words"),
]


@app.function(
    image=image,
    gpu="H100:4",
    timeout=3600,
    cpu=8.0,
    memory=64000,
    volumes={"/data": vol},
)
def translate_all():
    import time
    from vllm import LLM, SamplingParams

    print("loading 72B teacher on 4xH100...")
    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=4,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enforce_eager=False,
    )
    print(f"model ready in {time.time()-t0:.0f}s")

    en_words = json.loads(Path("/data/offline_dict_en.json").read_text())
    assert len(en_words) == 500, f"expected 500 EN entries, got {len(en_words)}"

    sp = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096, stop=None)

    out: dict[str, list[str]] = {}
    for lang_code, lang_desc in TARGET_LANGS.items():
        print(f"\n=== {lang_code} ({lang_desc}) ===", flush=True)
        translated: list[str] = [""] * 500

        for start, end, section_desc in SECTIONS:
            block = en_words[start:end]
            sys_msg = (
                "You are a professional translator producing word-for-word lookup "
                "entries for an offline AAC (augmentative communication) app used "
                "by disabled children. Each English entry must map to ONE natural, "
                "common-register equivalent in the target language. Preserve order. "
                "Output MUST be a JSON array of exactly the same length as the input."
            )
            user_msg = (
                f"Translate the following {len(block)} English {section_desc} into "
                f"{lang_desc}. Output strict JSON: an array of {len(block)} strings, "
                f"index-aligned to the input. No comments, no preamble, no trailing text.\n\n"
                f"INPUT (JSON array of {len(block)} English entries):\n"
                f"{json.dumps(block, ensure_ascii=False)}\n\n"
                f"OUTPUT (JSON array of {len(block)} {lang_desc} entries):"
            )
            prompt = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            res = llm.generate([prompt], sp)
            text = res[0].outputs[0].text.strip()
            # try to extract JSON array
            arr = _parse_json_array(text, expected_len=len(block))
            if arr is None:
                print(f"  [{lang_code}] section {start}-{end}: parse failed, retrying with stricter prompt")
                # one retry with stricter wording
                user_msg2 = user_msg + "\n\nRespond with ONLY the JSON array. No code fences. No surrounding text."
                prompt2 = prompt.replace(user_msg, user_msg2)
                res2 = llm.generate([prompt2], sp)
                arr = _parse_json_array(res2[0].outputs[0].text.strip(), expected_len=len(block))
            if arr is None:
                print(f"  [{lang_code}] section {start}-{end}: STILL failed, leaving English fallback")
                arr = list(block)
            for i, v in enumerate(arr):
                translated[start + i] = v if isinstance(v, str) and v.strip() else block[i]
            print(f"  [{lang_code}] {start:3d}-{end:3d} ({section_desc[:30]:30s}) ok")

        out[lang_code] = translated

    Path("/data/offline_dict_translations.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )
    vol.commit()
    print("\nwrote /data/offline_dict_translations.json")
    return {"langs": list(out.keys()), "counts": {k: len(v) for k, v in out.items()}}


def _parse_json_array(text: str, expected_len: int):
    """Tolerant JSON-array extractor. Strips ```json fences if present."""
    import re
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find the outermost [ ... ] that contains the array
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        arr = json.loads(blob)
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != expected_len:
        return None
    return arr


@app.local_entrypoint()
def upload_en():
    """Push /tmp/offline_dict_en.json to the volume."""
    import subprocess
    p = Path("/tmp/offline_dict_en.json")
    if not p.exists():
        raise SystemExit(f"missing {p} — generate via the EN extractor first")
    print(f"uploading {p} ({p.stat().st_size} B) to volume prism-offline-phrases:/offline_dict_en.json")
    subprocess.run(
        ["modal", "volume", "put", "prism-offline-phrases", str(p), "/offline_dict_en.json", "--force"],
        check=True,
    )
    print("upload complete")


@app.local_entrypoint()
def go():
    """Submit translate job."""
    print("submitting OFFLINE_DICT_1 translation job to Modal H100x4...")
    result = translate_all.remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def fetch():
    """Pull translations back."""
    import subprocess
    out = "/tmp/offline_dict_translations.json"
    subprocess.run(
        ["modal", "volume", "get", "prism-offline-phrases", "offline_dict_translations.json", out, "--force"],
        check=True,
    )
    print(f"fetched -> {out}")
