#!/usr/bin/env python3
"""Quick AAC validation baseline against local Ollama prism-coder:7b.

Runs all 50 cases in data/aac_validation.json in parallel, scores each by
fuzzy match against the 'acceptable' answers, and writes a JSON report.

Pass criterion (loose): response substring-matches any 'acceptable' answer
after lowercasing/punct-strip, OR semantic overlap >= 0.6 (token Jaccard).
This is a screening signal, not a final grade — anything <60% pass means
AAC capability is materially missing.
"""
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
MODEL = os.environ.get("MODEL", "prism-coder:7b")
VAL_PATH = Path(__file__).parent / "data" / "aac_validation.json"
OUT_PATH = Path(__file__).parent / "results" / f"aac_baseline_{MODEL.replace(':','_')}.json"

CATEGORY_SYSTEM = {
    "expand": "You expand AAC symbol fragments into a complete, natural utterance. Return ONLY the utterance, no preamble.",
    "predict": "Given a partial sentence, predict the most likely next 1-3 words. Return ONLY the predicted continuation.",
    "intent": "Classify the SCERTS pragmatic function of an utterance (request/comment/protest/social/repair/regulate). Return ONLY the function name.",
    "rephrase": "Rephrase clinical/restrictive language into trauma-informed dignity-preserving language. Return ONLY the rephrased version.",
    "multi_turn": "Given conversation context, generate the next contextually appropriate utterance. Return ONLY the utterance.",
    "vocab_match": "Match the output to the user's developmental/vocabulary level as specified. Return ONLY the matched output.",
}


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'^["\'\s]+|["\'\s]+$', '', s)
    s = re.sub(r'[^\w\s\']', ' ', s)
    return ' '.join(s.split())


def jaccard(a: str, b: str) -> float:
    A = set(normalize(a).split())
    B = set(normalize(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def score(response: str, acceptable: list[str]) -> tuple[bool, float]:
    r = normalize(response)
    if not r:
        return False, 0.0
    best = 0.0
    for a in acceptable:
        an = normalize(a)
        if an in r or r in an:
            return True, 1.0
        best = max(best, jaccard(r, a))
    return best >= 0.6, best


def call_ollama(prompt: str, system: str, num_predict: int = 80, timeout: int = 60) -> tuple[str, int]:
    payload = json.dumps({
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip(), int((time.time() - t0) * 1000)


def run_case(case: dict) -> dict:
    sys_prompt = CATEGORY_SYSTEM.get(case["category"], "You are a helpful AAC assistant. Return ONLY the answer.")
    try:
        resp, latency = call_ollama(case["prompt"], sys_prompt, num_predict=120)
        ok, sim = score(resp, case["acceptable"])
        return {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "response": resp,
            "acceptable": case["acceptable"],
            "ok": ok,
            "similarity": round(sim, 2),
            "latency_ms": latency,
        }
    except Exception as e:
        return {
            "id": case["id"], "category": case["category"], "prompt": case["prompt"],
            "response": None, "ok": False, "similarity": 0.0, "error": str(e),
        }


def main():
    val = json.load(open(VAL_PATH))
    cases = val["cases"]
    print(f"Running {len(cases)} AAC validation cases against {MODEL}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_case, c): c for c in cases}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            status = "OK " if r["ok"] else "FAIL"
            print(f"  [{r['id']:2d}/{len(cases)}] {status} {r['category']:12s} sim={r.get('similarity',0):.2f} {(r.get('response') or r.get('error',''))[:80]}")

    results.sort(key=lambda r: r["id"])
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, [0, 0])
        by_cat[c][1] += 1
        if r["ok"]:
            by_cat[c][0] += 1

    summary = {
        "model": MODEL,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "by_category": {k: {"passed": v[0], "total": v[1], "pct": round(v[0] / v[1], 4)} for k, v in by_cat.items()},
        "wall_time_s": round(time.time() - t0, 1),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== AAC baseline ({MODEL}) ===")
    print(f"Pass rate: {passed}/{total} = {summary['pass_rate']*100:.1f}%")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:12s} {s['passed']:2d}/{s['total']:2d} = {s['pct']*100:5.1f}%")
    print(f"\nReport: {OUT_PATH}")


if __name__ == "__main__":
    main()
