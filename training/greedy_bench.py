"""Greedy (temp 0.0) deterministic benchmark wrapper.

Runs swe_bench_test on a list of models with deterministic decoding.
Eliminates sampling variance so we can see true model differences.
"""
import json, urllib.request, time, re, sys, importlib

import swe_bench_test as sb


def make_greedy(layer3=True):
    """Patch call_ollama to use temp=0.0."""
    def greedy(prompt, timeout=120):
        start = time.time()
        payload = json.dumps({
            'model': sb.MODEL,
            'prompt': prompt,
            'stream': False, 'raw': True,
            'options': {'temperature': 0.0, 'num_predict': 512}
        }).encode()
        try:
            req = urllib.request.Request(sb.OLLAMA_API, data=payload,
                                         headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                raw = data.get('response','').strip()
        except Exception as e:
            return (str(e),'ERROR',{},time.time()-start)
        latency = time.time()-start
        m = sb.TOOL_CALL_RE.search(raw)
        if m:
            try:
                tj = json.loads(m.group(1))
                return (raw, tj.get('name','UNKNOWN'),
                        tj.get('arguments', tj.get('args',{})), latency)
            except json.JSONDecodeError: pass
        j = re.search(r'(\{[^{}]*\"name\"\s*:\s*\"[^\"]+?\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', raw)
        if j:
            try:
                tj = json.loads(j.group(0))
                return (raw, tj.get('name','UNKNOWN'),
                        tj.get('arguments', tj.get('args',{})), latency)
            except json.JSONDecodeError: pass
        return (raw, 'NO_TOOL', {}, latency)
    sb.call_ollama = greedy


def bench(model_tag, layer3=True):
    sb.MODEL = model_tag
    s,t,r = sb.main(no_validate_layer3=not layer3)
    return s, t, r


if __name__ == "__main__":
    models = sys.argv[1:] or ['prism-coder:7b-v4a','prism-coder:7b-v5b','prism-coder:7b-v5c',
                              'prism-coder:7b-v5d','prism-coder:7b-v5e','prism-coder:7b-v5f']
    make_greedy()

    summary = []
    for tag in models:
        print(f"\n{'='*70}\n  GREEDY BENCHMARK: {tag}\n{'='*70}")
        # Run with Layer 3 ON
        s_on, t, _ = bench(tag, layer3=True)
        # Run with Layer 3 OFF
        s_off, _, _ = bench(tag, layer3=False)
        summary.append((tag, s_on, s_off, t))

    print("\n" + "="*70)
    print("  GREEDY BENCHMARK SUMMARY (deterministic)")
    print("="*70)
    print(f"  {'Model':<30s} {'L3 ON':>8s} {'L3 OFF':>8s}")
    for tag, s_on, s_off, t in summary:
        print(f"  {tag:<30s} {s_on}/{t} = {s_on/t*100:>5.1f}%   {s_off}/{t} = {s_off/t*100:>5.1f}%")
