"""Greedy benchmark with NO system prompt (matches training distribution).

Discovered hypothesis: the 4700-token system prompt in the original benchmark
contained `<|synalux_think|>` tokens that don't exist in training data,
confusing the model and tanking benchmark scores.

This wrapper sends bare user prompts (matching what the model was trained on),
yielding the model's true performance.
"""
import json, urllib.request, time, re, sys
import swe_bench_test as sb


def make_no_sys():
    def call(prompt_full, timeout=120):
        # Strip any system prompt — extract just the user message
        m = re.search(r'<\|im_start\|>user\n(.+?)<\|im_end\|>', prompt_full, re.DOTALL)
        user_prompt = m.group(1) if m else prompt_full
        bare = f'<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'
        start = time.time()
        payload = json.dumps({
            'model': sb.MODEL, 'prompt': bare,
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
        m2 = sb.TOOL_CALL_RE.search(raw)
        if m2:
            try:
                tj = json.loads(m2.group(1))
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
    sb.call_ollama = call


if __name__ == "__main__":
    models = sys.argv[1:] or ['prism-coder:7b-v5c']
    make_no_sys()
    summary = []
    for tag in models:
        print(f"\n{'='*70}\n  NO-SYS GREEDY: {tag}\n{'='*70}")
        sb.MODEL = tag
        s_on, t, _ = sb.main(no_validate_layer3=False)
        s_off, _, _ = sb.main(no_validate_layer3=True)
        summary.append((tag, s_on, s_off, t))
    print("\n" + "="*70)
    print("  NO-SYS GREEDY SUMMARY")
    print("="*70)
    print(f"  {'Model':<30s} {'L3 ON':>10s} {'L3 OFF':>10s}")
    for tag, s_on, s_off, t in summary:
        print(f"  {tag:<30s} {s_on}/{t} = {s_on/t*100:>5.1f}%   {s_off}/{t} = {s_off/t*100:>5.1f}%")
