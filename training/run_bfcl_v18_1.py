"""Run official BFCL eval on v18.1 surgical adapter.

Approach: pre-merge the LoRA adapter into the base Qwen2.5-Coder-7B-Instruct
weights (so vllm 0.6.6 doesn't need LoRA-aware serving), then serve the
merged model via vllm's OpenAI-compatible endpoint and run `bfcl generate`
+ `bfcl evaluate` on the leaderboard categories.

Pinned to vllm==0.6.6.post1 + transformers==4.45.2 — newer transformers
removed Qwen2Tokenizer.all_special_tokens_extended which both vllm 0.8
and 0.10 still reference, causing AttributeError on startup.

Run:
  modal run --detach run_bfcl_v18_1.py::main
  modal run run_bfcl_v18_1.py::fetch    # download scores when done

Adapter source: prism-v18-1:/v18_1_adapter
Results sink:   bfcl-results:/v18_1_*
"""
import modal
import os

app = modal.App("prism-bfcl-v18-1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # vllm 0.6.6 + transformers 4.45.2 — last known-good pair for Qwen2.
        "vllm==0.6.6.post1",
        "transformers==4.45.2",
        "peft==0.13.2",          # for merging LoRA → base weights
        "bfcl-eval",
        "torch",
        "accelerate",
        "huggingface_hub",
        "qwen-agent",
        "soundfile",
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)

results_vol = modal.Volume.from_name("bfcl-results", create_if_missing=True)
adapter_vol = modal.Volume.from_name("prism-v18-1")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={"/results": results_vol, "/adapter_vol": adapter_vol},
)
def run_bfcl_eval(categories: list[str] | None = None, cat_timeout: int = 3600):
    import subprocess, time, json, urllib.request, sys, shutil

    BASE_HF = "Qwen/Qwen2.5-Coder-7B-Instruct"
    ADAPTER_PATH = "/adapter_vol/v18_1_adapter"
    FUSED_PATH = "/tmp/v18_1_fused"
    MODEL_NAME = "prism-coder-7b-v18-1"
    PORT = 8765

    # ── Step 0: Merge LoRA → base model ──────────────────────────────────
    print(f"\n[step 0] Merging adapter {ADAPTER_PATH} into {BASE_HF}…", flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_HF)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_HF, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.save_pretrained(FUSED_PATH, safe_serialization=True)
    tok.save_pretrained(FUSED_PATH)
    del base, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  fused model saved to {FUSED_PATH}", flush=True)

    # ── Step 1: Patch BFCL model_config to point at our fused model ──────
    import bfcl_eval.constants.model_config as mc
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    config_path = mc.__file__
    with open(config_path, "r") as f:
        content = f.read()
    insert_point = content.rfind("}\n\n\nMODEL_CONFIG_MAPPING")
    if insert_point > 0:
        patch = f'''
    "{MODEL_NAME}": ModelConfig(
        model_name="{FUSED_PATH}",
        display_name="Prism-Coder-7B v18.1 (FC)",
        url="https://huggingface.co/dcostenco/prism-coder-7b",
        org="Synalux", license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None, output_price=None,
        is_fc_model=True, underscore_to_dot=True,
    ),
'''
        content = content[:insert_point] + patch + content[insert_point:]
        with open(config_path, "w") as f:
            f.write(content)
        print(f"  BFCL config patched", flush=True)

    # ── Step 2: Start vLLM serving the fused model ───────────────────────
    print(f"\n[step 1] Starting vLLM for {FUSED_PATH}…", flush=True)
    vllm_proc = subprocess.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", FUSED_PATH, "--port", str(PORT),
         "--dtype", "bfloat16", "--max-model-len", "8192",
         "--gpu-memory-utilization", "0.9",
         "--served-model-name", FUSED_PATH],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    for i in range(900):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2) as resp:
                if json.loads(resp.read().decode()).get("data"):
                    print(f"  vLLM ready after {i}s", flush=True)
                    break
        except Exception:
            if i % 30 == 0 and i > 0:
                alive = vllm_proc.poll() is None
                print(f"  vLLM loading… {i}s ({'alive' if alive else 'DEAD'})", flush=True)
                if not alive:
                    err = vllm_proc.stderr.read().decode()[-2000:]
                    print(f"  vLLM CRASHED:\n{err}", flush=True)
                    return f"FAILED: vLLM crashed\n{err}"
            time.sleep(1)
    else:
        err = vllm_proc.stderr.read().decode()[-2000:]
        return f"FAILED: vLLM timeout\n{err}"

    # ── Step 3: Generate + evaluate per category ─────────────────────────
    env = {**os.environ,
           "LOCAL_SERVER_ENDPOINT": f"http://localhost:{PORT}/v1",
           "LOCAL_SERVER_MODEL_NAME": FUSED_PATH}

    if categories is None:
        categories = [
            "simple_python", "multiple", "parallel", "parallel_multiple",
            "simple_java", "simple_javascript",
            "irrelevance",                        # abstention
            "live_simple", "live_multiple",       # live (real APIs)
            "multi_turn_base",                    # the bottleneck
            "multi_turn_long_context",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
        ]
    print(f"\n[step 2] Categories ({len(categories)}): {categories}", flush=True)

    all_output = []
    def _commit_partial(tag):
        try:
            with open("/results/v18_1_partial.txt", "w") as f:
                f.write(f"[{tag}]\n" + "\n".join(all_output))
            results_vol.commit()
        except Exception as e:
            print(f"  partial commit failed: {e}", flush=True)

    for cat in categories:
        print(f"\n=== Generating: {cat} ===", flush=True)
        try:
            r = subprocess.run(
                ["bfcl", "generate", "--model", MODEL_NAME, "--test-category", cat,
                 "--skip-server-setup", "--temperature", "0.001", "--num-threads", "16"],
                capture_output=True, text=True, timeout=cat_timeout, env=env,
            )
            print(f"  rc={r.returncode} {r.stdout[-200:]}", flush=True)
            if r.returncode != 0:
                print(f"  stderr={r.stderr[-300:]}", flush=True)
            all_output.append(f"[gen {cat}] rc={r.returncode}\n{r.stdout[-500:]}\n{r.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {cat_timeout}s — continuing", flush=True)
            all_output.append(f"[gen {cat}] TIMEOUT")
        _commit_partial(f"after_gen_{cat}")

    for cat in categories:
        print(f"\n=== Evaluating: {cat} ===", flush=True)
        try:
            r = subprocess.run(
                ["bfcl", "evaluate", "--model", MODEL_NAME, "--test-category", cat],
                capture_output=True, text=True, timeout=900, env=env,
            )
            print(f"  {r.stdout[-300:]}", flush=True)
            all_output.append(f"[eval {cat}]\n{r.stdout}")
        except subprocess.TimeoutExpired:
            print(f"  EVAL TIMEOUT for {cat} — skipping", flush=True)
            all_output.append(f"[eval {cat}] TIMEOUT")
        _commit_partial(f"after_eval_{cat}")

    print("\n=== SCORES ===", flush=True)
    r = subprocess.run(["bfcl", "scores", "--model", MODEL_NAME],
                       capture_output=True, text=True, timeout=60, env=env)
    scores = r.stdout
    print(scores, flush=True)

    with open("/results/v18_1_full.txt", "w") as f:
        f.write("\n".join(all_output) + "\n\nSCORES:\n" + scores)
    with open("/results/v18_1_scores.txt", "w") as f:
        f.write(scores)
    results_vol.commit()

    vllm_proc.terminate()
    return scores or "\n".join(all_output[-3:])


@app.local_entrypoint()
def smoke():
    """3-category smoke (~10 min)."""
    cats = ["simple_python", "multi_turn_base", "irrelevance"]
    print(f"Launching SMOKE BFCL on v18.1 (A100-80GB): {cats}")
    out = run_bfcl_eval.remote(categories=cats, cat_timeout=1800)
    print("\n" + "=" * 60)
    print("BFCL v18.1 SMOKE RESULTS")
    print("=" * 60)
    print(out)


@app.local_entrypoint()
def main():
    print("Launching BFCL v18.1 on A100-80GB (use `--detach` for survival)…")
    out = run_bfcl_eval.remote()
    print("\n" + "=" * 60)
    print("BFCL v18.1 FULL RESULTS")
    print("=" * 60)
    print(out)


@app.local_entrypoint()
def fetch():
    """Download v18.1 scores from the Modal Volume."""
    out_dir = "/Users/admin/prism/training"
    for fname in ("v18_1_scores.txt", "v18_1_full.txt", "v18_1_partial.txt"):
        local_path = os.path.join(out_dir, f"bfcl_{fname}")
        try:
            with open(local_path, "wb") as f:
                for chunk in results_vol.read_file(fname):
                    f.write(chunk)
            print(f"Saved {local_path}")
        except FileNotFoundError:
            print(f"Not on volume yet: {fname}")
