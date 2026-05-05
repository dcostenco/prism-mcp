"""
Run official BFCL eval on Modal cloud GPU.
Patches BFCL model_config at runtime, serves model via vLLM.
"""
import modal
import os

app = modal.App("prism-bfcl-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Pinned pair: newer transformers (>=4.46) removed Qwen2Tokenizer's
        # `all_special_tokens_extended` attribute, which vllm 0.8/0.10 still
        # references during tokenizer caching → AttributeError on startup.
        # The 0.6.6.post1 / 4.45.2 combo is the last known-good for Qwen2.5.
        "vllm==0.6.6.post1",
        "transformers==4.45.2",
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

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/results": results_vol},
)
def run_bfcl_eval(categories: list[str] | None = None, cat_timeout: int = 3600):
    import subprocess, time, json, urllib.request, sys

    MODEL_HF = "Qwen/Qwen2.5-Coder-7B-Instruct"
    MODEL_NAME = "prism-coder-7b-FC"
    PORT = 8765

    # Step 1: Patch BFCL model_config BEFORE any bfcl import
    import bfcl_eval.constants.model_config as mc
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    mc.local_inference_model_map[MODEL_NAME] = mc.ModelConfig(
        model_name=MODEL_HF,
        display_name="Prism-Coder-7B (FC)",
        url="https://huggingface.co/dcostenco/prism-coder-7b",
        org="Synalux",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    mc.MODEL_CONFIG_MAPPING[MODEL_NAME] = mc.local_inference_model_map[MODEL_NAME]
    print(f"Registered {MODEL_NAME} in BFCL config: {MODEL_NAME in mc.MODEL_CONFIG_MAPPING}")

    # Also write a patch file that the subprocess will load
    patch_code = f'''
import bfcl_eval.constants.model_config as mc
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
mc.local_inference_model_map["{MODEL_NAME}"] = mc.ModelConfig(
    model_name="{MODEL_HF}", display_name="Prism-Coder-7B (FC)",
    url="https://huggingface.co/dcostenco/prism-coder-7b", org="Synalux",
    license="apache-2.0", model_handler=QwenFCHandler, input_price=None,
    output_price=None, is_fc_model=True, underscore_to_dot=True,
)
mc.MODEL_CONFIG_MAPPING["{MODEL_NAME}"] = mc.local_inference_model_map["{MODEL_NAME}"]
'''
    # Find and patch the installed bfcl model_config.py directly
    config_path = mc.__file__
    print(f"Patching {config_path}")
    with open(config_path, "r") as f:
        content = f.read()

    # Add our model to the local_inference_model_map
    insert_point = content.rfind("}\n\n\nMODEL_CONFIG_MAPPING")
    if insert_point > 0:
        patch = f'''
    # Prism-Coder-7B — patched at runtime
    "{MODEL_NAME}": ModelConfig(
        model_name="{MODEL_HF}",
        display_name="Prism-Coder-7B (FC)",
        url="https://huggingface.co/dcostenco/prism-coder-7b",
        org="Synalux",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
'''
        content = content[:insert_point] + patch + content[insert_point:]
        with open(config_path, "w") as f:
            f.write(content)
        print("Config patched successfully")
    else:
        print("WARNING: Could not find insertion point in model_config.py")

    # Verify
    result = subprocess.run(["bfcl", "models"], capture_output=True, text=True)
    if MODEL_NAME in result.stdout:
        print(f"✅ {MODEL_NAME} registered in BFCL")
    else:
        print(f"❌ {MODEL_NAME} NOT found. Available models containing 'prism':")
        print([l for l in result.stdout.split('\n') if 'prism' in l.lower()])

    # Step 2: Start vLLM server
    print(f"\nStarting vLLM for {MODEL_HF}...")
    vllm_proc = subprocess.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL_HF, "--port", str(PORT),
         "--dtype", "bfloat16", "--max-model-len", "8192",
         "--gpu-memory-utilization", "0.9"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    for i in range(600):
        try:
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    print(f"vLLM ready after {i}s", flush=True)
                    break
        except:
            if i % 30 == 0 and i > 0:
                alive = vllm_proc.poll() is None
                print(f"  vLLM loading... {i}s (process {'alive' if alive else 'DEAD'})", flush=True)
                if not alive:
                    stderr = vllm_proc.stderr.read().decode()[-2000:]
                    print(f"vLLM CRASHED:\n{stderr}", flush=True)
                    return f"FAILED: vLLM crashed\n{stderr}"
            time.sleep(1)
    else:
        stderr = vllm_proc.stderr.read().decode()[-2000:]
        print(f"vLLM timeout after 600s:\n{stderr}", flush=True)
        return f"FAILED: vLLM timeout\n{stderr}"

    # Step 3: Run BFCL
    env = {**os.environ,
           "LOCAL_SERVER_ENDPOINT": f"http://localhost:{PORT}/v1",
           "LOCAL_SERVER_MODEL_NAME": MODEL_HF}

    if categories is None:
        categories = ["irrelevance", "simple_python", "multiple", "parallel", "parallel_multiple", "simple_java", "simple_javascript"]
    print(f"Categories to run ({len(categories)}): {categories}", flush=True)
    print(f"Per-category timeout: {cat_timeout}s", flush=True)

    def _commit_partial(tag, all_output):
        try:
            with open("/results/partial_progress.txt", "w") as f:
                f.write(f"[{tag}]\n" + "\n".join(all_output))
            results_vol.commit()
        except Exception as e:
            print(f"  partial commit failed: {e}", flush=True)

    all_output = []
    for cat in categories:
        print(f"\n=== Generating: {cat} ===", flush=True)
        try:
            r = subprocess.run(
                ["bfcl", "generate", "--model", MODEL_NAME, "--test-category", cat,
                 "--skip-server-setup", "--temperature", "0.001",
                 "--num-threads", "16"],
                capture_output=True, text=True, timeout=cat_timeout, env=env)
            print(f"  rc={r.returncode} stdout={r.stdout[-200:]}", flush=True)
            if r.returncode != 0:
                print(f"  stderr={r.stderr[-300:]}", flush=True)
            all_output.append(f"[gen {cat}] rc={r.returncode}\n{r.stdout[-500:]}\n{r.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {cat_timeout}s — skipping {cat} and continuing", flush=True)
            all_output.append(f"[gen {cat}] TIMEOUT after {cat_timeout}s")
        _commit_partial(f"after_gen_{cat}", all_output)

    for cat in categories:
        print(f"\n=== Evaluating: {cat} ===", flush=True)
        try:
            r = subprocess.run(
                ["bfcl", "evaluate", "--model", MODEL_NAME, "--test-category", cat],
                capture_output=True, text=True, timeout=900, env=env)
            print(f"  {r.stdout[-300:]}", flush=True)
            all_output.append(f"[eval {cat}]\n{r.stdout}")
        except subprocess.TimeoutExpired:
            print(f"  EVAL TIMEOUT for {cat} — skipping", flush=True)
            all_output.append(f"[eval {cat}] TIMEOUT")
        _commit_partial(f"after_eval_{cat}", all_output)

    print("\n=== SCORES ===")
    r = subprocess.run(["bfcl", "scores", "--model", MODEL_NAME],
                       capture_output=True, text=True, timeout=60, env=env)
    scores = r.stdout
    print(scores)

    with open("/results/bfcl_full_output.txt", "w") as f:
        f.write("\n".join(all_output) + "\n\nSCORES:\n" + scores)
    with open("/results/scores.txt", "w") as f:
        f.write(scores)
    results_vol.commit()

    vllm_proc.terminate()
    return scores or "\n".join(all_output[-3:])


@app.local_entrypoint()
def smoke():
    """3-category smoke test — quick pipeline validation (~10-15 min)."""
    cats = ["simple_python", "multiple", "parallel"]
    print(f"Launching SMOKE BFCL on A100-80GB: {cats}")
    output = run_bfcl_eval.remote(categories=cats, cat_timeout=1800)
    print("\n" + "="*60)
    print("BFCL SMOKE RESULTS")
    print("="*60)
    print(output)
    print("\nFetch with: modal run run_bfcl_official.py::fetch")


@app.local_entrypoint()
def main():
    print("Launching BFCL on A100-80GB (use `modal run --detach` to survive disconnect)...")
    output = run_bfcl_eval.remote()
    print("\n" + "="*60)
    print("BFCL OFFICIAL RESULTS")
    print("="*60)
    print(output)
    print("\nResults persisted to Volume 'bfcl-results'. To download:")
    print("  modal run run_bfcl_official.py::fetch")


@app.local_entrypoint()
def fetch():
    """Download results from the Modal Volume to local disk."""
    out_dir = "/Users/admin/prism/training"
    for fname in ("scores.txt", "bfcl_full_output.txt"):
        local_path = os.path.join(out_dir, f"bfcl_{fname}")
        try:
            with open(local_path, "wb") as f:
                for chunk in results_vol.read_file(fname):
                    f.write(chunk)
            print(f"Saved {local_path}")
        except FileNotFoundError:
            print(f"Not on volume yet: {fname}")
