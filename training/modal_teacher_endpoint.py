"""Modal-hosted vLLM teacher endpoint — exposes port via web_server.

Deploy:
  modal deploy modal_teacher_endpoint.py
URL: https://dcostenco--prism-teacher-serve.modal.run
"""
import modal
import os
import subprocess

app = modal.App("prism-teacher")

_image_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # vLLM 0.8.5 + 0.10.0 both crash on:
        #   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
        # The attribute was removed in transformers >= 4.50. Pin to 4.45.x
        # which still exposes it. This is the LAST transformers release
        # that vLLM's tokenizer cache logic works against unmodified.
        "vllm==0.6.6.post1",
        "transformers==4.45.2",
        "huggingface_hub",
    )
)
_hf_token = os.environ.get("HF_TOKEN")
image = _image_base.env({"HF_TOKEN": _hf_token}) if _hf_token else _image_base

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"  # vLLM 0.8.5-compatible, single H100, high quality
PORT = 8000


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=PORT, startup_timeout=1500)
def serve():
    """Spawn vLLM. web_server proxies port 8000 to public URL."""
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
    ]
    subprocess.Popen(cmd)
