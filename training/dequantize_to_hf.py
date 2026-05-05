"""De-quantize an MLX 4-bit model to HF-compatible bf16 safetensors.

Usage:
  python dequantize_to_hf.py models/prism-v12-fused /tmp/v12_hf
"""
import sys, json, shutil
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, dequantize_model

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

print(f"Loading {src}...")
model, tokenizer = load(str(src))

print("Dequantizing...")
model = dequantize_model(model)

# Save weights as bf16
print(f"Saving bf16 weights to {dst}...")
weights = dict(model.parameters())
flat = {}
def _flatten(prefix, obj):
    if isinstance(obj, mx.array):
        flat[prefix] = obj.astype(mx.bfloat16)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v)
_flatten("", weights)
print(f"  {len(flat)} tensors")
mx.save_safetensors(str(dst / "model.safetensors"), flat)

# Copy/patch config (strip quantization)
cfg = json.loads((src / "config.json").read_text())
cfg.pop("quantization", None)
cfg.pop("quantization_config", None)
(dst / "config.json").write_text(json.dumps(cfg, indent=2))

# Copy tokenizer + chat template
for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
           "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
    s = src / fn
    if s.exists():
        shutil.copy(s, dst / fn)
        print(f"  copied {fn}")

print(f"\n✅ Done. HF model at {dst}")
print(f"   Size: {sum(p.stat().st_size for p in dst.iterdir())/1e9:.1f} GB")
