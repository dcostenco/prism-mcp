"""Merge a PEFT LoRA adapter into its base model and save the full HF model.

Used as the fusion step in the v16.3 autonomous pipeline because mlx_lm.fuse
can't ingest PEFT-format adapters (its loader expects MLX-native shape).
"""
import os
import sys
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    base = sys.argv[1]
    adapter = sys.argv[2]
    out = sys.argv[3]

    print(f"[merge] base    = {base}")
    print(f"[merge] adapter = {adapter}")
    print(f"[merge] out     = {out}")

    t0 = time.time()
    print("[merge] loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    print("[merge] loading base model (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"[merge] base loaded in {time.time()-t0:.0f}s", flush=True)

    print("[merge] applying LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    print("[merge] merging weights into base...", flush=True)
    model = model.merge_and_unload()
    print(f"[merge] merge complete in {time.time()-t0:.0f}s total", flush=True)

    os.makedirs(out, exist_ok=True)
    print(f"[merge] saving to {out}...", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[merge] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: v163_peft_merge.py <base_model_path_or_hf_id> <adapter_dir> <out_dir>")
    main()
