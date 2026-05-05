"""v18coder-Qwen3-8B polish pass: continue from base 8B SFT on curated 11K.

A/B partner of modal_v18coder_14b_polish_sft.py — same recipe, Qwen3-8B base.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18coder-qwen3-8b-polish")

import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.55.0",
        "trl==0.18.0",
        "peft==0.15.0",
        "datasets==3.6.0",
        "accelerate==1.7.0",
        "bitsandbytes==0.45.3",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)

DATA_VOL = modal.Volume.from_name("prism-sft-data")
BASE_RUN_VOL = modal.Volume.from_name("prism-v18coder-qwen3-8b")
OUT_VOL = modal.Volume.from_name("prism-v18coder-qwen3-8b-polish", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/base_run/final_adapter"
POLISH_DATA = "/data/train_v17_1.jsonl"


@app.function(
    image=image,
    gpu="H100:2",
    timeout=4 * 60 * 60,
    volumes={"/data": DATA_VOL, "/base_run": BASE_RUN_VOL, "/out": OUT_VOL},
)
def run_polish():
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print("=== v18coder-Qwen3-8B POLISH starting ===")
    print(f"  base   = {BASE_MODEL}")
    print(f"  adapter= {ADAPTER_PATH}")
    print(f"  data   = {POLISH_DATA}")

    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"{ADAPTER_PATH} missing — base Qwen3-8B SFT must finish first"
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=True)
    print("loaded base-pass adapter; continuing training")
    model.print_trainable_parameters()

    rows = []
    with open(POLISH_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj.get("text"), str) and len(obj["text"]) >= 50:
                    rows.append({"text": obj["text"]})
            except Exception:
                continue
    print(f"polish examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    cfg = SFTConfig(
        output_dir="/out/v18coder_qwen3_8b_polish_run",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=ds, processing_class=tok, args=cfg,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"v18coder-Qwen3-8B polish done in {train_secs:.0f}s")

    final_dir = "/out/final_polish_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "train_secs": round(train_secs, 1),
        "epochs": 1,
        "lr": cfg.learning_rate,
        "examples": len(rows),
        "base_model": BASE_MODEL,
        "input_adapter": ADAPTER_PATH,
        "polish_data": POLISH_DATA,
    }
    Path("/out/v18coder_qwen3_8b_polish_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


@app.local_entrypoint()
def run():
    print("Launching v18coder-Qwen3-8B polish (curated 11K, LR 5e-6) on H100×2…")
    meta = run_polish.remote()
    print("\n" + "=" * 60)
    print("v18coder-Qwen3-8B POLISH COMPLETE")
    print("=" * 60)
    print(json.dumps(meta, indent=2))
