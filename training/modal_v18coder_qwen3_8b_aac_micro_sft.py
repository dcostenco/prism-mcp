"""qwen3-8b AAC micro-SFT — surgical recovery of AAC behaviors.

Loads qwen3-8b/final_adapter (the BFCL-trained adapter from prism-v18coder-qwen3-8b)
and continues training on a balanced 1191-row AAC dataset at LR 1e-6 for 1 epoch.

Goal: restore AAC performance (caregiver, text_correct, emergency, translate,
ask_ai) without destroying BFCL — the prior polish on 11K BFCL rows over-fit
the model to tool-call patterns and erased AAC. This pass goes the other way:
small AAC dataset, very low LR, surgical.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18coder-qwen3-8b-aac-micro")

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
OUT_VOL = modal.Volume.from_name("prism-v18coder-qwen3-8b-aac-micro", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/base_run/final_adapter"
AAC_DATA = "/data/train_aac_micro.jsonl"


@app.function(
    image=image,
    gpu="H100:2",
    timeout=4 * 60 * 60,
    volumes={"/data": DATA_VOL, "/base_run": BASE_RUN_VOL, "/out": OUT_VOL},
)
def run_aac_micro():
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print("=== qwen3-8b AAC micro-SFT starting ===")
    print(f"  base    = {BASE_MODEL}")
    print(f"  adapter = {ADAPTER_PATH} (BFCL-trained, will be polished with AAC)")
    print(f"  data    = {AAC_DATA}")
    print(f"  LR      = 1e-6 (5x lower than failed BFCL polish)")

    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(f"{ADAPTER_PATH} missing")
    if not Path(AAC_DATA).exists():
        raise FileNotFoundError(f"{AAC_DATA} missing — run build_aac_micro.py first and upload to prism-sft-data")

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
    print("loaded BFCL adapter; continuing training on AAC micro data")
    model.print_trainable_parameters()

    rows = []
    with open(AAC_DATA) as f:
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
    print(f"AAC micro examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    cfg = SFTConfig(
        output_dir="/out/v18coder_qwen3_8b_aac_micro_run",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=200,
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
    print(f"qwen3-8b AAC micro-SFT done in {train_secs:.0f}s")

    final_dir = "/out/final_aac_micro_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "train_secs": round(train_secs, 1),
        "epochs": 1,
        "lr": cfg.learning_rate,
        "examples": len(rows),
        "base_model": BASE_MODEL,
        "input_adapter": ADAPTER_PATH,
        "data": AAC_DATA,
        "composition": "300 caregiver + 294 text_correct + 496 emergency + 51 translate + 50 ask_ai",
        "rationale": "surgical AAC recovery on top of BFCL adapter; LR 1e-6 to preserve BFCL",
    }
    Path("/out/v18coder_qwen3_8b_aac_micro_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


@app.local_entrypoint()
def run():
    print("Launching qwen3-8b AAC micro-SFT (1191 rows, LR 1e-6, 1 epoch) on H100×2 (spawn — detached)…")
    handle = run_aac_micro.spawn()
    print(f"Spawned function call: {handle.object_id}")
    print("Local entrypoint exiting; remote function continues independently.")
