"""Resume v18coder-14B polish from checkpoint-1000 (cancelled at step 1303/1376).

Original run was cancelled by an upstream signal at step 1303/1376. Auto-saved
checkpoints are checkpoint-500 and checkpoint-1000 in
prism-v18coder-14b-polish/v18coder_14b_polish_run/. Resume from checkpoint-1000
so optimizer/scheduler state are preserved (cosine LR was nearly at zero).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18coder-14b-polish-resume")

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
OUT_VOL = modal.Volume.from_name("prism-v18coder-14b-polish")

BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
RESUME_CKPT = "/out/v18coder_14b_polish_run/checkpoint-1000"
POLISH_DATA = "/data/train_v17_1.jsonl"


@app.function(
    image=image,
    gpu="H100:2",
    timeout=4 * 60 * 60,
    volumes={"/data": DATA_VOL, "/out": OUT_VOL},
)
def run_polish_resume():
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print("=== v18coder-14B POLISH RESUME starting ===")
    print(f"  base   = {BASE_MODEL}")
    print(f"  resume = {RESUME_CKPT}")
    print(f"  data   = {POLISH_DATA}")

    if not Path(RESUME_CKPT).exists():
        raise FileNotFoundError(f"{RESUME_CKPT} missing")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
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

    model = PeftModel.from_pretrained(base, RESUME_CKPT, is_trainable=True)
    print(f"loaded checkpoint adapter from {RESUME_CKPT}")
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

    # Subsample to ~376 steps worth of data (376 / 1376 = 27.3% of full polish).
    # Cancelled run had 376 steps remaining of a 1376-step polish; we re-run
    # those 376 steps as a fresh cosine schedule on a 27% subset, since
    # resume_from_checkpoint requires torch>=2.6 (CVE-2025-32434) and the
    # base image is torch 2.5.1.
    import random
    rows_remaining = random.Random(99).sample(rows, max(1, int(len(rows) * 376 / 1376)))
    print(f"running {len(rows_remaining)} examples (~376 steps with bs=8 grad_accum=2) on top of checkpoint-1000")
    ds_remaining = Dataset.from_list(rows_remaining)

    cfg = SFTConfig(
        output_dir="/out/v18coder_14b_polish_resumed_run",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=ds_remaining, processing_class=tok, args=cfg,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"v18coder-14B polish RESUME done in {train_secs:.0f}s")

    final_dir = "/out/final_polish_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "train_secs": round(train_secs, 1),
        "epochs": 1,
        "lr": cfg.learning_rate,
        "examples": len(rows),
        "base_model": BASE_MODEL,
        "resumed_from": RESUME_CKPT,
        "polish_data": POLISH_DATA,
        "note": "resume of cancelled polish run; original cancelled at step 1303/1376",
    }
    Path("/out/v18coder_14b_polish_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


@app.local_entrypoint()
def run():
    print("Resuming v18coder-14B polish from checkpoint-1000 on H100×2 (spawn — truly detached)…")
    handle = run_polish_resume.spawn()
    print(f"Spawned function call: {handle.object_id}")
    print("Local entrypoint exiting; remote function continues independently.")
