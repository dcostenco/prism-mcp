"""v18coder-32B full SFT on Synalux-anchored data — Phase 1 of the
32B/72B campaign for Synalux 10K-user scale.

Composition (~44K rows, ~25M tokens):
  - 4,517 v18aac (caregiver + text_correct + emergency)
  - 1,191 aac_micro (translate + ask_ai synthesized)
  - 3,280 synalux Phase 0 (Prism Memory anonymized — the asset)
  - 30,000 v18coder BFCL backbone subsample
  - 5,000 v18coder generalization slice

Base: Qwen/Qwen2.5-Coder-32B-Instruct
GPU:  H100×4 (320 GB VRAM)
LoRA: DoRA r=128, alpha=256 (half of qwen3-8b's r=256 — 32B has more capacity)
Run:  1 epoch, LR 1e-5 cosine, save_steps 500, .spawn() truly detached.

Estimated cost: ~$250-350 (10-12h × 4 H100s × ~$4/hr).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18coder-32b-synalux")

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
OUT_VOL = modal.Volume.from_name("prism-v18coder-32b-synalux", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
TRAIN_DATA = "/data/train_v18coder_32b.jsonl"


@app.function(
    image=image,
    gpu="H100:4",
    timeout=14 * 60 * 60,
    volumes={"/data": DATA_VOL, "/out": OUT_VOL},
)
def run_full_sft():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTTrainer, SFTConfig

    print(f"=== v18coder-32B Synalux SFT (DoRA r=128) ===")
    print(f"  base = {BASE_MODEL}")
    print(f"  data = {TRAIN_DATA}")

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

    lora_cfg = LoraConfig(
        r=128, lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", use_dora=True,
    )
    model = get_peft_model(base, lora_cfg)
    print("DoRA adapter attached:")
    model.print_trainable_parameters()

    rows = []
    with open(TRAIN_DATA) as f:
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
    print(f"training examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    class EpochCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            n = int(state.epoch)
            ckpt = f"/out/epoch_{n}_adapter"
            print(f"\n=== Saving epoch {n} → {ckpt} ===")
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            meta = {
                "epoch": n,
                "global_step": state.global_step,
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
            }
            Path(f"/out/epoch_{n}_meta.json").write_text(json.dumps(meta, indent=2))
            OUT_VOL.commit()
            return control

    cfg = SFTConfig(
        output_dir="/out/v18coder_32b_synalux_run",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1.0e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=ds, processing_class=tok,
        args=cfg, callbacks=[EpochCheckpointCallback()],
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"v18coder-32B Synalux SFT done in {train_secs:.0f}s")

    final_dir = "/out/final_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "train_secs": round(train_secs, 1), "epochs": 1,
        "lr": cfg.learning_rate, "examples": len(ds),
        "lora_rank": 128, "lora_alpha": 256, "use_dora": True,
        "base_model": BASE_MODEL,
        "data_composition": (
            "4,517 v18aac + 1,191 aac_micro + 3,280 synalux phase0 + "
            "30,000 v18coder bfcl + 5,000 v18coder gen"
        ),
        "campaign": "Phase 1 of 32B/72B Synalux",
    }
    Path("/out/v18coder_32b_synalux_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


@app.local_entrypoint()
def run():
    print("Launching v18coder-32B Synalux SFT (H100×4, .spawn() detached)…")
    handle = run_full_sft.spawn()
    print(f"Spawned function call: {handle.object_id}")
    print("Local entrypoint exiting; remote function continues independently.")
