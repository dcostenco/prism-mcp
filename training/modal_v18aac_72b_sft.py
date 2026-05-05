"""v18coder: BFCL-optimized full SFT from Qwen3-8B base.

Goal: Berkeley BFCL V4 ≥ 45% Overall, ≥ 85% Non-Live AST, ≥ 90% Irrelevance.
Approach: Hammer-style function-masking SFT on commercial-safe data.

Composition: 189,710 rows
  - glaive-function-calling-v2 (112,960, Apache 2.0)
  - ToolACE (11,300, Apache 2.0)
  - xlam-function-calling-60k (60,000, CC-BY-4.0)
  - internal v17_1_bfcl (5,450, internal)
  - 24% function-masked (Hammer recipe)

Base: Qwen3-8B (best 7-9B base on BFCL V4: 42.57% Overall pre-SFT).
Target: significantly exceed Qwen3-8B's pre-SFT baseline by adding our SFT.

Per-epoch checkpoints saved automatically.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18aac-72b-sft")

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
OUT_VOL = modal.Volume.from_name("prism-v18aac-72b", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-72B-Instruct"


@app.function(
    image=image,
    gpu="H100:4",
    timeout=10 * 60 * 60,  # large dataset, allow 10h
    volumes={"/data": DATA_VOL, "/out": OUT_VOL},
)
def run_full_sft():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTTrainer, SFTConfig

    print(f"=== v18aac-72B FULL SFT (max precision r=384) starting ===")
    print(f"  base = {BASE_MODEL}  (Qwen2.5-Coder-72B-Instruct (proven base for AAC))")

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

    # DoRA configuration — 7 projections, rank 256
    lora_cfg = LoraConfig(
        r=384, lora_alpha=768,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", use_dora=True,
    )
    model = get_peft_model(base, lora_cfg)
    print("DoRA adapter attached:")
    model.print_trainable_parameters()

    data_path = Path("/data/train_v18aac.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing")
    rows = []
    with data_path.open() as f:
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

    # Per-epoch checkpoint callback
    class EpochCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_num = int(state.epoch)
            ckpt_dir = f"/out/epoch_{epoch_num}_adapter"
            print(f"\n=== Saving epoch {epoch_num} checkpoint to {ckpt_dir} ===")
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            meta = {
                "epoch": epoch_num,
                "global_step": state.global_step,
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
            }
            Path(f"/out/epoch_{epoch_num}_meta.json").write_text(json.dumps(meta, indent=2))
            OUT_VOL.commit()
            return control

    cfg = SFTConfig(
        output_dir="/out/v18aac_72b_run",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
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
    print(f"v18aac-72B full SFT done in {train_secs:.0f}s")

    final_dir = "/out/final_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "train_secs": round(train_secs, 1), "epochs": 3,
        "lr": cfg.learning_rate, "examples": len(rows),
        "lora_rank": 256, "use_dora": True,
        "base_model": BASE_MODEL,
        "data_composition": "glaive-v2 + ToolACE + xlam-60k + internal_bfcl, Hammer-style 24% function-masked",
        "checkpoints": ["/out/epoch_1_adapter", "/out/epoch_2_adapter", "/out/epoch_3_adapter", "/out/final_adapter"],
    }
    Path("/out/v18aac_72b_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


@app.local_entrypoint()
def run():
    print("Launching v18aac-72B (r=384) full SFT on H100…")
    meta = run_full_sft.remote()
    print("\n" + "=" * 60)
    print("v18aac-72B FULL SFT COMPLETE")
    print("=" * 60)
    print(json.dumps(meta, indent=2))
