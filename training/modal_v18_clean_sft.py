"""v18-clean: full SFT from clean Qwen base for BFCL push.

Goal: BFCL ≥95% AND all AAC gates pass — single model, single train.

Strategy: NO continuation from v17.x adapter. Train clean Qwen2.5-Coder-7B-
Instruct with DoRA rank 256 on balanced dataset for 3 epochs.

CHECKPOINT SAFETY: save adapter at each epoch (epoch_1, epoch_2, epoch_3) plus
final. Each checkpoint has its own meta.json with smoke results so we can
deploy any intermediate checkpoint as fallback if final regresses.

Volumes:
  prism-sft-data    input  — /train_v18_clean.jsonl
  prism-v18-clean   output — /epoch_{1,2,3}_adapter, /final_adapter, /meta.json

Run:
  modal run --detach modal_v18_clean_sft.py::run

Fetch any checkpoint:
  modal volume get prism-v18-clean epoch_1_adapter ./epoch_1_adapter --force
  modal volume get prism-v18-clean epoch_2_adapter ./epoch_2_adapter --force
  modal volume get prism-v18-clean epoch_3_adapter ./epoch_3_adapter --force
  modal volume get prism-v18-clean final_adapter ./final_adapter --force
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v18-clean-sft")

import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "trl==0.12.2",
        "peft==0.13.2",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub",
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)

DATA_VOL = modal.Volume.from_name("prism-sft-data")
OUT_VOL = modal.Volume.from_name("prism-v18-clean", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,  # full SFT can run longer
    volumes={
        "/data": DATA_VOL,
        "/out": OUT_VOL,
    },
)
def run_full_sft():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTTrainer, SFTConfig

    print(f"=== v18-clean FULL SFT starting ===")
    print(f"  base = {BASE_MODEL}  (CLEAN — no v17 adapter)")

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

    # DoRA configuration — 7 projections, rank 256 (matches v17 line capacity)
    lora_cfg = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )
    model = get_peft_model(base, lora_cfg)
    print("DoRA adapter attached:")
    model.print_trainable_parameters()

    data_path = Path("/data/train_v18_clean.jsonl")
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
            # Quick smoke at this checkpoint
            smoke = _smoke_format_check(model, tok)
            meta = {
                "epoch": epoch_num,
                "global_step": state.global_step,
                "train_loss": state.log_history[-1].get("loss") if state.log_history else None,
                "smoke": smoke,
            }
            Path(f"/out/epoch_{epoch_num}_meta.json").write_text(json.dumps(meta, indent=2))
            OUT_VOL.commit()
            print(f"epoch {epoch_num} smoke: {smoke}")
            return control

    # Full SFT config — 3 epochs, higher LR than surgical
    cfg = SFTConfig(
        output_dir="/out/v18_clean_run",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,        # eff batch 8
        gradient_checkpointing=True,
        learning_rate=1.5e-5,                  # full SFT LR (10x higher than surgical)
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tok,
        args=cfg,
        callbacks=[EpochCheckpointCallback()],
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"full SFT done in {train_secs:.0f}s")

    # Save FINAL adapter
    final_dir = "/out/final_adapter"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"final adapter saved to {final_dir}")

    smoke = _smoke_format_check(model, tok)
    print(f"final smoke: {smoke}")

    meta = {
        "train_secs": round(train_secs, 1),
        "epochs": 3,
        "lr": cfg.learning_rate,
        "examples": len(rows),
        "lora_rank": 256,
        "use_dora": True,
        "smoke_format": smoke,
        "checkpoints": ["/out/epoch_1_adapter", "/out/epoch_2_adapter", "/out/epoch_3_adapter", "/out/final_adapter"],
    }
    Path("/out/v18_clean_meta.json").write_text(json.dumps(meta, indent=2))
    OUT_VOL.commit()
    return meta


_SMOKE_PROBES = [
    {
        "tools": [{"type": "function", "function": {"name": "get_weather",
                  "description": "Get weather", "parameters": {"type": "object",
                  "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "What's the weather in Paris?",
        "expect_call": "get_weather",
    },
    {
        "tools": [{"type": "function", "function": {"name": "search_web",
                  "description": "Search", "parameters": {"type": "object",
                  "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
        "user": "Tell me a joke.",
        "expect_call": None,
    },
    {
        "tools": None,
        "system": (
            "You are a fast text-cleanup engine for an AAC app. "
            "Fix typos and word boundaries. Do NOT add or remove words. "
            "Return ONLY the corrected text."
        ),
        "user": 'Language: en. Input: "no..stop..hurts"',
        "expect_pattern": "stop",
    },
    {
        "tools": None,
        "system": (
            "You are an emergency-response AI on a phone call. The patient is "
            "Alex, 8, nonverbal, uses AAC, cannot breathe. Be concise."
        ),
        "user": "Is the patient verbal?",
        "expect_pattern": "nonverbal",
    },
]


def _smoke_format_check(model, tok) -> dict:
    import torch
    out = {}
    for i, case in enumerate(_SMOKE_PROBES):
        if case.get("tools"):
            sys_msg = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                "# Tools\n\nYou may call one or more functions.\n\n"
                f"<tools>\n{json.dumps(case['tools'], ensure_ascii=False)}\n</tools>\n\n"
                "Return tool calls within <tool_call></tool_call> XML tags."
            )
        else:
            sys_msg = case["system"]

        prompt = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{case['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(
                **inputs, max_new_tokens=120, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        resp = tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

        if "expect_pattern" in case:
            ok = case["expect_pattern"] in resp
            out[f"probe_{i}"] = {"ok": ok, "expected": case["expect_pattern"], "resp": resp[:200]}
        elif case["expect_call"] is None:
            has_wrapper = "<tool_call>" in resp and "</tool_call>" in resp
            ok = not has_wrapper
            out[f"probe_{i}"] = {"ok": ok, "kind": "abstain", "resp": resp[:200]}
        else:
            has_wrapper = "<tool_call>" in resp and "</tool_call>" in resp
            ok = has_wrapper and case["expect_call"] in resp
            out[f"probe_{i}"] = {"ok": ok, "kind": "call", "resp": resp[:200]}
    return out


@app.local_entrypoint()
def run():
    print("Launching v18-clean full SFT on H100…")
    meta = run_full_sft.remote()
    print("\n" + "=" * 60)
    print("v18-clean FULL SFT COMPLETE")
    print("=" * 60)
    print(json.dumps(meta, indent=2))
    print("\nFetch checkpoints:")
    for ckpt in ["epoch_1_adapter", "epoch_2_adapter", "epoch_3_adapter", "final_adapter"]:
        print(f"  modal volume get prism-v18-clean {ckpt} ./{ckpt} --force")
