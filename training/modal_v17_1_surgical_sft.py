"""Surgical SFT pass on top of v17 adapter — restore BFCL while keeping AAC.

Targets v17 regressions vs v12 (BFCL 100% → 71.2%):
  - disambiguation 37.5% → ≥85%   (distractor + multi-turn examples)
  - format_sensitivity 40% → ≥90%  (canonical <tool_call> in 7K rows)
  - ast_parameter 60% → ≥85%       (ast_hardening + optional_restraint)
  - hallucination 80% → ≥95%       (irrelevance + miss_func + borderline_adversarial)
  - simple/edge_case → hold/improve

AAC fixes:
  - caregiver 57.1% → ≥85%         (boost_word patterns + reorder_phrase + existing extra)
  - all other AAC tasks held within 2pp via 500 v17 regression-prevention rows

Approach: load v17 adapter, train on 11K canonical-format rows at LR 5e-6,
DoRA already in v17 (kept). 2 epochs.

Volumes:
  prism-v17        input  — /v17_adapter (LoRA from v17 SFT)
  prism-sft-data   input  — /train_v17_1.jsonl (this pass's combined data)
  prism-v17-1      output — /v17_1_adapter + /v17_1_meta.json

Run:
  modal run --detach modal_v17_1_surgical_sft.py::run

Fetch adapter:
  modal volume get prism-v17-1 v17_1_adapter ./v17_1_adapter --force
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("prism-v17-1-surgical")

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
V17_VOL = modal.Volume.from_name("prism-v17")
OUT_VOL = modal.Volume.from_name("prism-v17-1", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={
        "/data": DATA_VOL,
        "/v17": V17_VOL,
        "/out": OUT_VOL,
    },
)
def run_surgical_sft():
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print(f"=== v17.1 surgical SFT starting ===")
    print(f"  base model = {BASE_MODEL}")
    print(f"  prior adapter = /v17/v17_adapter")

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
    print(f"loaded base; params={base.num_parameters():_}")

    # Load v17 adapter on top — continue training the same LoRA weights
    model = PeftModel.from_pretrained(base, "/v17/v17_adapter", is_trainable=True)
    print("loaded v17 adapter")
    model.print_trainable_parameters()

    # Load merged surgical training data
    data_path = Path("/data/train_v17_1.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} missing — upload first:\n"
            "  modal volume put prism-sft-data data/train_v17_1.jsonl /train_v17_1.jsonl --force"
        )
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
    print(f"surgical examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    # 11K rows × 2 epochs at eff_batch=8 → ~2750 steps. Cap at 3000.
    cfg = SFTConfig(
        output_dir="/out/v17_1_run",
        num_train_epochs=2,
        max_steps=3000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
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
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"surgical train done in {train_secs:.0f}s")

    out_dir = "/out/v17_1_adapter"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"adapter saved to {out_dir}")

    smoke_results = _smoke_format_check(model, tok)
    print(f"format smoke: {smoke_results}")

    meta = {
        "train_secs": round(train_secs, 1),
        "steps": cfg.max_steps,
        "lr": cfg.learning_rate,
        "examples": len(rows),
        "smoke_format": smoke_results,
        "base_adapter": "v17",
    }
    Path("/out/v17_1_meta.json").write_text(json.dumps(meta, indent=2))
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
        # AAC caregiver smoke — bare JSON array, NOT wrapped in <tool_call>
        "tools": None,
        "system": (
            "You are an AAC app configuration assistant for a BCBA/caregiver.\n"
            "Available categories: help, food, feelings, school, quick\n"
            "Parse the caregiver instruction. Return ONLY a JSON array. No explanation."
        ),
        "user": "Caregiver says: \"He's using 'because' a lot now\"",
        "expect_pattern": "boost_word",
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
            out[f"probe_{i}_aac"] = {"ok": ok, "expected": case["expect_pattern"], "resp": resp[:200]}
        elif case["expect_call"] is None:
            has_wrapper = "<tool_call>" in resp and "</tool_call>" in resp
            ok = not has_wrapper
            out[f"probe_{i}_abstain"] = {"ok": ok, "has_wrapper": has_wrapper, "resp": resp[:200]}
        else:
            has_wrapper = "<tool_call>" in resp and "</tool_call>" in resp
            ok = has_wrapper and case["expect_call"] in resp
            out[f"probe_{i}_call"] = {"ok": ok, "has_wrapper": has_wrapper, "resp": resp[:200]}
    return out


@app.local_entrypoint()
def run():
    print("Launching v17.1 surgical SFT on H100…")
    meta = run_surgical_sft.remote()
    print("\n" + "=" * 60)
    print("v17.1 SURGICAL SFT COMPLETE")
    print("=" * 60)
    print(json.dumps(meta, indent=2))
    print("\nFetch adapter:")
    print("  modal volume get prism-v17-1 v17_1_adapter ./v17_1_adapter --force")
