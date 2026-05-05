"""Modal A100 high-capacity training: SFT with rank=256 LoRA + DPO.

Strategy:
  - Start from Qwen/Qwen2.5-Coder-7B-Instruct (clean base, no entrenched priors)
  - Stage 1: SFT on 2288 normalized examples with rank=256 LoRA across all
    layers + all projections (huge capacity vs the rank-64 local DoRA)
  - Stage 2: DPO on 178 preference pairs to sharpen confused-pair decisions
  - Stage 3: Quick in-container eval on the 39 benchmark prompts
  - Save final adapter + benchmark report to Modal Volume

Local launch:
  modal run --detach modal_full_ft.py::main
Fetch results when done:
  modal run modal_full_ft.py::fetch
"""
import modal
import os
import json
from pathlib import Path

app = modal.App("prism-modal-ft")

image = (
    modal.Image.debian_slim(python_version="3.11")
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

vol = modal.Volume.from_name("prism-modal-ft", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/vol": vol},
)
def train_full_pipeline():
    """SFT (rank=256 LoRA) then DPO. Returns metrics dict."""
    import torch
    from datasets import load_dataset, Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DPOTrainer, DPOConfig, SFTConfig
    import re

    BASE_MODEL = "/vol/prism_v12_hf"  # dequantized prism-v12-fused — keeps v8→v12 priors
    OUT_DIR = "/vol/prism_modal_ft"
    SFT_DIR = f"{OUT_DIR}/sft"
    DPO_DIR = f"{OUT_DIR}/dpo"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[setup] loading tokenizer + model {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ───── Maximum-capacity LoRA config (highest-tier) ────────────
    lora_config = LoraConfig(
        r=512,
        lora_alpha=1024,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "down_proj", "up_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ───── STAGE 1: SFT ───────────────────────────────────────────
    print("[stage 1] SFT")
    train_data = []
    for line in open("/vol/data/train.jsonl"):
        train_data.append({"text": json.loads(line)["text"]})
    valid_data = []
    for line in open("/vol/data/valid.jsonl"):
        valid_data.append({"text": json.loads(line)["text"]})
    print(f"  train: {len(train_data)}  valid: {len(valid_data)}")

    sft_train_ds = Dataset.from_list(train_data)
    sft_valid_ds = Dataset.from_list(valid_data)

    sft_args = SFTConfig(
        output_dir=SFT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_seq_length=2048,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=1,
        dataset_text_field="text",
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_train_ds,
        eval_dataset=sft_valid_ds,
        processing_class=tok,
    )
    sft_trainer.train()
    sft_trainer.save_model(SFT_DIR)
    print(f"  [stage 1] saved → {SFT_DIR}")

    # ───── STAGE 2: DPO ───────────────────────────────────────────
    print("[stage 2] DPO")
    dpo_pairs = []
    for line in open("/vol/data/contrastive_dpo_seed.jsonl"):
        if not line.strip():
            continue
        ex = json.loads(line)
        # Normalize old <think>/<|tool_call_end|> tokens to canonical
        for k in ("chosen", "rejected"):
            ex[k] = (ex[k]
                     .replace("<think>", "<|synalux_think|>")
                     .replace("</think>", "</|synalux_think|>")
                     .replace("<|tool_call_end|>", "</|tool_call|>"))
        # TRL DPOTrainer expects: prompt, chosen, rejected
        prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        dpo_pairs.append({
            "prompt": prompt,
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        })
    print(f"  dpo pairs: {len(dpo_pairs)}")
    dpo_ds = Dataset.from_list(dpo_pairs)

    dpo_args = DPOConfig(
        output_dir=DPO_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,
        max_prompt_length=1024,
        beta=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=1,
    )
    dpo_trainer = DPOTrainer(
        model=sft_trainer.model,
        ref_model=None,  # uses base (peft auto-handles)
        args=dpo_args,
        train_dataset=dpo_ds,
        processing_class=tok,
    )
    dpo_trainer.train()
    dpo_trainer.save_model(DPO_DIR)
    print(f"  [stage 2] saved → {DPO_DIR}")

    # ───── STAGE 3: Quick in-container eval on 39 prompts ─────────
    print("[stage 3] benchmark")
    summary = _run_benchmark(dpo_trainer.model, tok)

    with open(f"{OUT_DIR}/benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    return summary


def _run_benchmark(final_model, tok):
    """Run the 39-prompt benchmark using TRAINING-FORMAT prompts.

    Match the format the model was trained on: explicit Prism system block
    + <|im_start|> tokens. Don't use apply_chat_template (Qwen default differs).
    """
    import torch, re, json
    test_prompts = json.load(open("/vol/data/benchmark_prompts.json"))
    SYSTEM = open("/vol/data/system_prompt.txt").read()
    final_model.eval()

    correct, total = 0, len(test_prompts)
    results = []
    for tc in test_prompts:
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{tc['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tok(prompt, return_tensors="pt").to(final_model.device)
        with torch.no_grad():
            out = final_model.generate(
                **inputs, max_new_tokens=200, temperature=0.001, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        m = re.search(r'"name":\s*"([^"]+)"', resp)
        called = m.group(1) if m else "NO_TOOL"
        expected = tc.get("expected_tool") or "NO_TOOL"
        ok = (called == expected)
        if ok:
            correct += 1
        results.append({
            "prompt": tc["prompt"][:60],
            "expected": expected,
            "called": called or "NO_TOOL",
            "ok": ok,
            "response_preview": resp[:200],
        })

    score = correct / total
    print(f"\n=== BENCHMARK: {correct}/{total} = {score:.1%} ===")
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} expected={r['expected']:30s} got={r['called']:30s} | {r['prompt']}")
    return {"tool_call_accuracy": score, "correct": correct,
            "total": total, "results": results}


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def eval_only():
    """Re-run benchmark on the existing saved DPO model (no retraining)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    BASE_MODEL = "/vol/prism_v12_hf"  # dequantized prism-v12-fused — keeps v8→v12 priors
    DPO_DIR = "/vol/prism_modal_ft/dpo"
    OUT_DIR = "/vol/prism_modal_ft"

    print(f"[eval_only] loading base + DPO adapter from {DPO_DIR}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, DPO_DIR)
    summary = _run_benchmark(model, tok)
    with open(f"{OUT_DIR}/benchmark.json", "w") as f:
        import json as _json
        _json.dump(summary, f, indent=2)
    vol.commit()
    return summary


def _upload_data():
    """Upload data + system prompt to Modal Volume."""
    import subprocess, sys
    data_dir = Path(__file__).parent / "data"
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark import TEST_CASES
    from normalize_and_merge import SYSTEM as _SYS

    # Write system prompt file locally then upload
    (data_dir / "system_prompt.txt").write_text(_SYS)

    bench_path = data_dir / "benchmark_prompts.json"
    bench_path.write_text(json.dumps(
        [{"prompt": tc["prompt"], "expected_tool": tc.get("expected_tool")}
         for tc in TEST_CASES], indent=2))

    files = ["train.jsonl", "valid.jsonl", "contrastive_dpo_seed.jsonl",
             "benchmark_prompts.json", "system_prompt.txt"]
    for f in files:
        src = data_dir / f
        subprocess.run(
            ["modal", "volume", "put", "prism-modal-ft",
             str(src), f"data/{f}", "--force"],
            check=True,
        )
        print(f"  uploaded {f}")


@app.local_entrypoint()
def main():
    """Upload data + launch full pipeline."""
    print("[upload] data → Volume")
    _upload_data()

    print("\n[train] launching A100-80GB training...")
    summary = train_full_pipeline.remote()
    print("\n=== FINAL RESULT ===")
    print(f"Accuracy: {summary['tool_call_accuracy']:.1%} ({summary['correct']}/{summary['total']})")


@app.local_entrypoint()
def run_eval():
    """Re-run benchmark on existing trained DPO model with correct format."""
    print("[upload] system_prompt + benchmark_prompts to Volume")
    _upload_data()
    print("\n[eval] launching A100-80GB...")
    summary = eval_only.remote()
    print(f"\n=== ACCURACY: {summary['tool_call_accuracy']:.1%} "
          f"({summary['correct']}/{summary['total']}) ===")


@app.local_entrypoint()
def fetch():
    """Download benchmark.json to local disk."""
    out_path = Path(__file__).parent / "results" / "benchmark_modal_ft.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in vol.read_file("prism_modal_ft/benchmark.json"):
            f.write(chunk)
    print(f"Saved {out_path}")
    print(json.dumps(json.load(open(out_path)), indent=2)[:2000])
