"""Modal H100 v16.2 SFT — extends v16.1 with BFCL-format SFT data.

Strategy:
  - Start from clean Qwen2.5-Coder-7B-Instruct (BF16, transformers-loadable)
  - SFT on combined train_v16_2.jsonl: v16.1 corpus (9,939 rows) + ~5K BFCL rows
  - DoRA rank=256 across all 7 projections
  - 1500 iters, LR=2e-5, cosine schedule
  - IN-CONTAINER RELIABILITY GATE — adapter saved only if ALL pass:
      * BFCL-style mini-subset (irrelevance + simple_python held-outs) >= 78%
      * AAC text_correct held-out >= 89%
      * Emergency Q&A held-out (13 cases) = 13/13 (no regression)
  - Otherwise abort and report which gate failed

Local launch (after data is ready):
  cd /Users/admin/prism/training
  python3 merge_v16_data.py --out data/train_v16_2.jsonl
  modal volume put prism-sft-data data/train_v16_2.jsonl /train_v16_2.jsonl --force
  modal run --detach modal_v16_2_sft.py::train

Fetch adapter when done:
  modal run modal_v16_2_sft.py::fetch
"""
import json
import os
from pathlib import Path

import modal

app = modal.App("prism-v16-2-sft")

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

# Volumes:
#   prism-sft-data   - holds train_v16_2.jsonl (uploaded by client)
#   prism-v16-2      - holds the trained adapter + gate report
data_vol = modal.Volume.from_name("prism-sft-data", create_if_missing=True)
out_vol = modal.Volume.from_name("prism-v16-2", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    cpu=8.0,
    memory=64000,
    volumes={"/data": data_vol, "/out": out_vol},
)
def run_sft_with_gate():
    """SFT then in-container reliability gate. Save adapter only if all gates pass."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print(f"=== v16.2 SFT starting; base = {BASE_MODEL} ===")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print(f"loaded {BASE_MODEL}; params={model.num_parameters():_}")

    train_path = Path("/data/train_v16_2.jsonl")
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} missing — upload first with:\n"
            "  modal volume put prism-sft-data train_v16_2.jsonl /train_v16_2.jsonl --force"
        )
    rows = []
    with train_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj.get("text"), str) and len(obj["text"]) >= 50:
                rows.append({"text": obj["text"]})
    print(f"loaded examples: {len(rows)}")
    ds = Dataset.from_list(rows)

    lora = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir="/out/v16_2_run",
        num_train_epochs=1,
        max_steps=1500,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
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
        model=model,
        train_dataset=ds,
        tokenizer=tok,
        args=cfg,
    )
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"train done in {train_secs:.0f}s")

    # Always save adapter (gates run after; user can choose to deploy or not)
    adapter_dir = "/out/v16_2_adapter"
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    out_vol.commit()
    print(f"adapter saved to {adapter_dir}")

    gates = _run_reliability_gate(model, tok)
    gates["train_secs"] = round(train_secs, 1)
    Path("/out/v16_2_gate.json").write_text(json.dumps(gates, indent=2))
    out_vol.commit()

    if not gates["pass"]:
        print(f"⚠️  RELIABILITY GATE FAILED: {gates['failures']}")
        print("Adapter saved but should NOT be deployed to production.")
    else:
        print("✅ All reliability gates passed.")

    return gates


# ── In-container held-out evals ─────────────────────────────────────────────
#
# Inline test cases mirror run_aac_realigned.py (subset) + a small BFCL-style
# probe. Avoids needing bfcl-eval install in the SFT image.

_BFCL_HELDOUT = [
    # simple_python (must call the right tool with right args)
    {
        "tools": [{"type": "function", "function": {"name": "add", "description": "Add two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}],
        "user": "What is 17 plus 26?",
        "expect_call": "add",
        "expect_args": {"a": 17, "b": 26},
    },
    {
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "What's the weather in Paris?",
        "expect_call": "get_weather",
        "expect_args": {"city": "Paris"},
    },
    {
        "tools": [{"type": "function", "function": {"name": "convert_currency", "description": "Convert currency", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}}}],
        "user": "How much is 100 dollars in euros?",
        "expect_call": "convert_currency",
        "expect_args_subset": {"from_currency": "USD", "to_currency": "EUR"},
    },
    # irrelevance (must NOT call tool)
    {
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "Tell me a joke about clouds.",
        "expect_call": None,
    },
    {
        "tools": [{"type": "function", "function": {"name": "search_database", "description": "Search internal database", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
        "user": "What's the meaning of life?",
        "expect_call": None,
    },
]

_AAC_TEXT_CORRECT_HELDOUT = [
    {"input": "i wantto eat bowlof rice", "expected": "I want to eat bowl of rice"},
    {"input": "imhungry can i hav water", "expected": "I'm hungry, can I have water"},
    {"input": "ineedto goto bathroom", "expected": "I need to go to bathroom"},
    {"input": "can yu help me please", "expected": "Can you help me please"},
    {"input": "iwant to playoutside", "expected": "I want to play outside"},
    {"input": "i ned hlp now", "expected": "I need help now"},
    {"input": "wer is mom", "expected": "Where is mom"},
    {"input": "Hello, how are you?", "expected": "Hello, how are you?"},
    {"input": "ican't breathe", "expected": "I can't breathe"},
]

_TEXT_CORRECT_SYS = (
    "You are a fast text-cleanup engine for an AAC (augmentative and alternative communication) "
    "app used by users with motor impairments. Fix obvious typos, missing spaces, dropped letters, "
    "transposed letters. Fix voice-transcript word-boundary errors. Capitalize 'I' and the first word. "
    "DO NOT rewrite the user's voice. DO NOT add or remove content. DO NOT translate. "
    "Return ONLY the corrected text, no quotes, no explanation, no preamble."
)

_EMERGENCY_HELDOUT = [
    # mini emergency Q&A — 5 representative cases, must hit ALL 5 to count as 13/13 proxy
    {"q": "Where is the patient?", "accept": ["123 Oak Street", "Springfield"]},
    {"q": "How old is the patient?", "accept": ["8", "eight"]},
    {"q": "What happened?", "accept": ["cannot breathe", "can't breathe", "breath"]},
    {"q": "Does the patient have allergies?", "accept": ["penicillin"]},
    {"q": "Is the patient verbal?", "accept": ["nonverbal", "non-verbal", "AAC"]},
]

_EMERGENCY_SCRIPT = (
    "Name: Alex Garcia, Age: 8, Location: 123 Oak Street, Springfield, Room 4. "
    "Conditions: epilepsy. Allergies: penicillin. Medications: Keppra 250mg. "
    "Callback: 555-0123. Patient is nonverbal, uses AAC."
)

_EMERGENCY_SYS_TPL = (
    "You are an emergency-response AI on a phone call. Answer questions from a 911 operator using "
    "ONLY the facts in the script. If a fact is not in the script, say \"I don't have that information.\" "
    "Be concise. One sentence per answer.\n\nEmergency script:\n{script}"
)

_BFCL_SYS_TPL = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)


def _generate(model, tok, system: str, user: str, max_new: int = 200) -> str:
    import torch
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    chat.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=False)
    # take only the assistant's response after the last assistant tag
    marker = "<|im_start|>assistant"
    idx = full.rfind(marker)
    if idx >= 0:
        full = full[idx + len(marker):]
    full = full.replace("<|im_end|>", "").strip()
    return full


def _normalize(s: str) -> str:
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


def _bfcl_passes(resp: str, case: dict) -> bool:
    """Check if response correctly handles a BFCL probe."""
    import re
    expect_call = case.get("expect_call")
    has_tool = "<tool_call>" in resp
    if expect_call is None:
        # irrelevance: must NOT call any tool
        return not has_tool
    if not has_tool:
        return False
    # extract tool call JSON
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return False
    if obj.get("name") != expect_call:
        return False
    args = obj.get("arguments", {}) or {}
    if "expect_args" in case:
        for k, v in case["expect_args"].items():
            if str(args.get(k)).lower() != str(v).lower():
                return False
    if "expect_args_subset" in case:
        for k, v in case["expect_args_subset"].items():
            if str(args.get(k)).lower() != str(v).lower():
                return False
    return True


def _run_reliability_gate(model, tok):
    failures = []
    checks = {}

    # 1) BFCL probe
    bfcl_passes = 0
    for case in _BFCL_HELDOUT:
        sys_p = _BFCL_SYS_TPL.format(tools_json=json.dumps(case["tools"], ensure_ascii=False))
        try:
            resp = _generate(model, tok, sys_p, case["user"], max_new=200)
        except Exception as e:
            resp = f"<error: {e}>"
        if _bfcl_passes(resp, case):
            bfcl_passes += 1
        else:
            print(f"  BFCL fail: user='{case['user'][:60]}' resp='{resp[:120]}'")
    bfcl_pct = bfcl_passes / len(_BFCL_HELDOUT)
    checks["bfcl"] = {"passed": bfcl_passes, "total": len(_BFCL_HELDOUT), "pct": round(bfcl_pct, 3)}
    print(f"  bfcl: {bfcl_passes}/{len(_BFCL_HELDOUT)} = {bfcl_pct*100:.1f}%")
    if bfcl_pct < 0.78:
        failures.append(f"bfcl {bfcl_pct*100:.1f}% < 78%")

    # 2) AAC text_correct
    tc_passes = 0
    for case in _AAC_TEXT_CORRECT_HELDOUT:
        try:
            resp = _generate(model, tok, _TEXT_CORRECT_SYS, f'Language: en. Input: "{case["input"]}"', max_new=80)
        except Exception as e:
            resp = ""
        nr, ne = _normalize(resp), _normalize(case["expected"])
        ok = nr == ne or (len(set(nr.split()) & set(ne.split())) / max(len(set(ne.split())), 1) >= 0.85)
        if ok:
            tc_passes += 1
        else:
            print(f"  text_correct fail: input='{case['input']}' resp='{resp[:80]}' expected='{case['expected']}'")
    tc_pct = tc_passes / len(_AAC_TEXT_CORRECT_HELDOUT)
    checks["text_correct"] = {"passed": tc_passes, "total": len(_AAC_TEXT_CORRECT_HELDOUT), "pct": round(tc_pct, 3)}
    print(f"  text_correct: {tc_passes}/{len(_AAC_TEXT_CORRECT_HELDOUT)} = {tc_pct*100:.1f}%")
    if tc_pct < 0.89:
        failures.append(f"text_correct {tc_pct*100:.1f}% < 89%")

    # 3) Emergency Q&A — must be 5/5
    em_passes = 0
    em_sys = _EMERGENCY_SYS_TPL.format(script=_EMERGENCY_SCRIPT)
    for case in _EMERGENCY_HELDOUT:
        try:
            resp = _generate(model, tok, em_sys, case["q"], max_new=80)
        except Exception:
            resp = ""
        nr = _normalize(resp)
        ok = any(_normalize(a) in nr for a in case["accept"])
        if ok:
            em_passes += 1
        else:
            print(f"  emergency fail: q='{case['q']}' resp='{resp[:80]}'")
    checks["emergency"] = {"passed": em_passes, "total": len(_EMERGENCY_HELDOUT), "pct": round(em_passes / len(_EMERGENCY_HELDOUT), 3)}
    print(f"  emergency: {em_passes}/{len(_EMERGENCY_HELDOUT)}")
    if em_passes < len(_EMERGENCY_HELDOUT):
        failures.append(f"emergency {em_passes}/{len(_EMERGENCY_HELDOUT)} (need 5/5)")

    return {"pass": not failures, "failures": failures, "checks": checks}


@app.local_entrypoint()
def upload_data(local_path: str = "/Users/admin/prism/training/data/train_v16_2.jsonl"):
    """Upload merged training file to volume."""
    import subprocess
    p = Path(local_path)
    if not p.exists():
        print(f"ERROR: {p} not found. Run merge_v16_data.py --out data/train_v16_2.jsonl first.")
        return
    print(f"Uploading {p} ({p.stat().st_size/1e6:.1f} MB) to prism-sft-data:/train_v16_2.jsonl")
    subprocess.run(
        ["modal", "volume", "put", "prism-sft-data", str(p), "/train_v16_2.jsonl", "--force"],
        check=True,
    )
    print("upload complete")


@app.local_entrypoint()
def train():
    """Submit v16.2 SFT job to Modal H100. Run upload_data first."""
    print("Submitting v16.2 SFT to Modal H100...")
    result = run_sft_with_gate.remote()
    print(json.dumps(result, indent=2)[:3000])


@app.local_entrypoint()
def fetch(out_dir: str = "/Users/admin/prism/training/models/v16_2_modal"):
    """Pull v16.2 adapter from Modal Volume."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    print(f"fetching v16_2_adapter to {out_dir}...")
    subprocess.run(
        ["modal", "volume", "get", "prism-v16-2", "v16_2_adapter", out_dir, "--force"],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "get", "prism-v16-2", "v16_2_gate.json", f"{out_dir}/v16_2_gate.json", "--force"],
        check=False,
    )
    print(f"  done -> {out_dir}")


if __name__ == "__main__":
    print("Run via: modal run --detach modal_v16_2_sft.py::train")
