#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for tool-use accuracy.
Uses deterministic reward function — no reward model needed.
"""
import json
import os
import sys
import subprocess
import re

MODEL_PATH = "/Users/admin/prism/training/models/qwen-7b-mlx"
SFT_ADAPTER = "/Users/admin/prism/training/models/prism-sft-lora"
TOOL_SCHEMA = "/Users/admin/prism/training/data/tool_schema.json"
OUTPUT_ADAPTER = "/Users/admin/prism/training/models/prism-grpo-lora"
GRPO_DATA = "/Users/admin/prism/training/data/grpo_prompts.jsonl"

# Load valid tool names from schema
with open(TOOL_SCHEMA) as f:
    VALID_TOOLS = {t["name"] for t in json.load(f)["tools"]}

TOOL_PARAMS = {}
with open(TOOL_SCHEMA) as f:
    for t in json.load(f)["tools"]:
        TOOL_PARAMS[t["name"]] = {
            "required": set(t["parameters"].get("required", [])),
            "optional": set(t["parameters"]["properties"].keys()) - set(t["parameters"].get("required", []))
        }


def compute_reward(response_text: str) -> float:
    """
    Deterministic reward function for tool-use accuracy.
    STRICTLY rewards <think> + <tool_call> structure.
    """
    reward = 0.0

    # ── Structural Reward (CRITICAL) ──
    # Model MUST start with a <think> tag
    if response_text.strip().startswith('<think>'):
        reward += 5.0
    else:
        reward -= 2.0

    if '<tool_call>' in response_text and '</tool_call>' in response_text:
        reward += 10.0  # Dominant reward for correct format
    
    # ── CoT reasoning reward ──
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        if len(think_text) > 50:
            reward += 2.0  # Reward substantive thought
        if "tool" in think_text.lower() or "requires" in think_text.lower():
            reward += 1.0  # Reward strategy-oriented thought

    # Check if response contains a tool call (try different formats)
    tool_content = None
    tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response_text, re.DOTALL)
    if tool_match:
        tool_content = tool_match.group(1)
    
    if not tool_content:
        return reward + (0.0 if len(response_text) > 20 else -1.0)

    try:
        tool_call = json.loads(tool_content)
        reward += 2.0  # Valid JSON
    except json.JSONDecodeError:
        return reward - 5.0

    tool_name = tool_call.get("name", "")
    if tool_name in VALID_TOOLS:
        reward += 5.0
    else:
        reward -= 10.0
        return reward

    args = tool_call.get("arguments", {})
    params = TOOL_PARAMS.get(tool_name, {"required": set(), "optional": set()})

    missing_required = params["required"] - set(args.keys())
    if not missing_required:
        reward += 5.0
    else:
        reward -= 5.0 * len(missing_required)

    return reward


def generate_grpo_prompts():
    """Generate prompts for GRPO training."""
    prompts = [
        "Load the full context for the prism-mcp project",
        "Save this session: implemented RBAC roles",
        "Explain how React Server Components work",
        "Search for sessions about JWT authentication in synalux-private",
        "List all sessions for project bcba-private",
        "Delete the session from yesterday about the billing bug",
        "What do we know about the 'Zero-Search' architecture in prism?",
        "Store this knowledge: The ACT-R decay rate is 0.5 for rollup nodes",
        "Search for patterns about memory consolidation",
        "Hand off the billing task from dev to security: payment logic is ready",
        "Initialize a deep session for project synalux-docs",
        "Write a hello world in Python",
        "Find work related to the schema migration in v9.4",
        "What is the status of the HIPAA security audit?",
        "Log work: fixed the abortPipeline syntax error in dashboard",
    ]

    with open(GRPO_DATA, "w") as f:
        for prompt in prompts:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are Prism, an AI coding assistant with persistent memory. Use MCP tools when appropriate."},
                    {"role": "user", "content": prompt}
                ]
            }) + "\n")
    return prompts


def generate_synthetic_chosen(prompt: str) -> str:
    """Generate a perfect response for a given prompt."""
    if "Load" in prompt and "prism-mcp" in prompt:
        return '<think>The user wants to load project context for "prism-mcp". This is a read operation. I should use session_load_context with the project name.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "prism-mcp", "level": "shallow"}}\n</tool_call>'
    if "Save" in prompt and "RBAC" in prompt:
        return '<think>The user wants to save a work session about RBAC. This is a write operation to the session ledger. I should use session_save.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "prism-mcp", "summary": "implemented RBAC roles"}}\n</tool_call>'
    if "Search" in prompt and "JWT" in prompt:
        return '<think>The user is asking about JWT authentication work. I should use session_search to find relevant history.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "JWT authentication", "project": "synalux-private"}}\n</tool_call>'
    if "Store this knowledge" in prompt:
        return '<think>The user wants to store a principle about memory decay. I should use knowledge_save for permanent storage.</think>\n\n<tool_call>\n{"name": "knowledge_save", "arguments": {"project": "prism", "concept": "ACT-R Decay Rate", "description": "The ACT-R decay rate is 0.5 for rollup nodes", "confidence": 1.0}}\n</tool_call>'
    return None


def main():
    print("=" * 60)
    print("GRPO Alignment for Tool-Use Accuracy")
    print("=" * 60)

    prompts = generate_grpo_prompts()
    dpo_data = []
    
    try:
        from mlx_lm import load, generate
        print("\nLoading SFT model + adapter...")
        model, tokenizer = load(MODEL_PATH, adapter_path=SFT_ADAPTER)
        
        for i, prompt in enumerate(prompts):
            sys_msg = "You are Prism, an AI coding assistant with persistent memory. Use MCP tools when appropriate."
            full_prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            completions = []
            for j in range(4):
                try:
                    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=256)
                    reward = compute_reward(response)
                    print(f"    [Prompt {i+1} Gen {j+1}] Reward: {reward:+.1f} | Response: {response[:60].replace('\n', ' ')}...")
                    completions.append((response, reward))
                except Exception as e:
                    continue
            
            if len(completions) >= 2:
                completions.sort(key=lambda x: x[1], reverse=True)
                best = completions[0]
                worst = completions[-1]
                
                # Injection is now mandatory for these specific prompts to force structural change
                synthetic_chosen = generate_synthetic_chosen(prompt)
                if synthetic_chosen:
                    dpo_data.append({
                        "prompt": prompt,
                        "chosen": synthetic_chosen,
                        "rejected": worst[0],
                    })
                elif best[1] > worst[1]:
                    dpo_data.append({
                        "prompt": prompt,
                        "chosen": best[0],
                        "rejected": worst[0],
                    })
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(prompts)} prompts, {len(dpo_data)} preference pairs")
        
        if len(dpo_data) >= 1:
            dpo_train_path = "/Users/admin/prism/training/data/dpo_train.jsonl"
            with open(dpo_train_path, "w") as f:
                # Heavy repetition to force the structural shift
                for _ in range(50):
                    for d in dpo_data:
                        entry = {
                            "chosen": [
                                {"role": "user", "content": d["prompt"]},
                                {"role": "assistant", "content": d["chosen"]}
                            ],
                            "rejected": [
                                {"role": "user", "content": d["prompt"]},
                                {"role": "assistant", "content": d["rejected"]}
                            ]
                        }
                        f.write(json.dumps(entry) + "\n")
            
            print(f"\nRunning Structural DPO training...")
            cmd = [
                sys.executable, "-m", "mlx_lm.lora",
                "--model", MODEL_PATH,
                "--train",
                "--data", os.path.dirname(dpo_train_path),
                "--adapter-path", OUTPUT_ADAPTER,
                "--num-layers", "12",
                "--batch-size", "1",
                "--iters", "500",
                "--max-seq-length", "1024",
                "--learning-rate", "4e-5", # Aggressive for structure
                "--steps-per-report", "50",
                "--save-every", "250",
                "--resume-adapter-file", os.path.join(SFT_ADAPTER, "adapters.safetensors"),
            ]
            subprocess.run(cmd)
        else:
            print(f"\nNot enough preference pairs ({len(dpo_data)}) for DPO.")
    
    except Exception as e:
        print(f"GRPO failed: {e}")

if __name__ == "__main__":
    main()
