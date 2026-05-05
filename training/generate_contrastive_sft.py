#!/usr/bin/env python3
"""Hand-crafted contrastive SFT generator.

Reviewer fix #2: produces ~200 hard-negative SFT examples whose <think>
trace explicitly DEBATES the two confused tools, rather than vaguely
asserting the correct one. Uses Claude as a teacher to expand seed
prompts into varied phrasings and reasoning paths.

Output: data/contrastive_sft.jsonl in the {"text": "..."} chatml format
that sft_train.py / mlx_lm.lora consumes.

Confused pairs targeted (from BFCL failure analysis):
  - session_forget_memory   vs  session_search_memory
  - session_save_ledger     vs  session_save_handoff
  - session_export_memory   vs  session_save_handoff
  - knowledge_search        vs  session_search_memory
  - knowledge_forget        vs  session_forget_memory
  - session_compact_ledger  vs  session_save_ledger
  - session_health_check    vs  session_load_context
  - session_save_experience vs  session_save_ledger
"""
import os, sys, json, time, argparse, urllib.request
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "contrastive_sft.jsonl"
TEACHER_MODEL = "qwen2.5-coder:32b"   # local Ollama teacher
OLLAMA_API = "http://localhost:11434/api/generate"
EXAMPLES_PER_PAIR = 50   # 8 pairs * 50 = 400 — meets ≥50 per-pair TODO target

CONFUSED_PAIRS = [
    {
        "right": "session_forget_memory",
        "right_desc": "delete a SPECIFIC memory entry by id",
        "right_args_example": {"memory_id": "mem-abc-123"},
        "wrong": "session_search_memory",
        "wrong_desc": "QUERY past sessions for information",
        "axis": "delete a known item vs. retrieve information",
    },
    {
        "right": "session_save_ledger",
        "right_desc": "log a chunk of completed work in the session ledger",
        "right_args_example": {"project": "my-project", "conversation_id": "session-1", "summary": "Completed work"},
        "wrong": "session_save_handoff",
        "wrong_desc": "snapshot full state for transfer to another developer",
        "axis": "ledger entry (today's accomplishments) vs. handoff (full state transfer)",
    },
    {
        "right": "session_export_memory",
        "right_desc": "dump memory data to a file/directory for backup",
        "right_args_example": {"output_dir": "/tmp/prism-backup", "format": "json"},
        "wrong": "session_save_handoff",
        "wrong_desc": "snapshot state for the next developer to resume",
        "axis": "user wants a file/backup vs. user wants to hand off to a teammate",
    },
    {
        "right": "knowledge_search",
        "right_desc": "search the curated KNOWLEDGE base / institutional knowledge",
        "right_args_example": {"query": "rate limiting strategies"},
        "wrong": "session_search_memory",
        "wrong_desc": "search past CONVERSATION/session memory",
        "axis": "curated knowledge base vs. raw past-session memory",
    },
    {
        "right": "knowledge_forget",
        "right_desc": "wipe knowledge entries for a project (project-scoped)",
        "right_args_example": {"project": "prism-mcp"},
        "wrong": "session_forget_memory",
        "wrong_desc": "delete one specific memory entry by id",
        "axis": "bulk knowledge purge by project vs. delete one specific memory_id",
    },
    {
        "right": "session_compact_ledger",
        "right_desc": "MAINTENANCE — roll up OLD ledger entries into AI summaries to keep the ledger lean (acts on existing entries, doesn't add new ones)",
        "right_args_example": {"project": "prism-mcp", "keep_recent": 10, "threshold": 50},
        "wrong": "session_save_ledger",
        "wrong_desc": "APPEND a new immutable end-of-session log entry recording today's accomplishments",
        "axis": "compress/clean up existing history vs. append a new entry to history",
    },
    {
        "right": "session_health_check",
        "right_desc": "DIAGNOSTIC — fsck for memory: detect missing embeddings, duplicate entries, orphaned handoffs, stale rollups; optionally auto_fix",
        "right_args_example": {"auto_fix": False},
        "wrong": "session_load_context",
        "wrong_desc": "READ past project state/TODOs/summaries at the START of a new session",
        "axis": "diagnose memory integrity vs. retrieve past context to resume work",
    },
    {
        "right": "session_save_experience",
        "right_desc": "record a TYPED behavioral event (correction/success/failure/learning/validation_result) with structured context+action+outcome for pattern detection",
        "right_args_example": {"project": "prism-mcp", "event_type": "correction", "context": "agent suggested A", "action": "ran A", "outcome": "user reverted to B"},
        "wrong": "session_save_ledger",
        "wrong_desc": "append a flat end-of-session work-log summary (project + conversation_id + summary)",
        "axis": "structured typed event for behavioral learning vs. flat session summary log",
    },
]

SYSTEM_PROMPT = """You generate contrastive supervised-fine-tuning examples for a tool-calling LLM.

Each example must:
1. Have a USER prompt that PLAUSIBLY could be answered by EITHER of two confused tools.
2. Have an ASSISTANT response with a <think> trace that EXPLICITLY DEBATES both tools — naming the wrong tool, explaining why it is tempting, then explaining the discriminating signal that makes the right tool correct.
3. End with a tool call to the RIGHT tool only.

Reasoning style: ~2-3 sentences inside <think>. No filler. Always mention the wrong tool by name and say what would have made it correct, then commit to the right tool.

Output ONLY valid JSON — an array of {prompt, think, args} objects. No prose, no markdown, no code fences."""

USER_TEMPLATE = """Generate {n} diverse contrastive training examples for these two confused tools.

RIGHT tool: {right}
  Purpose: {right_desc}
  Required arg keys: {right_args_keys}

WRONG (tempting) tool: {wrong}
  Purpose: {wrong_desc}

Discriminating axis: {axis}

Each example needs:
- "prompt": natural-language user request that BLURS the two tools (varied vocabulary, lengths, registers — terse, verbose, slangy, formal, indirect)
- "think": 2-3 sentence contrastive reasoning that names {wrong} as the tempting alternative, says what would have justified it, then explains why {right} is correct
- "args": JSON object of arguments for {right}, realistic but varied (different project names, queries, ids — never literally "my-project" or "target-id")

Return ONLY a JSON array of {n} objects. Make every prompt distinct."""


def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        # take content between first and last fence
        text = text.split("```", 2)
        text = text[1] if len(text) >= 2 else text[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    # locate first '[' so we tolerate any preamble the model adds
    if "[" in text:
        text = text[text.index("["):]
    if "]" in text:
        text = text[: text.rindex("]") + 1]
    return text


def call_teacher(pair, n, max_retries=2):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        + USER_TEMPLATE.format(
            n=n,
            right=pair["right"],
            right_desc=pair["right_desc"],
            right_args_keys=list(pair["right_args_example"].keys()),
            wrong=pair["wrong"],
            wrong_desc=pair["wrong_desc"],
            axis=pair["axis"],
        )
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0.4, "num_predict": 8000, "num_ctx": 8192},
    }).encode("utf-8")
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(OLLAMA_API, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = _strip_fence(data.get("response", ""))
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  ! attempt {attempt+1}: JSON parse failed ({e}); retrying...")
        except Exception as e:
            last_err = e
            print(f"  ! attempt {attempt+1}: {e}; retrying...")
    raise RuntimeError(f"teacher failed after {max_retries+1} attempts: {last_err}")


def to_chatml(prompt, think, tool_name, args):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think}\n</think>\n\n"
        f"<|tool_call|>\n"
        f"{json.dumps({'name': tool_name, 'arguments': args})}\n"
        f"<|tool_call_end|><|im_end|>"
    )
    return {"text": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", type=int, default=EXAMPLES_PER_PAIR,
                    help=f"Examples per confused pair (default {EXAMPLES_PER_PAIR})")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as fout:
        for pair in CONFUSED_PAIRS:
            print(f"[{pair['right']} vs {pair['wrong']}] requesting {args.per_pair} examples via {TEACHER_MODEL}...")
            t0 = time.time()
            try:
                examples = call_teacher(pair, args.per_pair)
            except Exception as e:
                print(f"  ! teacher failed: {e}", file=sys.stderr)
                continue
            kept = 0
            for ex in examples:
                if not all(k in ex for k in ("prompt", "think", "args")):
                    continue
                line = to_chatml(ex["prompt"], ex["think"], pair["right"], ex["args"])
                fout.write(json.dumps(line) + "\n")
                kept += 1
            total += kept
            print(f"  kept {kept}/{len(examples)} in {time.time()-t0:.1f}s")

    print(f"\nWrote {total} contrastive SFT examples to {out_path}")


if __name__ == "__main__":
    main()
