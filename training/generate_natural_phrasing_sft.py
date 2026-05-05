#!/usr/bin/env python3
"""Natural-phrasing SFT generator.

Targets the v5b weakness: natural_phrasing 53% (7/15 failures). The model
handles direct/keyword-rich requests well but stumbles on indirect,
conversational, slangy phrasings ("jot down", "punt it", "bring me up to
speed", etc.).

Generates ~200 examples spread across 11 tools, each with:
- A colloquial/indirect prompt (no Prism keywords; relies on intent)
- A think trace that surfaces the implicit intent and commits to the tool
- A correct tool call

Output: data/natural_phrasing_sft.jsonl
"""
import os, sys, json, time, argparse, urllib.request
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "natural_phrasing_sft.jsonl"
TEACHER_MODEL = "qwen2.5-coder:32b"
OLLAMA_API = "http://localhost:11434/api/generate"
EXAMPLES_PER_TOOL = 18   # 11 tools * 18 = 198

# (tool name, what it does, required arg keys w/ realistic example, intent triggers)
TOOLS = [
    ("session_load_context",
     "resume past work on a project — load saved memory/state",
     {"project": "billing-portal"},
     "user implicitly wants to pick up where they left off (catch up, get up to speed, refresh, where were we, sync up)"),
    ("session_save_ledger",
     "log completed work in the session ledger",
     {"project": "auth-service", "conversation_id": "session-2025-04-29", "summary": "Refactored OAuth flow"},
     "user implicitly wants to record/document/log today's accomplishments (jot down, write up, capture, note for posterity)"),
    ("session_save_handoff",
     "snapshot full state for the next developer to resume",
     {"project": "mobile-app"},
     "user implicitly is wrapping up and someone else is taking over (passing the torch, end of shift, EOD wrap, signing off, going on vacation)"),
    ("session_search_memory",
     "search past CONVERSATION/session memory for prior discussions",
     {"query": "what we decided about caching"},
     "user implicitly wants to recall a prior decision/conversation (remind me, did we ever, refresh my memory, pretty sure we discussed)"),
    ("session_forget_memory",
     "delete a SPECIFIC memory entry by id",
     {"memory_id": "mem-2024-11-deprecated-script"},
     "user implicitly wants a single wrong/stale entry gone (kill that, scrap it, get rid of, that note is bogus)"),
    ("session_health_check",
     "diagnose the memory backend / run a system status check",
     {},
     "user implicitly wants reassurance the system is working (something feels off, run a check, give me a status, are we good)"),
    ("session_compact_ledger",
     "summarize and archive old ledger entries to reduce size",
     {"project": "data-pipeline"},
     "user implicitly wants the ledger trimmed/condensed (it's getting bloated, clean up old stuff, archive the old entries)"),
    ("session_export_memory",
     "dump memory data to a file/directory for backup",
     {"output_dir": "/tmp/backup-2025-04", "format": "json"},
     "user implicitly wants a portable copy/backup file (snapshot it to disk, send me a dump, give me an offline copy, archive to a file)"),
    ("session_task_route",
     "decide whether a task should go to the local model or be escalated",
     {"task_description": "refactor the css grid layout to use named areas"},
     "user implicitly is asking whether to delegate (can the small model handle, should I do this myself, is this trivial enough, who should take this)"),
    ("knowledge_search",
     "query curated knowledge base / institutional knowledge",
     {"query": "rate limiting strategies in our microservices"},
     "user implicitly wants accumulated team wisdom on a topic (what do we usually do, any docs on, established practice, our standard approach)"),
    ("knowledge_forget",
     "wipe knowledge entries for a project (project-scoped purge)",
     {"project": "deprecated-monolith"},
     "user implicitly wants all knowledge for a stale project gone (purge everything about, scrub our notes on, the project is dead, dump everything)"),
]

SYSTEM_PROMPT = """You generate natural-phrasing supervised fine-tuning examples for a tool-calling LLM.

Each example is a USER prompt that conveys the user's intent through INDIRECT, CONVERSATIONAL, or COLLOQUIAL phrasing — without using any Prism tool keywords directly. The model has to infer the right tool from intent alone.

The ASSISTANT response must:
1. Have a <think> trace that surfaces the IMPLICIT intent ("the user is asking me to X") and names the tool that satisfies it.
2. End with a correct tool call.

Vocabulary rules:
- Use varied informal registers: slang, idioms, casual phrasings, indirect questions, hand-wavy descriptions.
- Avoid keyword-matching cheats — DO NOT use words that exactly match the tool name (e.g. for session_load_context, avoid 'load' and 'context').
- Reference real-world coding/work scenarios: bugs, refactors, deploys, standup, EOD, handoffs, code review.

Output ONLY a JSON array of {prompt, think, args} objects. No prose, no markdown, no code fences."""

USER_TEMPLATE = """Generate {n} diverse natural-phrasing examples for this tool.

Tool: {tool}
Purpose: {purpose}
Required arg keys (use realistic varied values, NEVER literal placeholders like 'my-project' or 'target-id'): {arg_keys}
Intent triggers: {triggers}

Each example needs:
- "prompt": indirect/colloquial/conversational user request that does NOT contain the tool name's keywords. Vary length (short / medium / long), register (terse / wordy / professional / casual / slangy), and context (standup / EOD / handoff / debugging / migration / planning).
- "think": 2-3 sentences naming the implicit intent and committing to the tool. Mention that the user did NOT use direct keywords but the meaning is clear.
- "args": realistic, varied JSON object for the tool's required keys. Use specific project names, queries, ids, paths.

Return ONLY a JSON array of {n} objects. Each prompt must be distinct."""


def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) >= 2 else text[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    if "[" in text:
        text = text[text.index("["):]
    if "]" in text:
        text = text[: text.rindex("]") + 1]
    return text


def call_teacher(tool, purpose, args_example, triggers, n, max_retries=2):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        + USER_TEMPLATE.format(
            n=n, tool=tool, purpose=purpose,
            arg_keys=list(args_example.keys()) if args_example else [],
            triggers=triggers,
        )
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0.55, "num_predict": 3500, "num_ctx": 4096},
    }).encode("utf-8")
    last = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(OLLAMA_API, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = _strip_fence(data.get("response", ""))
            return json.loads(text)
        except Exception as e:
            last = e
            print(f"  ! attempt {attempt+1}: {e}; retrying...")
    raise RuntimeError(f"teacher failed: {last}")


def to_chatml(prompt, think, tool, args):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think}\n</think>\n\n"
        f"<|tool_call|>\n"
        f"{json.dumps({'name': tool, 'arguments': args})}\n"
        f"<|tool_call_end|><|im_end|>"
    )
    return {"text": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tool", type=int, default=EXAMPLES_PER_TOOL)
    ap.add_argument("--chunk", type=int, default=6,
                    help="Examples per Ollama call (smaller = more reliable, fewer hangs)")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as fout:
        for tool, purpose, args_ex, triggers in TOOLS:
            chunks = []
            remaining = args.per_tool
            while remaining > 0:
                chunks.append(min(args.chunk, remaining))
                remaining -= args.chunk

            print(f"[{tool}] requesting {args.per_tool} examples in {len(chunks)} chunks of {args.chunk}...", flush=True)
            tool_kept = 0
            for ci, n in enumerate(chunks, 1):
                t0 = time.time()
                try:
                    examples = call_teacher(tool, purpose, args_ex, triggers, n)
                except Exception as e:
                    print(f"  ! chunk {ci}/{len(chunks)} failed: {e}", file=sys.stderr, flush=True)
                    continue
                kept = 0
                for ex in examples:
                    if not all(k in ex for k in ("prompt", "think", "args")):
                        continue
                    line = to_chatml(ex["prompt"], ex["think"], tool, ex["args"])
                    fout.write(json.dumps(line) + "\n")
                    kept += 1
                fout.flush()
                tool_kept += kept
                print(f"  chunk {ci}/{len(chunks)}: kept {kept}/{len(examples)} in {time.time()-t0:.1f}s", flush=True)
            total += tool_kept
            print(f"  tool total: {tool_kept}", flush=True)

    print(f"\nWrote {total} natural-phrasing SFT examples to {out_path}")


if __name__ == "__main__":
    main()
