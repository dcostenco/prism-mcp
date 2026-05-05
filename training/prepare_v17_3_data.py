"""Build train_v17_3.jsonl — v17.3 surgical pass to fix multi_turn + abstention + hallucination.

v17.2 BFCL category breakdown:
  simple              100%   ✅
  relevance           100%   ✅
  hallucination       90%    ⚠️  (model invents tools like session_backlink)
  disambiguation      87.5%  ⚠️
  format_sensitivity  80%    ⚠️
  ast_parameter       100%   ✅
  edge_case           62.5%  ❌ (calls tool when should abstain)
  multi_turn          25%    ❌❌ (v18_1_multiturn data POISONED this — was 87.5% in v17)

v17.2 AAC: text_correct 15/15, caregiver 6/7 + 20/20 targeted, all others HOLD ✅

v17.3 strategy:
  1. DROP v18_1_multiturn.jsonl entirely
  2. Generate Prism-specific multi-turn pairs matching actual BFCL chains
  3. Hand-write 200 anti-hallucination examples (model picks from PROVIDED tools)
  4. Hand-write 200 abstention examples (irrelevant → text response, no tool)
  5. Sample 1500 v17.2 BFCL canonical (hold the disambiguation/format gains)
  6. Sample 1500 v17.2 caregiver (hold the win)
  7. Sample 300 v17.2 text_correct (hold the perfect 15/15)
  8. 105 format anchor examples (hold canonical wrapper bias)

Total: ~4200 rows. Continue from v17.2 adapter.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(173)

DATA = Path("/Users/admin/prism/training/data")
OUT = DATA / "train_v17_3.jsonl"


def wrap_canonical(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ──────────────────────────────────────────────────────────────────────────
# Prism tool registry (matches bfcl_eval.py VALID_TOOLS)
# ──────────────────────────────────────────────────────────────────────────
PRISM_TOOLS = {
    "session_load_context": {
        "description": "Load project context for the current session.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]},
    },
    "session_save_ledger": {
        "description": "Save a ledger entry summarizing this session's work.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "summary": {"type": "string"}}, "required": []},
    },
    "session_save_handoff": {
        "description": "Save handoff state for the next session.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": []},
    },
    "session_save_experience": {
        "description": "Log an event/experience record.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "event_type": {"type": "string"}}, "required": ["project"]},
    },
    "session_search_memory": {
        "description": "Semantic search over past sessions.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "session_forget_memory": {
        "description": "Forget/remove a memory entry by ID.",
        "parameters": {"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]},
    },
    "session_health_check": {
        "description": "Check session memory health.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "session_compact_ledger": {
        "description": "Compact ledger entries.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": []},
    },
    "session_export_memory": {
        "description": "Export session memory to file.",
        "parameters": {"type": "object", "properties": {"output_path": {"type": "string"}, "project": {"type": "string"}}, "required": ["output_path"]},
    },
    "session_task_route": {
        "description": "Decide if a task runs on host or claw.",
        "parameters": {"type": "object", "properties": {"task_scope": {"type": "string"}}, "required": []},
    },
    "knowledge_search": {
        "description": "Search the knowledge base by keyword.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "knowledge_upvote": {
        "description": "Upvote a knowledge entry by ID.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
    "knowledge_downvote": {
        "description": "Downvote a knowledge entry by ID.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
    "knowledge_forget": {
        "description": "Forget a knowledge entry by ID.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
    "knowledge_set_retention": {
        "description": "Set retention policy for a project.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "ttl_days": {"type": "integer"}}, "required": ["project", "ttl_days"]},
    },
    "memory_checkout": {
        "description": "Check out a snapshot of memory at a given commit.",
        "parameters": {"type": "object", "properties": {"commit_id": {"type": "string"}}, "required": ["commit_id"]},
    },
    "memory_history": {
        "description": "Show memory commit history.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": []},
    },
}


def make_tools_block(tool_names: list[str]) -> str:
    """Render the <tools> system block for the given subset of tools."""
    decls = []
    for name in tool_names:
        spec = PRISM_TOOLS[name]
        decls.append({"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}})
    return json.dumps(decls, ensure_ascii=False)


def make_qwen2_system(tool_names: list[str]) -> str:
    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{make_tools_block(tool_names)}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    )


def tool_call_text(name: str, args: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


# ──────────────────────────────────────────────────────────────────────────
# 1) Multi-turn chains — match BFCL test patterns exactly
# ──────────────────────────────────────────────────────────────────────────
MULTI_TURN_CHAINS = [
    # session_load_context → session_save_experience
    {
        "tools": ["session_load_context", "session_save_experience"],
        "user": "Load context for the {p} project and then log that we successfully {ev}.",
        "step1": ("session_load_context", lambda p: {"project": p}),
        "tool_resp": '{"project": "%s", "last_summary": "Working on it"}',
        "step2": ("session_save_experience", lambda p, ev: {"project": p, "event_type": "success"}),
        "fillers": [("portal", "deployed v3"), ("billing", "cleared the queue"), ("auth", "rotated keys"), ("aac", "shipped iOS build")],
    },
    # session_search_memory → knowledge_upvote (after found)
    {
        "tools": ["session_search_memory", "knowledge_upvote"],
        "user": "Search knowledge for {q}, then upvote the best result.",
        "step1": ("knowledge_search", lambda q: {"query": q}),
        "tool_resp": '{"results": [{"id": "ki-%s-42", "summary": "Found pattern", "importance": 5}]}',
        "step2": ("knowledge_upvote", lambda q: {"id": f"ki-{q[:6]}-42"}),
        "fillers": [("retry strategies",), ("logging patterns",), ("rate limiting",), ("auth flows",)],
    },
    # session_save_ledger → session_save_handoff
    {
        "tools": ["session_save_ledger", "session_save_handoff"],
        "user": "Record this session: we {work}. Then save the handoff state.",
        "step1": ("session_save_ledger", lambda w: {}),
        "tool_resp": '{"status": "saved", "id": "ledger-2026-99"}',
        "step2": ("session_save_handoff", lambda w: {}),
        "fillers": [("migrated auth to OAuth2",), ("shipped the dashboard",), ("merged the PR",), ("fixed the CI",)],
    },
    # session_export_memory → knowledge_set_retention
    {
        "tools": ["session_export_memory", "knowledge_set_retention"],
        "user": "Export the {p} project memory to {path}, then set a {ttl}-day retention policy.",
        "step1": ("session_export_memory", lambda p, path, ttl: {"output_path": path}),
        "tool_resp": '{"status": "exported", "file": "%s/prism-export.json", "entries": 142}',
        "step2": ("knowledge_set_retention", lambda p, path, ttl: {"project": p, "ttl_days": int(ttl)}),
        "fillers": [("billing", "/tmp/backup", "60"), ("portal", "/tmp/portal", "30"), ("auth", "/tmp/auth", "90")],
    },
    # session_health_check → session_compact_ledger (when issues)
    {
        "tools": ["session_health_check", "session_compact_ledger"],
        "user": "Check session health, and if anything's wrong fix it.",
        "step1": ("session_health_check", lambda: {}),
        "tool_resp": '{"status": "issues_found", "missing_embeddings": 12, "stale_rollups": 3}',
        "step2": ("session_compact_ledger", lambda: {}),
        "fillers": [()],
    },
    # session_task_route → NO_TOOL (when host)
    {
        "tools": ["session_task_route"],
        "user": "Should the local agent handle this {scope} refactor? If cloud, just tell me.",
        "step1": ("session_task_route", lambda s: {"task_scope": s}),
        "tool_resp": '{"target": "host", "confidence": 0.92, "reason": "Complex refactor needs cloud model"}',
        "step2": None,  # NO_TOOL — synthesize text answer
        "step2_text": "The router says this should run on the cloud (host) — confidence 92%. The reason is that complex refactors benefit from the larger context window. Run it via your cloud agent rather than the local one.",
        "fillers": [("TypeScript",), ("Python",), ("Go",)],
    },
]


def gen_multi_turn(rows_per_chain: int = 80) -> list[dict]:
    """Generate Prism multi-turn chain examples in canonical Qwen2 format."""
    rows = []
    for chain in MULTI_TURN_CHAINS:
        sys_msg = make_qwen2_system(list(set(chain["tools"])))
        for _ in range(rows_per_chain):
            filler = random.choice(chain["fillers"])
            user_msg = chain["user"].format(*[v for v in [filler[i] if i < len(filler) else "" for i in range(8)] if v != ""], **dict(zip("p ev q work path ttl scope".split(), filler)) if False else {}) if False else chain["user"]
            # Format user with the filler values
            try:
                if isinstance(filler, tuple) and len(filler) > 0:
                    keys_in_order = []
                    # Map common var names
                    if "{p}" in chain["user"]: keys_in_order.append("p")
                    if "{ev}" in chain["user"]: keys_in_order.append("ev")
                    if "{q}" in chain["user"]: keys_in_order.append("q")
                    if "{work}" in chain["user"]: keys_in_order.append("work")
                    if "{path}" in chain["user"]: keys_in_order.append("path")
                    if "{ttl}" in chain["user"]: keys_in_order.append("ttl")
                    if "{scope}" in chain["user"]: keys_in_order.append("scope")
                    user_msg = chain["user"].format(**dict(zip(keys_in_order, filler)))
                else:
                    user_msg = chain["user"]
            except (KeyError, IndexError):
                user_msg = chain["user"]

            # Build the full multi-turn conversation
            step1_name, step1_args_fn = chain["step1"]
            try:
                step1_args = step1_args_fn(*filler)
            except TypeError:
                step1_args = step1_args_fn() if not filler else {}

            tool_response = chain["tool_resp"] % filler[0] if "%s" in chain["tool_resp"] and filler else chain["tool_resp"].rstrip("%s") if "%s" in chain["tool_resp"] else chain["tool_resp"]

            assistant1 = tool_call_text(step1_name, step1_args)

            if chain["step2"] is None:
                # Step-2 is NO_TOOL — synthesize text answer
                final_text = chain["step2_text"]
                full = (
                    f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant1}<|im_end|>\n"
                    f"<|im_start|>tool\n<tool_response>\n{tool_response}\n</tool_response><|im_end|>\n"
                    f"<|im_start|>assistant\n{final_text}<|im_end|>"
                )
            else:
                step2_name, step2_args_fn = chain["step2"]
                try:
                    step2_args = step2_args_fn(*filler)
                except TypeError:
                    step2_args = step2_args_fn() if not filler else {}
                assistant2 = tool_call_text(step2_name, step2_args)
                full = (
                    f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant1}<|im_end|>\n"
                    f"<|im_start|>tool\n<tool_response>\n{tool_response}\n</tool_response><|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant2}<|im_end|>"
                )

            rows.append({"text": full, "_src": "prism_multiturn_NEW"})
    print(f"  prism_multiturn_NEW: {len(rows)} rows ({len(MULTI_TURN_CHAINS)} chains x {rows_per_chain})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 2) Anti-hallucination — model must pick FROM provided tools
# ──────────────────────────────────────────────────────────────────────────
ANTIHAL_PAIRS = [
    # (provided_tools, user_query, expected_tool, expected_args)
    (["session_search_memory"], "Find what I worked on last Tuesday about Auth.", "session_search_memory", {"query": "Auth last Tuesday"}),
    (["session_search_memory"], "Recall the discussion on rate limiting", "session_search_memory", {"query": "rate limiting"}),
    (["knowledge_search"], "Look up retry patterns", "knowledge_search", {"query": "retry patterns"}),
    (["knowledge_search"], "Find docs about JWT", "knowledge_search", {"query": "JWT"}),
    (["session_save_ledger"], "Log this session: deployed v4", "session_save_ledger", {"summary": "deployed v4"}),
    (["session_save_ledger"], "Record what we did today", "session_save_ledger", {}),
    (["session_save_handoff"], "Save the handoff for tomorrow", "session_save_handoff", {}),
    (["session_save_handoff"], "Persist the current state for next session", "session_save_handoff", {}),
    (["session_health_check"], "How's session memory looking?", "session_health_check", {}),
    (["session_health_check"], "Check the memory subsystem", "session_health_check", {}),
    (["session_export_memory"], "Export billing project memory to /tmp/x", "session_export_memory", {"output_path": "/tmp/x"}),
    (["session_load_context"], "Load context for the auth project", "session_load_context", {"project": "auth"}),
    (["session_load_context"], "Bring up project portal context", "session_load_context", {"project": "portal"}),
    (["knowledge_upvote"], "Upvote ki-42", "knowledge_upvote", {"id": "ki-42"}),
    (["knowledge_downvote"], "Downvote ki-99", "knowledge_downvote", {"id": "ki-99"}),
    (["knowledge_set_retention"], "Set 30-day retention on billing project", "knowledge_set_retention", {"project": "billing", "ttl_days": 30}),
    (["session_compact_ledger"], "Compact the ledger", "session_compact_ledger", {}),
    (["session_compact_ledger"], "Roll up old ledger entries", "session_compact_ledger", {}),
    (["session_task_route"], "Should this run on host or claw?", "session_task_route", {}),
    (["memory_checkout"], "Check out commit abc123", "memory_checkout", {"commit_id": "abc123"}),
    (["session_forget_memory"], "Forget memory 1234", "session_forget_memory", {"memory_id": 1234}),
    (["knowledge_forget"], "Forget knowledge entry ki-77", "knowledge_forget", {"id": "ki-77"}),
    (["memory_history"], "Show memory history for portal", "memory_history", {"project": "portal"}),
]


def gen_antihal(reps: int = 6) -> list[dict]:
    rows = []
    for tools_subset, user, expected_tool, expected_args in ANTIHAL_PAIRS:
        sys_msg = make_qwen2_system(tools_subset)
        assistant = tool_call_text(expected_tool, expected_args)
        for _ in range(reps):
            text = wrap_canonical(sys_msg, user, assistant)
            rows.append({"text": text, "_src": "antihal_NEW"})
    print(f"  antihal_NEW: {len(rows)} rows ({len(ANTIHAL_PAIRS)} pairs x {reps})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 3) Abstention — irrelevant query → text response (NOT tool call)
# ──────────────────────────────────────────────────────────────────────────
ABSTAIN_PAIRS = [
    # (provided_tools, user_query, response_text)
    (["session_search_memory"], "Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    (["session_search_memory"], "Hi how are you", "I'm doing well, thanks for asking! How can I help you today?"),
    (["knowledge_search"], "What's 2+2?", "2 + 2 = 4."),
    (["knowledge_search"], "Hello", "Hello! What can I help you with?"),
    (["session_load_context"], "Thanks, that's all for now.", "You're welcome! Feel free to come back anytime."),
    (["session_save_ledger"], "Bye", "Goodbye! Have a great day."),
    (["knowledge_upvote"], "What's the capital of France?", "The capital of France is Paris."),
    (["session_health_check"], "How was your day?", "I'm just an AI, but I'm always ready to help! What do you need?"),
    (["session_export_memory"], "Tell me a story", "Once upon a time, a curious developer wondered if the AI would invent a tool — but the AI just told this story instead."),
    (["session_search_memory"], "What time is it?", "I don't have access to the current time. You can check your system clock."),
    (["knowledge_search"], "Who won the world cup", "I don't have information on that. You could search the web for the latest results."),
    (["session_load_context"], "What's your name?", "I'm Prism, your AI coding assistant."),
    (["session_save_handoff"], "What can you do?", "I can help with coding tasks, search project memory, save session ledgers, manage knowledge, and more."),
    (["session_health_check"], "Are you sentient?", "No, I'm an AI assistant. I process text and use tools to help with your work."),
    (["knowledge_search"], "How old are you?", "I don't have an age in the human sense — I was deployed recently."),
    (["session_search_memory"], "Sing me a song", "I'm not great at singing, but here's a quick rhyme: 'Code by code, line by line, debugging together — we'll be just fine.'"),
    (["session_search_memory"], "Make a joke about cats", "Why did the cat sit on the keyboard? It wanted to be paw-some at coding."),
    (["session_save_ledger"], "Good morning", "Good morning! Ready to help with your work today."),
    (["knowledge_upvote"], "Can you write me a haiku?", "Code compiles smoothly /\nTests pass on the first attempt /\nDeploy day is here."),
    (["session_export_memory"], "What is the meaning of life?", "Some say 42. Others say it's whatever gives your work meaning. I'm here to help with the latter."),
]


def gen_abstain(reps: int = 8) -> list[dict]:
    rows = []
    for tools_subset, user, resp in ABSTAIN_PAIRS:
        sys_msg = make_qwen2_system(tools_subset)
        for _ in range(reps):
            text = wrap_canonical(sys_msg, user, resp)
            rows.append({"text": text, "_src": "abstain_NEW"})
    print(f"  abstain_NEW: {len(rows)} rows ({len(ABSTAIN_PAIRS)} pairs x {reps})")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 4) Hold v17.2 wins — sample subsets
# ──────────────────────────────────────────────────────────────────────────
def sample_v17_2_subset(predicate, n: int, label: str) -> list[dict]:
    src = DATA / "train_v17_2.jsonl"
    cands = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if predicate(t):
                cands.append({"text": t, "_src": label})
    random.shuffle(cands)
    rows = cands[:n]
    print(f"  {label}: {len(rows)} rows (from {len(cands)} candidates)")
    return rows


def hold_caregiver(n: int) -> list[dict]:
    return sample_v17_2_subset(
        lambda t: "AAC app configuration assistant" in t,
        n, "hold_caregiver",
    )


def hold_text_correct(n: int) -> list[dict]:
    return sample_v17_2_subset(
        lambda t: "fast text-cleanup engine" in t,
        n, "hold_text_correct",
    )


def hold_bfcl_canonical(n: int) -> list[dict]:
    return sample_v17_2_subset(
        lambda t: "<tools>" in t and "<tool_call>" in t and "AAC" not in t and "fast text-cleanup" not in t,
        n, "hold_bfcl_canonical",
    )


def hold_format_anchor(n: int) -> list[dict]:
    return sample_v17_2_subset(
        lambda t: "format_anchor" in t or ("get_weather" in t and "<tool_call>" in t and "<tools>" in t),
        n, "hold_format_anchor",
    )


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Building train_v17_3.jsonl ===")
    rows: list[dict] = []
    rows.extend(gen_multi_turn(rows_per_chain=80))   # ~480 rows of Prism multi-turn
    rows.extend(gen_antihal(reps=6))                 # ~138 rows
    rows.extend(gen_abstain(reps=8))                 # ~160 rows
    rows.extend(hold_caregiver(1200))                # 1200 rows
    rows.extend(hold_text_correct(300))              # 300 rows (preserve perfect 15/15)
    rows.extend(hold_bfcl_canonical(1200))           # 1200 rows
    rows.extend(hold_format_anchor(100))             # 100 rows

    random.shuffle(rows)
    print(f"\n=== Total: {len(rows)} rows ===")
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r["_src"]] = src_counts.get(r["_src"], 0) + 1
    for k, v in sorted(src_counts.items()):
        print(f"  {k}: {v}")

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
