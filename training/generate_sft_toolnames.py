#!/usr/bin/env python3
"""
Targeted SFT to teach correct full Prism MCP tool names.
Fixes abbreviated names from base SFT (e.g., session_save → session_save_ledger).
"""
import json
import os
import random
from pathlib import Path

TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "</|tool_call|>"

# Each entry: (prompt, correct_full_tool_name, args)
TOOL_NAME_EXAMPLES = [
    # session_load_context (not "session_load")
    ("Load context for prism-mcp project", "session_load_context", {"project": "prism-mcp", "level": "deep", "toolAction": "loading project context", "toolSummary": "Loading deep context for the prism-mcp project"}),
    ("Show me the context for billing-portal", "session_load_context", {"project": "billing-portal", "level": "deep", "toolAction": "loading project context", "toolSummary": "Loading context for billing-portal project"}),
    ("Initialize context for billing-portal", "session_load_context", {"project": "billing-portal", "level": "deep", "toolAction": "loading project context", "toolSummary": "Loading project state from memory"}),

    # session_save_ledger (not "session_save")
    ("Save this session: fixed the OAuth bug", "session_save_ledger", {"project": "billing-portal", "conversation_id": "current", "summary": "Fixed the OAuth bug in the portal authentication flow", "toolAction": "saving work", "toolSummary": "Recording the OAuth fix to the session ledger"}),
    ("Record this work: migrated Stripe webhooks to v2 API", "session_save_ledger", {"project": "billing-portal", "conversation_id": "current", "summary": "Migrated Stripe webhooks to v2 API", "toolAction": "saving work", "toolSummary": "Recording Stripe webhook migration to ledger"}),
    ("Log what we did today on the billing module", "session_save_ledger", {"project": "billing-portal", "conversation_id": "current", "summary": "Billing module work completed", "toolAction": "saving work", "toolSummary": "Logging billing module progress"}),

    # session_search_memory (not "session_search")
    ("What did we work on last week related to billing?", "session_search_memory", {"query": "billing work last week"}),
    ("Look up past work on the OAuth2 refresh flow", "session_search_memory", {"query": "OAuth2 refresh flow implementation"}),
    ("Find what we decided about the caching layer", "session_search_memory", {"query": "caching layer decision"}),
    ("Search for notes about the database migration", "session_search_memory", {"query": "database migration notes"}),

    # session_save_handoff (not "session_handoff" or "session_save")
    ("Create a handoff for the billing-portal project", "session_save_handoff", {"project": "billing-portal"}),
    ("Save the current handoff state for docs-portal", "session_save_handoff", {"project": "docs-portal"}),
    ("Prepare handoff notes for the next session", "session_save_handoff", {"project": "prism-mcp"}),

    # session_forget_memory (not "session_delete" or "session_forget")
    ("Delete the memory entry for the broken config change", "session_forget_memory", {"memory_id": "broken_config_change", "reason": "Outdated configuration entry"}),
    ("Remove the memory about the failed deploy last Friday", "session_forget_memory", {"memory_id": "failed_deploy_friday", "reason": "Outdated deployment record"}),
    ("Forget the session entry about the old API design", "session_forget_memory", {"memory_id": "old_api_design", "reason": "Superseded by new API design"}),

    # session_health_check (not "health_check" or "knowledge_search")
    ("Check if the memory database has any integrity issues", "session_health_check", {"auto_fix": False}),
    ("Run a health check on the memory system", "session_health_check", {"auto_fix": False}),
    ("Is our memory system healthy? Auto-fix if not.", "session_health_check", {"auto_fix": True}),

    # knowledge_search (not "search" or general reasoning)
    ("Search knowledge base for anything about CORS policies", "knowledge_search", {"query": "CORS policies"}),
    ("What do we know about edge function cold starts?", "knowledge_search", {"query": "edge function cold starts"}),
    ("What's in our knowledge base about Supabase RLS policies?", "knowledge_search", {"query": "Supabase RLS policies"}),

    # session_compact_ledger (not abstaining)
    ("Compact old ledger entries for the prism-mcp project", "session_compact_ledger", {"project": "prism-mcp", "threshold": 50, "keep_recent": 10}),
    ("Clean up the session history, compact entries older than 30 days", "session_compact_ledger", {"project": "prism-mcp", "dry_run": True}),

    # session_export_memory (not "session_save" or "export")
    ("Export prism-mcp memory to /tmp/export", "session_export_memory", {"output_dir": "/tmp/export", "project": "prism-mcp", "format": "json"}),
    ("Export all memory data for billing-portal to downloads", "session_export_memory", {"output_dir": "/tmp/export", "project": "billing-portal", "format": "json"}),

    # session_task_route
    ("Route this task: refactoring the auth middleware", "session_task_route", {"task_description": "refactoring the auth middleware", "estimated_scope": "refactor"}),
    ("Should the local agent or cloud handle this CSS fix?", "session_task_route", {"task_description": "CSS fix for the login page", "estimated_scope": "minor_edit"}),

    # session_save_experience
    ("Log a success event: deployed the billing module", "session_save_experience", {"project": "billing-portal", "event_type": "success", "context": "billing module deployment", "action": "deployed", "outcome": "successful deployment without errors"}),

    # knowledge_upvote / knowledge_downvote
    ("Upvote that memory about the RBAC fix", "knowledge_upvote", {"id": "rbac_fix_memory"}),
    ("Downvote the stale entry about the old API endpoint", "knowledge_downvote", {"id": "old_api_endpoint"}),

    # session_backfill_links
    ("Backfill graph edges for the prism-mcp project", "session_backfill_links", {"project": "prism-mcp"}),

    # session_synthesize_edges
    ("Find semantic relationships between memory nodes", "session_synthesize_edges", {"project": "analytics-dashboard"}),
]

def generate_sft_data():
    """Generate SFT training data with correct tool names."""
    # R12-fix: Load tool schemas and system prompt
    try:
        from config import format_system_prompt
        _schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tool_schema.json")
        try:
            with open(_schema_path) as _f:
                _all_tools = json.load(_f).get("tools", [])
        except (FileNotFoundError, json.JSONDecodeError):
            _all_tools = []
        _sys_prompt = format_system_prompt(_all_tools)
    except ImportError:
        _sys_prompt = None

    data = []
    for prompt, tool_name, args in TOOL_NAME_EXAMPLES:
        # Build the expected completion
        tool_call_json = json.dumps({"name": tool_name, "arguments": args})
        completion = f'<|synalux_think|>\nThe user wants me to use the {tool_name} tool.\n</|synalux_think|>\n\n{TOOL_CALL_OPEN}\n{tool_call_json}\n{TOOL_CALL_CLOSE}'
        
        msgs = []
        if _sys_prompt:
            msgs.append({"role": "system", "content": _sys_prompt})
        msgs.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ])
        data.append({"messages": msgs})
    
    # Add reasoning examples (should NOT call tools)
    reasoning_pairs = [
        ("What is the difference between TCP and UDP?", "TCP is a connection-oriented protocol..."),
        ("How does React's virtual DOM work?", "React's virtual DOM is an in-memory representation..."),
        ("Write a hello world in Python", "```python\nprint('Hello, World!')\n```"),
        ("Explain how JWT tokens work", "JWT (JSON Web Token) is a compact..."),
        ("What are the pros and cons of microservices?", "Microservices offer several advantages..."),
        # Adversarial keyword traps
        ("How do I save state in React with useState?", "In React, useState is a hook..."),
        ("Explain how session tokens work in web authentication", "Session tokens are used to maintain..."),
        ("What is knowledge distillation in machine learning?", "Knowledge distillation is a technique..."),
        ("How do I save data to localStorage in the browser?", "localStorage is a web storage API..."),
        ("What is task routing in distributed systems like Celery?", "In Celery, task routing determines..."),
    ]
    
    for prompt, answer in reasoning_pairs:
        # R13-fix: Wrap answer in <|synalux_answer|> tags to match System Prompt Rule 1
        completion = f'<|synalux_think|>\nThis is a general knowledge question. I should answer directly without using any tools.\n</|synalux_think|>\n<|synalux_answer|>{answer}</|synalux_answer|>'
        msgs = []
        if _sys_prompt:
            msgs.append({"role": "system", "content": _sys_prompt})
        msgs.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ])
        data.append({"messages": msgs})
    
    return data

def main():
    # IMPORTANT: Do NOT duplicate rows with `data * N`. That destroys batch
    # variance and causes catastrophic memorization. Instead, rely on
    # mlx_lm.lora's --iters parameter for proper epoch-level repetition
    # with internal shuffling.
    data = generate_sft_data()
    random.shuffle(data)
    
    # Split 90/10
    split = int(len(data) * 0.9)
    train = data[:split]
    valid = data[split:]
    
    from config import AUX_DATA_DIR
    # R12-fix: Write to toolname_sft/ not AUX_DATA_DIR to avoid overwriting diverse SFT data
    out_dir = os.path.join(str(Path(AUX_DATA_DIR).parent), "toolname_sft")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    
    with open(f"{out_dir}/valid.jsonl", "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    
    print(f"Generated {len(data)} unique examples ({len(TOOL_NAME_EXAMPLES)} tool + {len(data) - len(TOOL_NAME_EXAMPLES)} reasoning)")
    print(f"  Train: {len(train)}, Valid: {len(valid)}")
    print(f"  NOTE: Use --iters in mlx_lm.lora for epoch repetition (not array multiplication)")
    print(f"  Saved to {out_dir}/train.jsonl and valid.jsonl")

if __name__ == "__main__":
    main()
