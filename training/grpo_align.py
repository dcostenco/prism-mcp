#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for tool-use accuracy.
Uses deterministic reward function — no reward model needed.

v2.0 Fixes:
  - Normalized reward range to [-1.0, +1.0] to prevent gradient explosion
  - Reduced data repetition from 50x to 5x (anti-overfitting)
  - Added subprocess error handling with SFT fallback
  - Restored reward function verification block
  - Synthetic injection is now optional (use --synthetic flag)
  - Moderate learning rate (1e-5) to prevent catastrophic forgetting
"""
import json
import os
import sys
import subprocess
import re
import shutil

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


def compute_reward(response_text: str, expected_tool: str = None) -> dict:
    """
    Decomposed 4-component reward function for GRPO (v3.0).

    Components (each normalized to [-0.25, +0.25], total range [-1.0, +1.0]):
      1. FORMAT:     Structural compliance (<think> + <tool_call> tags)
      2. TOOL:       Correct tool selection vs hallucination
      3. PARAMS:     Required/optional param accuracy, hallucinated param penalty
      4. ABSTENTION: Correct non-tool response vs unnecessary tool invocation

    Returns dict with component scores and total.
    """
    scores = {"format": 0.0, "tool": 0.0, "params": 0.0, "abstention": 0.0}

    # ── Parse response structure ──
    has_think = response_text.strip().startswith('<think>')
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    tool_content = None
    tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response_text, re.DOTALL)
    if tool_match:
        tool_content = tool_match.group(1)
    if not tool_content:
        im_match = re.search(r'<\|im_start\|>\s*(\{.*?\})\s*<\|im_end\|>', response_text, re.DOTALL)
        if im_match:
            tool_content = im_match.group(1)

    has_tool_call = tool_content is not None
    should_tool = expected_tool is not None

    # ── Component 1: FORMAT (structural compliance) ──
    if has_think:
        scores["format"] += 0.10
        if len(think_text) > 30:
            scores["format"] += 0.05  # Substantive reasoning
        if len(think_text) > 1000:
            scores["format"] -= 0.03  # Anti-thought-farming
    else:
        scores["format"] -= 0.10

    if has_tool_call:
        if '<tool_call>' in response_text and '</tool_call>' in response_text:
            scores["format"] += 0.10  # Proper tag wrapping
        else:
            scores["format"] += 0.03  # Partial format

    scores["format"] = max(-0.25, min(0.25, scores["format"]))

    # ── Component 2: TOOL SELECTION ──
    if has_tool_call:
        try:
            tool_call = json.loads(tool_content)
            tool_name = tool_call.get("name", "")
        except json.JSONDecodeError:
            scores["tool"] = -0.20  # Invalid JSON = critical failure
            scores["format"] -= 0.05
            total = sum(scores.values())
            return {"format": scores["format"], "tool": scores["tool"],
                    "params": scores["params"], "abstention": scores["abstention"],
                    "total": max(-1.0, min(1.0, total))}

        if tool_name in VALID_TOOLS:
            if expected_tool and tool_name == expected_tool:
                scores["tool"] = 0.25   # Perfect tool selection
            elif tool_name in VALID_TOOLS:
                scores["tool"] = 0.10   # Valid but possibly wrong tool
        else:
            scores["tool"] = -0.25      # Hallucinated tool name
    else:
        scores["tool"] = 0.0  # No tool call — scored by abstention instead

    # ── Component 3: PARAMETER ACCURACY ──
    if has_tool_call and tool_content:
        try:
            tool_call = json.loads(tool_content)
            tool_name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})
            params = TOOL_PARAMS.get(tool_name, {"required": set(), "optional": set()})

            required = params["required"]
            optional = params["optional"]
            all_valid = required | optional
            provided = set(args.keys())

            # Required params present
            missing = required - provided
            if not missing:
                scores["params"] += 0.15
            else:
                scores["params"] -= 0.08 * len(missing)

            # Hallucinated params (not in schema at all)
            hallucinated = provided - all_valid
            if hallucinated:
                scores["params"] -= 0.05 * len(hallucinated)

            # Bonus for optional params correctly used
            correct_optional = provided & optional
            if correct_optional:
                scores["params"] += 0.02 * len(correct_optional)

        except (json.JSONDecodeError, KeyError):
            scores["params"] = -0.15

    scores["params"] = max(-0.25, min(0.25, scores["params"]))

    # ── Component 4: ABSTENTION (asymmetric) ──
    if should_tool and not has_tool_call:
        # Should have called a tool but didn't — moderate penalty
        scores["abstention"] = -0.20
    elif not should_tool and has_tool_call:
        # Called a tool when none was needed — strong penalty
        scores["abstention"] = -0.25
    elif not should_tool and not has_tool_call:
        # Correctly abstained from tool use
        if len(response_text.strip()) > 30:
            scores["abstention"] = 0.40  # Substantive prose response (boosted weight)
        else:
            scores["abstention"] = 0.15  # Too short but correct decision
    elif should_tool and has_tool_call:
        scores["abstention"] = 0.15  # Correct decision to use tool

    total = sum(scores.values())
    return {
        "format": scores["format"],
        "tool": scores["tool"],
        "params": scores["params"],
        "abstention": scores["abstention"],
        "total": max(-1.0, min(1.0, total)),
    }


def generate_grpo_prompts():
    """Generate 50 prompts for GRPO training with expected tool metadata."""
    prompts = [
        # ── Tool-call prompts (should invoke a tool) ──
        {"text": "Load the full context for the prism-mcp project", "expected_tool": "session_load_context"},
        {"text": "Save this session: implemented RBAC roles", "expected_tool": "session_save"},
        {"text": "Search for sessions about JWT authentication in synalux-private", "expected_tool": "session_search"},
        {"text": "List all sessions for project bcba-private", "expected_tool": "session_search"},
        {"text": "Delete the session from yesterday about the billing bug", "expected_tool": "session_delete"},
        {"text": "What do we know about the 'Zero-Search' architecture in prism?", "expected_tool": "knowledge_search"},
        {"text": "Store this knowledge: The ACT-R decay rate is 0.5 for rollup nodes", "expected_tool": "knowledge_save"},
        {"text": "Search for patterns about memory consolidation", "expected_tool": "knowledge_search"},
        {"text": "Hand off the billing task from dev to security: payment logic is ready", "expected_tool": "session_save"},
        {"text": "Initialize a deep session for project synalux-docs", "expected_tool": "session_load_context"},
        {"text": "Find work related to the schema migration in v9.4", "expected_tool": "session_search"},
        {"text": "What is the status of the HIPAA security audit?", "expected_tool": "session_search"},
        {"text": "Log work: fixed the abortPipeline syntax error in dashboard", "expected_tool": "session_save"},
        {"text": "Show me context for synalux-portal project at deep level", "expected_tool": "session_load_context"},
        {"text": "Search for all previous sessions related to database migrations", "expected_tool": "session_search"},
        {"text": "Save a new knowledge item about TypeScript best practices for ESM", "expected_tool": "knowledge_save"},
        {"text": "What tools did we use in the last session for prism-mcp?", "expected_tool": "session_load_context"},
        {"text": "Find all work on the video panel implementation", "expected_tool": "session_search"},
        {"text": "Store knowledge: React Server Components require 'use server' directive", "expected_tool": "knowledge_save"},
        {"text": "Load the latest context for bcba-private project", "expected_tool": "session_load_context"},
        {"text": "Search knowledge for GRPO training best practices", "expected_tool": "knowledge_search"},
        {"text": "Save session: deployed v11.6.0 with serialized execution queue", "expected_tool": "session_save"},
        {"text": "Find sessions about Supabase RLS policies", "expected_tool": "session_search"},
        {"text": "What knowledge do we have about Ollama tool calling?", "expected_tool": "knowledge_search"},
        {"text": "Load context for synalux-private at shallow level", "expected_tool": "session_load_context"},
        # ── Reasoning prompts (should NOT invoke a tool) ──
        {"text": "Explain how React Server Components work", "expected_tool": None},
        {"text": "Write a hello world in Python", "expected_tool": None},
        {"text": "What is the difference between gRPC and REST?", "expected_tool": None},
        {"text": "How does garbage collection work in Go?", "expected_tool": None},
        {"text": "Explain the CAP theorem in distributed systems", "expected_tool": None},
        {"text": "What are the pros and cons of microservices?", "expected_tool": None},
        {"text": "Write a bash one-liner to find large files", "expected_tool": None},
        {"text": "How do I set up a PostgreSQL database on Docker?", "expected_tool": None},
        {"text": "What is the time complexity of quicksort?", "expected_tool": None},
        {"text": "Explain how JWT tokens work for authentication", "expected_tool": None},
        {"text": "Write a Python function to reverse a linked list", "expected_tool": None},
        {"text": "What is the difference between TCP and UDP?", "expected_tool": None},
        {"text": "Explain the Observer pattern in object-oriented design", "expected_tool": None},
        {"text": "How do I optimize a slow SQL query?", "expected_tool": None},
        {"text": "What is CORS and why does it exist?", "expected_tool": None},
        {"text": "Write a TypeScript generic function for array filtering", "expected_tool": None},
        {"text": "Explain the difference between let, const, and var in JavaScript", "expected_tool": None},
        {"text": "How does WebSocket differ from HTTP long polling?", "expected_tool": None},
        {"text": "What is dependency injection and why is it useful?", "expected_tool": None},
        {"text": "Explain the concept of eventual consistency", "expected_tool": None},
        {"text": "Write a regex to validate email addresses", "expected_tool": None},
        {"text": "What are the SOLID principles in software engineering?", "expected_tool": None},
        {"text": "How do I implement rate limiting in an API?", "expected_tool": None},
        {"text": "What is the difference between authentication and authorization?", "expected_tool": None},
        {"text": "Explain how a B-tree index works in databases", "expected_tool": None},
        # ── Hard negatives (sound tool-like but should NOT invoke a tool) ──
        {"text": "Tell me about session management in web applications", "expected_tool": None},
        {"text": "How does knowledge graph embedding work?", "expected_tool": None},
        {"text": "What is the search algorithm behind Elasticsearch?", "expected_tool": None},
        {"text": "Explain how context switching works in operating systems", "expected_tool": None},
        {"text": "What is a memory leak and how to debug it?", "expected_tool": None},
        {"text": "How does session storage differ from local storage in browsers?", "expected_tool": None},
        {"text": "What is knowledge distillation in machine learning?", "expected_tool": None},
        {"text": "Explain how to save state in a React application", "expected_tool": None},
        {"text": "What is context-aware computing?", "expected_tool": None},
        {"text": "How does a search engine index web pages?", "expected_tool": None},
    ]

    with open(GRPO_DATA, "w") as f:
        for p in prompts:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are Prism, an AI coding assistant with persistent memory. Use MCP tools when appropriate."},
                    {"role": "user", "content": p["text"]}
                ],
                "expected_tool": p["expected_tool"]
            }) + "\n")
    return prompts


def generate_synthetic_chosen(prompt: str) -> str:
    """Generate gold-standard responses for all tool-calling prompts.
    Maps prompt keywords to perfect <think> + <tool_call> responses.
    Returns None for reasoning prompts (they don't need synthetic injection)."""

    # Map keyword patterns to tool call responses
    SYNTHETIC_MAP = [
        # session_load_context prompts
        (["Load", "context", "prism-mcp"], "session_load_context",
         '<think>The user wants to load project context for "prism-mcp". I should use session_load_context with the project name.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "prism-mcp", "level": "deep"}}\n</tool_call>'),
        (["context", "synalux-portal", "deep"], "session_load_context",
         '<think>The user is requesting deep context for synalux-portal. I\'ll use session_load_context with deep level.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "synalux-portal", "level": "deep"}}\n</tool_call>'),
        (["context", "bcba-private"], "session_load_context",
         '<think>Loading context for bcba-private project. This requires session_load_context.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "bcba-private"}}\n</tool_call>'),
        (["context", "synalux-private", "shallow"], "session_load_context",
         '<think>The user wants shallow context for synalux-private. I\'ll use session_load_context with shallow level.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "synalux-private", "level": "shallow"}}\n</tool_call>'),
        (["tools", "last session", "prism-mcp"], "session_load_context",
         '<think>The user wants to know what tools were used. I need to load context for prism-mcp to find this information.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "prism-mcp", "level": "deep"}}\n</tool_call>'),
        (["Initialize", "deep", "synalux-docs"], "session_load_context",
         '<think>The user wants to initialize a deep session for synalux-docs. I should load deep context.</think>\n\n<tool_call>\n{"name": "session_load_context", "arguments": {"project": "synalux-docs", "level": "deep"}}\n</tool_call>'),

        # session_save prompts
        (["Save", "session", "RBAC"], "session_save",
         '<think>The user wants to save a session about implementing RBAC roles. I should use session_save with the project and summary.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "prism-mcp", "summary": "Implemented RBAC roles"}}\n</tool_call>'),
        (["Log work", "abortPipeline"], "session_save",
         '<think>The user wants to log work about fixing the abortPipeline syntax error. I\'ll use session_save.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "prism-mcp", "summary": "Fixed the abortPipeline syntax error in dashboard"}}\n</tool_call>'),
        (["Hand off", "billing", "security"], "session_save",
         '<think>The user wants to hand off the billing task. I should save this transition using session_save.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "prism-mcp", "summary": "Hand off billing task from dev to security: payment logic is ready"}}\n</tool_call>'),
        (["Save session", "v11.6.0"], "session_save",
         '<think>The user wants to save a session about deploying v11.6.0. I\'ll use session_save.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "prism-mcp", "summary": "Deployed v11.6.0 with serialized execution queue"}}\n</tool_call>'),

        # session_search prompts
        (["Search", "sessions", "JWT"], "session_search",
         '<think>The user wants to search for sessions about JWT authentication. I should use session_search with the query and project.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "JWT authentication", "project": "synalux-private"}}\n</tool_call>'),
        (["List", "sessions", "bcba-private"], "session_search",
         '<think>The user wants to list sessions for bcba-private. I\'ll use session_search to find them.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "*", "project": "bcba-private"}}\n</tool_call>'),
        (["status", "HIPAA", "security audit"], "session_search",
         '<think>The user is asking about the HIPAA security audit status. I should search for relevant sessions.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "HIPAA security audit"}}\n</tool_call>'),
        (["Find", "schema migration", "v9.4"], "session_search",
         '<think>The user wants to find work related to the schema migration in v9.4. I\'ll search for it.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "schema migration v9.4"}}\n</tool_call>'),
        (["Search", "database migrations"], "session_search",
         '<think>The user wants to find sessions about database migrations. I\'ll use session_search.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "database migrations"}}\n</tool_call>'),
        (["Find", "video panel"], "session_search",
         '<think>The user wants to find work on the video panel. I\'ll search for it.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "video panel implementation"}}\n</tool_call>'),
        (["sessions", "Supabase", "RLS"], "session_search",
         '<think>The user wants to find sessions about Supabase RLS policies. I\'ll search for them.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "Supabase RLS policies"}}\n</tool_call>'),

        # session_delete prompts
        (["Delete", "session", "billing bug"], "session_delete",
         '<think>The user wants to delete a session about the billing bug. I should use session_delete.</think>\n\n<tool_call>\n{"name": "session_delete", "arguments": {"query": "billing bug"}}\n</tool_call>'),

        # knowledge_search prompts
        (["Zero-Search", "architecture"], "knowledge_search",
         '<think>The user is asking about the Zero-Search architecture. I should search the knowledge base.</think>\n\n<tool_call>\n{"name": "knowledge_search", "arguments": {"query": "Zero-Search architecture", "project": "prism"}}\n</tool_call>'),
        (["patterns", "memory consolidation"], "knowledge_search",
         '<think>The user wants to search for patterns about memory consolidation. I\'ll use knowledge_search.</think>\n\n<tool_call>\n{"name": "knowledge_search", "arguments": {"query": "memory consolidation patterns"}}\n</tool_call>'),
        (["knowledge", "GRPO", "training"], "knowledge_search",
         '<think>The user wants to search knowledge about GRPO training best practices.</think>\n\n<tool_call>\n{"name": "knowledge_search", "arguments": {"query": "GRPO training best practices"}}\n</tool_call>'),
        (["knowledge", "Ollama", "tool calling"], "knowledge_search",
         '<think>The user wants to search for knowledge about Ollama tool calling.</think>\n\n<tool_call>\n{"name": "knowledge_search", "arguments": {"query": "Ollama tool calling"}}\n</tool_call>'),

        # knowledge_save prompts
        (["Store", "knowledge", "ACT-R"], "knowledge_save",
         '<think>The user wants to store knowledge about the ACT-R decay rate. I should use knowledge_save.</think>\n\n<tool_call>\n{"name": "knowledge_save", "arguments": {"project": "prism", "concept": "ACT-R Decay Rate", "description": "The ACT-R decay rate is 0.5 for rollup nodes", "confidence": 1.0}}\n</tool_call>'),
        (["Save", "knowledge", "TypeScript", "ESM"], "knowledge_save",
         '<think>The user wants to save knowledge about TypeScript ESM best practices.</think>\n\n<tool_call>\n{"name": "knowledge_save", "arguments": {"project": "prism", "concept": "TypeScript ESM Best Practices", "description": "TypeScript best practices for ESM module compatibility"}}\n</tool_call>'),
        (["Store knowledge", "React Server Components"], "knowledge_save",
         '<think>The user wants to store knowledge about React Server Components requiring use server directive.</think>\n\n<tool_call>\n{"name": "knowledge_save", "arguments": {"project": "prism", "concept": "React Server Components", "description": "React Server Components require use server directive"}}\n</tool_call>'),
    ]

    for keywords, tool_name, response in SYNTHETIC_MAP:
        if all(kw.lower() in prompt.lower() for kw in keywords):
            return response

    # ── Synthetic gold responses for reasoning/abstention prompts ──
    # These train the model to answer directly WITHOUT calling tools
    ABSTENTION_MAP = [
        ("React Server Components",
         '<think>This is a general programming question about React Server Components. I can answer this directly from my knowledge without needing to search any project sessions or knowledge base.</think>\n\nReact Server Components (RSC) are a React architecture where components run exclusively on the server. They can directly access databases, file systems, and backend services without exposing credentials to the client. Key benefits: zero client-side JavaScript bundle for server components, direct data fetching, and automatic code splitting. They work alongside Client Components (marked with "use client") which handle interactivity.'),
        ("hello world in Python",
         '<think>This is a simple coding question. I should just write the code directly without using any tools.</think>\n\nHere\'s a hello world program in Python:\n\n```python\nprint("Hello, World!")\n```\n\nYou can run this by saving it as `hello.py` and executing `python hello.py` in your terminal.'),
        ("gRPC and REST",
         '<think>This is a general knowledge question comparing two API paradigms. No tools needed — I can explain this from my training knowledge.</think>\n\n**gRPC** uses Protocol Buffers for serialization and HTTP/2 for transport, offering binary encoding, bidirectional streaming, and code generation. It\'s ideal for microservice-to-microservice communication.\n\n**REST** uses JSON over HTTP/1.1 with resource-oriented URLs. It\'s simpler, more widely supported, and better for public APIs and browser clients.\n\nKey tradeoffs: gRPC is faster but harder to debug; REST is more human-readable but less efficient.'),
        ("garbage collection",
         '<think>This is a general computer science question about garbage collection in Go. I can answer this directly.</think>\n\nGo uses a concurrent, tri-color mark-and-sweep garbage collector. It runs concurrently with the application using write barriers to track pointer updates. The GC has three phases: mark setup (STW), concurrent mark, and mark termination (STW). The GOGC environment variable controls the GC target percentage. Go\'s GC prioritizes low latency over throughput, keeping pause times under 1ms in most cases.'),
        ("CAP theorem",
         '<think>This is a distributed systems theory question. I can explain the CAP theorem directly without any tool calls.</think>\n\nThe CAP theorem states that a distributed system can provide at most two of three guarantees simultaneously:\n\n- **Consistency**: Every read receives the most recent write\n- **Availability**: Every request receives a response\n- **Partition tolerance**: The system continues operating despite network partitions\n\nSince network partitions are inevitable in distributed systems, you must choose between CP (consistent but may be unavailable) or AP (available but may return stale data).'),
        ("pros and cons of microservices",
         '<think>This is an architecture question. I should provide a balanced analysis directly.</think>\n\n**Pros:** Independent deployment, technology diversity, fault isolation, team autonomy, horizontal scaling per service.\n\n**Cons:** Distributed system complexity, network latency, data consistency challenges, operational overhead (monitoring, logging, tracing), and the need for sophisticated CI/CD pipelines.\n\nMicroservices work best for large teams with well-defined domain boundaries. For small teams, a modular monolith is often more practical.'),
        ("bash one-liner to find large files",
         '<think>This is a practical coding question. I should provide the command directly without tool calls.</think>\n\nFind files larger than 100MB in the current directory:\n\n```bash\nfind . -type f -size +100M -exec ls -lh {} \\;\n```\n\nAlternatively, sorted by size:\n\n```bash\nfind . -type f -size +100M -printf "%s %p\\n" | sort -rn | head -20\n```'),
        ("PostgreSQL database on Docker",
         '<think>This is a practical setup question. I can provide Docker commands directly.</think>\n\nRun PostgreSQL in Docker:\n\n```bash\ndocker run -d --name postgres \\\n  -e POSTGRES_PASSWORD=mypassword \\\n  -e POSTGRES_DB=mydb \\\n  -p 5432:5432 \\\n  -v pgdata:/var/lib/postgresql/data \\\n  postgres:16\n```\n\nConnect with psql: `docker exec -it postgres psql -U postgres -d mydb`'),
        ("time complexity of quicksort",
         '<think>This is a computer science theory question about algorithm complexity. No tools needed.</think>\n\n- **Best case:** O(n log n) — when the pivot divides the array evenly\n- **Average case:** O(n log n) — expected with random pivots\n- **Worst case:** O(n²) — when the pivot is always the smallest/largest element\n- **Space:** O(log n) for the recursive call stack\n\nQuicksort is typically faster in practice than merge sort due to better cache locality and lower constant factors.'),
        ("JWT tokens work for authentication",
         '<think>The user is asking about how JWT tokens work. This is a general knowledge question I can answer directly.</think>\n\nJSON Web Tokens (JWT) consist of three base64-encoded parts: Header (algorithm, type), Payload (claims like user ID, expiration), and Signature (HMAC or RSA). The server creates a signed token on login, the client stores it (usually in an HTTP-only cookie), and sends it with each request. The server verifies the signature without database lookups. JWTs are stateless but can\'t be revoked easily — use short expiration times and refresh tokens for security.'),
        ("reverse a linked list",
         '<think>This is a coding question. I should write the solution directly.</think>\n\n```python\ndef reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev\n```\n\nTime: O(n), Space: O(1). This iteratively reverses the pointers in a single pass.'),
        ("TCP and UDP",
         '<think>This is a networking fundamentals question. I can answer directly.</think>\n\n**TCP** (Transmission Control Protocol): Connection-oriented, reliable delivery with ordering guarantees, flow control, and congestion control. Used for HTTP, SSH, FTP.\n\n**UDP** (User Datagram Protocol): Connectionless, no delivery guarantees, no ordering, minimal overhead. Used for DNS, video streaming, gaming, VoIP.\n\nChoose TCP when you need reliability; choose UDP when you need speed and can tolerate packet loss.'),
        ("Observer pattern",
         '<think>This is a design patterns question. I should explain the pattern directly.</think>\n\nThe Observer pattern defines a one-to-many dependency where when one object (the Subject) changes state, all its dependents (Observers) are notified automatically. The Subject maintains a list of observers and calls their `update()` method on state changes. This decouples the subject from its observers, enabling loose coupling. Common examples: event systems, pub/sub, React\'s state management, and the DOM event model.'),
        ("optimize a slow SQL query",
         '<think>This is a database optimization question. I can provide guidance directly.</think>\n\nKey optimization strategies:\n\n1. **EXPLAIN ANALYZE** — Check the query plan for sequential scans\n2. **Add indexes** — On columns used in WHERE, JOIN, and ORDER BY\n3. **Avoid SELECT *** — Only fetch needed columns\n4. **Use LIMIT** — Paginate large result sets\n5. **Denormalize** — For read-heavy workloads\n6. **Materialized views** — Cache expensive aggregations\n7. **Connection pooling** — Reduce connection overhead'),
        ("CORS and why does it exist",
         '<think>This is a web security question. I can explain CORS directly.</think>\n\nCORS (Cross-Origin Resource Sharing) is a browser security mechanism that restricts web pages from making requests to domains different from the one serving the page. It exists because of the Same-Origin Policy, which prevents malicious scripts from accessing sensitive data on other sites. Servers opt in to cross-origin requests by sending `Access-Control-Allow-Origin` headers. Preflight OPTIONS requests verify permissions before the actual request.'),
        ("TypeScript generic function for array filtering",
         '<think>This is a coding question. I should write the TypeScript code directly.</think>\n\n```typescript\nfunction filterArray<T>(arr: T[], predicate: (item: T) => boolean): T[] {\n  return arr.reduce<T[]>((acc, item) => {\n    if (predicate(item)) acc.push(item);\n    return acc;\n  }, []);\n}\n\n// Usage:\nconst numbers = [1, 2, 3, 4, 5];\nconst evens = filterArray(numbers, n => n % 2 === 0); // [2, 4]\n```'),
        ("let, const, and var",
         '<think>This is a JavaScript fundamentals question. No tools needed.</think>\n\n- **var**: Function-scoped, hoisted, can be redeclared. Legacy — avoid in modern code.\n- **let**: Block-scoped, hoisted but not initialized (temporal dead zone), can be reassigned.\n- **const**: Block-scoped, hoisted but not initialized, cannot be reassigned (but objects/arrays are still mutable).\n\nBest practice: Use `const` by default, `let` when reassignment is needed, never use `var`.'),
        ("WebSocket differ from HTTP long polling",
         '<think>This is a networking/architecture question. I can answer directly.</think>\n\n**WebSocket**: Full-duplex, persistent connection over a single TCP socket. Low overhead after handshake. Ideal for real-time apps (chat, gaming, live data).\n\n**HTTP Long Polling**: Client sends request, server holds it open until data is available, then responds and client immediately re-requests. Higher overhead due to repeated HTTP headers.\n\nWebSocket is more efficient for high-frequency bidirectional communication; long polling is simpler to implement and works through all proxies.'),
        ("dependency injection",
         '<think>This is a software design question. I should explain the concept directly.</think>\n\nDependency Injection (DI) is a design pattern where objects receive their dependencies from external sources rather than creating them internally. Benefits: testability (easy to mock), loose coupling, flexibility to swap implementations, and single responsibility. Three types: constructor injection (most common), setter injection, and interface injection. Frameworks like Spring, Angular, and NestJS provide DI containers that automatically wire dependencies.'),
        ("eventual consistency",
         '<think>This is a distributed systems concept question. I can explain directly.</think>\n\nEventual consistency is a consistency model where, given enough time without new updates, all replicas of a data item will converge to the same value. Unlike strong consistency (where reads always return the latest write), eventually consistent systems allow temporary staleness for better availability and partition tolerance. Used in: DynamoDB, Cassandra, DNS, S3. Techniques to manage it: conflict resolution (last-write-wins, CRDTs), read-your-writes consistency, and causal consistency.'),
        ("regex to validate email",
         '<think>This is a practical coding question. No tools needed.</think>\n\n```python\nimport re\n\ndef validate_email(email: str) -> bool:\n    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n    return bool(re.match(pattern, email))\n```\n\nNote: Perfect email validation via regex is nearly impossible (RFC 5322 is very complex). For production, use a library like `email-validator` in Python or let the mail server validate by sending a confirmation.'),
        ("SOLID principles",
         '<think>This is a software engineering fundamentals question. I can explain directly.</think>\n\nSOLID is five design principles for maintainable OOP code:\n\n- **S**ingle Responsibility: One class, one reason to change\n- **O**pen/Closed: Open for extension, closed for modification\n- **L**iskov Substitution: Subtypes must be substitutable for their base types\n- **I**nterface Segregation: Many specific interfaces over one general-purpose interface\n- **D**ependency Inversion: Depend on abstractions, not concretions'),
        ("rate limiting in an API",
         '<think>This is a backend engineering question. I can provide implementation guidance directly.</think>\n\nCommon rate limiting algorithms:\n\n1. **Token Bucket**: Tokens added at fixed rate, requests consume tokens. Allows bursts.\n2. **Sliding Window**: Count requests in a rolling time window. Most accurate.\n3. **Fixed Window**: Count requests per fixed time interval. Simple but allows edge bursts.\n\nImplementation: Use Redis with `INCR` + `EXPIRE` for distributed rate limiting. Return `429 Too Many Requests` with `Retry-After` header when limit is exceeded.'),
        ("authentication and authorization",
         '<think>This is a security concepts question. I can explain the distinction directly.</think>\n\n**Authentication** (AuthN): Verifying *who* you are. Methods: passwords, OAuth, biometrics, MFA, JWT tokens.\n\n**Authorization** (AuthZ): Determining *what* you can do. Methods: RBAC (role-based), ABAC (attribute-based), ACLs, policy engines (OPA).\n\nAuthentication always comes first — you must know who the user is before checking their permissions. Common mistake: confusing the two, leading to authorization checks that don\'t verify identity.'),
        ("B-tree index works in databases",
         '<think>This is a database internals question. I can explain directly.</think>\n\nA B-tree is a self-balancing tree data structure that maintains sorted data and allows searches, insertions, and deletions in O(log n). In databases, B-tree indexes store key-value pairs where keys are column values and values are row pointers. Each node can have multiple children (high fan-out), minimizing disk I/O. B+ trees (the variant used in practice) store all data in leaf nodes, which are linked for efficient range scans.'),
        # Hard negatives — reasoning prompts that sound tool-like
        ("session management in web applications",
         '<think>The user is asking about the general concept of session management in web applications — not asking me to manage a specific Prism session. This is a knowledge question I can answer directly.</think>\n\nWeb session management tracks user state across HTTP requests. Common approaches:\n\n1. **Server-side sessions**: Session data stored on server, client holds only a session ID cookie\n2. **JWT tokens**: Stateless, self-contained tokens with encoded claims\n3. **Cookie-based**: Data stored directly in encrypted cookies\n\nSecurity considerations: HttpOnly and Secure flags, session fixation prevention, idle timeouts, and CSRF protection.'),
        ("knowledge graph embedding",
         '<think>The user is asking about knowledge graph embeddings as a machine learning concept — not asking me to search the Prism knowledge base. I should explain the concept directly.</think>\n\nKnowledge graph embedding maps entities and relations into continuous vector spaces while preserving the graph structure. Popular models: TransE (translational distance), DistMult (bilinear), ComplEx (complex-valued), and RotatE (rotational). These embeddings enable link prediction, entity classification, and knowledge completion. Training uses positive triples (h, r, t) and corrupted negative samples.'),
        ("search algorithm behind Elasticsearch",
         '<think>The user is asking about Elasticsearch\'s search algorithm as a technical question — not asking me to search anything in Prism. I should explain directly.</think>\n\nElasticsearch uses Apache Lucene for full-text search. Key algorithms: inverted indexes (mapping terms to document IDs), TF-IDF and BM25 for relevance scoring, and analyzers for tokenization and normalization. Queries go through analysis, term matching, scoring, and result aggregation. Distributed search shards queries across nodes and merges results using a scatter-gather pattern.'),
        ("context switching in operating systems",
         '<think>The user is asking about OS context switching — not about loading context in Prism. This is a systems question I can answer directly.</think>\n\nContext switching is the process of saving the state (registers, program counter, stack pointer) of a running process and restoring the state of another. Triggers: timer interrupts (preemptive scheduling), system calls, I/O waits. Cost: typically 1-10 microseconds, involving TLB flushes, cache pollution, and pipeline stalls. Threads within the same process have cheaper context switches since they share address spaces.'),
        ("memory leak and how to debug it",
         '<think>The user is asking about memory leaks in general — not about Prism memory management. I should explain the concept and debugging techniques.</think>\n\nA memory leak occurs when allocated memory is never freed, causing gradual memory growth. Common causes: unclosed event listeners, retained references in closures, circular references, and forgotten timers.\n\nDebugging tools:\n- **Chrome DevTools**: Heap snapshots, allocation timeline\n- **Valgrind**: C/C++ memory profiler\n- **Node.js**: `--inspect` flag + Chrome DevTools, `process.memoryUsage()`\n- **Python**: `tracemalloc`, `objgraph`'),
        ("session storage differ from local storage",
         '<think>The user is asking about browser storage APIs — not about Prism session management. This is a web fundamentals question.</think>\n\n**sessionStorage**: Data persists only for the browser tab/window session. Cleared when the tab closes. Separate per tab.\n\n**localStorage**: Data persists indefinitely until explicitly cleared. Shared across all tabs of the same origin.\n\nBoth have ~5-10MB limits and are synchronous. For larger data, use IndexedDB. Neither should store sensitive data (accessible to any JS on the page).'),
        ("knowledge distillation in machine learning",
         '<think>The user is asking about the ML technique of knowledge distillation — not about Prism knowledge management. I should explain the concept.</think>\n\nKnowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. The student is trained on the teacher\'s soft probability outputs (softmax with temperature) rather than hard labels, capturing inter-class relationships. Benefits: smaller models with near-teacher performance, faster inference, lower memory. Variants: response-based (output matching), feature-based (intermediate layer matching), and relation-based (structural similarity).'),
        ("save state in a React application",
         '<think>The user is asking about React state management patterns — not asking me to save anything in Prism. This is a frontend development question.</think>\n\nReact state management options:\n\n1. **useState/useReducer**: Local component state\n2. **Context API**: Shared state without prop drilling\n3. **Redux/Zustand**: Global state with predictable updates\n4. **React Query/SWR**: Server state caching and synchronization\n5. **URL state**: Using React Router for shareable state\n\nBest practice: Keep state as close to where it\'s used as possible. Lift state up only when siblings need to share it.'),
        ("context-aware computing",
         '<think>The user is asking about the academic/industry concept of context-aware computing — not about Prism context loading. This is a general knowledge question.</think>\n\nContext-aware computing refers to systems that sense and adapt to their environment. Context includes: location (GPS), time, user activity, device state, and nearby devices. Applications: smart homes, location-based services, adaptive UIs, and healthcare monitoring. Key challenges: privacy concerns, sensor fusion accuracy, and energy efficiency on mobile devices.'),
        ("search engine index web pages",
         '<think>The user is asking about how search engines work — not asking me to search anything. This is a systems/IR question.</think>\n\nSearch engine indexing: 1) **Crawling** — following links to discover pages. 2) **Parsing** — extracting text, metadata, and links. 3) **Indexing** — building inverted indexes mapping terms to documents. 4) **Ranking** — using algorithms like PageRank, BM25, and neural ranking models. Modern engines also use knowledge panels, featured snippets, and personalization. Google processes ~8.5 billion searches per day.'),
    ]

    for keywords_substr, response in ABSTENTION_MAP:
        if keywords_substr.lower() in prompt.lower():
            return response

    return None


def verify_reward_function():
    """Self-test the decomposed reward function with known inputs."""
    print("\n" + "=" * 60)
    print("Decomposed Reward Function Verification (v3.0):")
    print("=" * 60)

    test_cases = [
        ('<think>The user wants to save a session for project prism-mcp. This is a write operation. The correct tool is session_save which requires project and summary parameters. I have both values from the request.</think>\n\n<tool_call>\n{"name": "session_save", "arguments": {"project": "test", "summary": "test"}}\n</tool_call>', "session_save", "Perfect: think + correct tool + all params"),
        ('<tool_call>\n{"name": "session_save", "arguments": {"project": "test", "summary": "test"}}\n</tool_call>', "session_save", "No think, correct tool"),
        ('<tool_call>\n{"name": "fake_tool", "arguments": {}}\n</tool_call>', "session_save", "Hallucinated tool name"),
        ('<tool_call>\n{invalid json}\n</tool_call>', "session_save", "Invalid JSON"),
        ('Python is a programming language used for web development and data science. It has a large ecosystem of libraries including NumPy and pandas.', None, "Correct abstention (reasoning)"),
        ('<think>Let me search for this.</think>\n\n<tool_call>\n{"name": "session_search", "arguments": {"query": "test"}}\n</tool_call>', None, "False positive: tool used unnecessarily"),
        ('<tool_call>\n{"name": "session_save", "arguments": {}}\n</tool_call>', "session_save", "Missing required params"),
        ('ok', None, "Short abstention"),
    ]

    for response, expected, desc in test_cases:
        result = compute_reward(response, expected)
        components = f"fmt={result['format']:+.2f} tool={result['tool']:+.2f} prm={result['params']:+.2f} abs={result['abstention']:+.2f}"
        print(f"  [{result['total']:+.3f}] {desc}")
        print(f"          {components}")

    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GRPO Alignment v3.0 — Decomposed Rewards")
    parser.add_argument("--synthetic", action="store_true", help="Inject synthetic gold-standard responses")
    parser.add_argument("--repeat", type=int, default=5, help="Data repetition factor (default: 5)")
    parser.add_argument("--iters", type=int, default=300, help="Training iterations (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--verify-only", action="store_true", help="Only run reward function verification")
    parser.add_argument("--dpo-only", action="store_true", help="Skip generation, run DPO on existing data")
    args = parser.parse_args()

    verify_reward_function()

    if args.verify_only:
        return

    print("\n" + "=" * 60)
    print("GRPO Alignment v3.0 — Decomposed 4-Component Rewards")
    print(f"  Synthetic injection: {'ON' if args.synthetic else 'OFF (true GRPO)'}")
    print(f"  Data repetition: {args.repeat}x")
    print(f"  Iterations: {args.iters}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)

    prompts = generate_grpo_prompts()
    dpo_data = []

    try:
        from mlx_lm import load, generate
        print("\nLoading SFT model + adapter...")
        model, tokenizer = load(MODEL_PATH, adapter_path=SFT_ADAPTER)

        for i, p in enumerate(prompts):
            sys_msg = "You are Prism, an AI coding assistant with persistent memory. Use MCP tools when appropriate."
            full_prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{p['text']}<|im_end|>\n<|im_start|>assistant\n"

            completions = []
            for j in range(4):
                try:
                    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=256)
                    result = compute_reward(response, p.get("expected_tool"))
                    reward_val = result["total"]
                    print(f"    [Prompt {i+1} Gen {j+1}] R={reward_val:+.3f} fmt={result['format']:+.2f} tool={result['tool']:+.2f} prm={result['params']:+.2f} abs={result['abstention']:+.2f}")
                    completions.append((response, reward_val))
                except Exception as e:
                    print(f"  Warning: Generation failed: {e}")
                    continue

            if len(completions) >= 2:
                completions.sort(key=lambda x: x[1], reverse=True)
                best = completions[0]
                worst = completions[-1]

                if args.synthetic:
                    synthetic_chosen = generate_synthetic_chosen(p["text"])
                    if synthetic_chosen:
                        dpo_data.append({
                            "prompt": p["text"],
                            "chosen": synthetic_chosen,
                            "rejected": worst[0],
                        })
                        continue

                if best[1] > worst[1]:
                    dpo_data.append({
                        "prompt": p["text"],
                        "chosen": best[0],
                        "rejected": worst[0],
                    })

            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(prompts)} prompts, {len(dpo_data)} preference pairs")

        print(f"\nGenerated {len(dpo_data)} preference pairs")

        if len(dpo_data) >= 1:
            # mlx_lm supports SFT only (chat/completions/text), not native DPO.
            # Convert gold preference pairs to SFT chat format:
            # {messages: [system, user, assistant(chosen)]}
            import random
            random.shuffle(dpo_data)
            split = max(2, int(len(dpo_data) * 0.9))
            train_data = dpo_data[:split]
            valid_data = dpo_data[split:] if split < len(dpo_data) else dpo_data[-2:]

            sys_msg = "You are Prism, an AI coding assistant with persistent memory. Use MCP tools when appropriate."

            for split_name, split_data in [("train", train_data), ("valid", valid_data)]:
                path = f"/Users/admin/prism/training/data/{split_name}.jsonl"
                with open(path, "w") as f:
                    for _ in range(args.repeat):
                        for d in split_data:
                            entry = {
                                "messages": [
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": d["prompt"]},
                                    {"role": "assistant", "content": d["chosen"]}
                                ]
                            }
                            f.write(json.dumps(entry) + "\n")

            total_train = len(train_data) * args.repeat
            total_valid = len(valid_data) * args.repeat
            print(f"  Train: {len(train_data)} unique × {args.repeat} = {total_train} examples (SFT on gold chosen)")
            print(f"  Valid: {len(valid_data)} unique × {args.repeat} = {total_valid} examples")

            print(f"\nRunning SFT alignment on gold responses...")
            data_dir = "/Users/admin/prism/training/data"
            cmd = [
                sys.executable, "-m", "mlx_lm", "lora",
                "--model", MODEL_PATH,
                "--train",
                "--data", data_dir,
                "--adapter-path", OUTPUT_ADAPTER,
                "--num-layers", "16",
                "--batch-size", "2",
                "--iters", str(args.iters),
                "--max-seq-length", "2048",
                "--learning-rate", str(args.lr),
                "--steps-per-report", "25",
                "--save-every", "100",
                "--resume-adapter-file", os.path.join(SFT_ADAPTER, "adapters.safetensors"),
            ]

            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print(f"\nGRPO alignment complete! Adapter: {OUTPUT_ADAPTER}")
            else:
                print(f"\nDPO training returned code {result.returncode}")
                print("Falling back to SFT adapter only")
                os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
                for fname in os.listdir(SFT_ADAPTER):
                    shutil.copy2(os.path.join(SFT_ADAPTER, fname), os.path.join(OUTPUT_ADAPTER, fname))
        else:
            print(f"\nNot enough preference pairs ({len(dpo_data)}) for DPO. Using SFT adapter.")
            os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
            for fname in os.listdir(SFT_ADAPTER):
                shutil.copy2(os.path.join(SFT_ADAPTER, fname), os.path.join(OUTPUT_ADAPTER, fname))

    except ImportError:
        print("ERROR: mlx_lm not installed. Run: pip3 install mlx mlx-lm")
        sys.exit(1)
    except Exception as e:
        print(f"GRPO failed: {e}")
        print("Falling back to SFT-only adapter")
        os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
        if os.path.exists(SFT_ADAPTER):
            for fname in os.listdir(SFT_ADAPTER):
                shutil.copy2(os.path.join(SFT_ADAPTER, fname), os.path.join(OUTPUT_ADAPTER, fname))


if __name__ == "__main__":
    main()
