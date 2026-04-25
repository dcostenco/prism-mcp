#!/usr/bin/env python3
"""
Diverse SFT Data Generator — Fixes Generalization Failure

The previous SFT used 1-3 phrasings per tool, causing the model to 
memorize exact prompt patterns. This generator creates 15-20 diverse 
phrasings per tool, including:
  - Formal requests
  - Casual/conversational
  - Abbreviated commands
  - Indirect/implied intent
  - Context-rich with extra detail
"""
import json
import random

TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|tool_call_end|>"

# === DIVERSE PHRASINGS PER TOOL ===
# Each tool has 15-20 varied prompts with different styles

TOOL_PROMPTS = {
    "session_load_context": {
        "default_args": {"project": "prism-mcp", "level": "deep", "toolAction": "Loading context", "toolSummary": "Loading project context"},
        "prompts": [
            ("Load context for prism-mcp project", {"project": "prism-mcp"}),
            ("Pull up the context for synalux-portal", {"project": "synalux-portal"}),
            ("Hey, start a new session. Show me where we left off on prism-mcp.", {"project": "prism-mcp"}),
            ("Initialize session context for bcba-private", {"project": "bcba-private"}),
            ("What was our last state on the synalux project?", {"project": "synalux-portal"}),
            ("Resume work on prism-mcp", {"project": "prism-mcp"}),
            ("Show me what we were working on last time", {"project": "prism-mcp"}),
            ("Boot up the prism-mcp context please", {"project": "prism-mcp"}),
            ("I want to continue where we left off on synalux-portal", {"project": "synalux-portal"}),
            ("Get me up to speed on the bcba-private project", {"project": "bcba-private"}),
            ("Restore the previous session state", {"project": "prism-mcp"}),
            ("Pull up everything we had on the synalux docs", {"project": "synalux-docs"}),
            ("What's the current state of prism-mcp?", {"project": "prism-mcp"}),
            ("Load context.", {"project": "prism-mcp"}),
            ("Start session.", {"project": "prism-mcp"}),
        ]
    },
    "session_save_ledger": {
        "default_args": {"project": "prism-mcp", "conversation_id": "current", "summary": "Session work completed"},
        "prompts": [
            ("Save this session: fixed the OAuth bug in the portal", {"summary": "Fixed the OAuth bug in the portal"}),
            ("Can you jot down what we accomplished? We rewrote the webhook handler.", {"summary": "Rewrote the webhook handler"}),
            ("Record this work: migrated Stripe webhooks to v2 API", {"summary": "Migrated Stripe webhooks to v2 API"}),
            ("Log what we did today on the billing module", {"summary": "Billing module work completed"}),
            ("Save our progress", {"summary": "Session progress saved"}),
            ("Write down what we've done — fixed three test failures and refactored the auth layer", {"summary": "Fixed three test failures and refactored the auth layer"}),
            ("Commit this session to the ledger please", {"summary": "Session work logged"}),
            ("Let's wrap up. Save everything we did.", {"summary": "Session wrapped up and saved"}),
            ("Mark this session as done. We deployed the new cache layer.", {"summary": "Deployed the new cache layer"}),
            ("Note: we finished the database migration today", {"summary": "Finished the database migration"}),
            ("Save session notes: resolved the CORS issue on staging", {"summary": "Resolved the CORS issue on staging"}),
            ("Time to call it a day. Record what we did.", {"summary": "End of day session recorded"}),
            ("Log this — we implemented the rate limiter", {"summary": "Implemented the rate limiter"}),
            ("Don't forget to save that we fixed the memory leak", {"summary": "Fixed the memory leak"}),
            # Targeted: short/ambiguous save commands → always session_save_ledger
            ("Save.", {"summary": "Session saved"}),
            ("Save session.", {"summary": "Session saved"}),
            ("Save this.", {"summary": "Session work saved"}),
            ("Save what we did.", {"summary": "Session work saved"}),
            ("Record this.", {"summary": "Session recorded"}),
            ("Log it.", {"summary": "Session logged"}),
        ]
    },
    "session_search_memory": {
        "default_args": {"query": "search query"},
        "prompts": [
            ("What did we work on last week related to billing?", {"query": "billing work last week"}),
            ("Remind me — did we ever decide between Redis and Memcached for the session store?", {"query": "Redis vs Memcached session store decision"}),
            ("Find what we decided about the database migration strategy", {"query": "database migration strategy decision"}),
            ("Search for anything we discussed about the authentication overhaul", {"query": "authentication overhaul discussion"}),
            ("Look up past work on the OAuth2 refresh flow", {"query": "OAuth2 refresh flow implementation"}),
            ("What have we done before regarding Kubernetes pod scaling?", {"query": "Kubernetes pod scaling"}),
            ("Did we ever implement retry logic for the webhook system?", {"query": "webhook retry logic implementation"}),
            ("Search memory for any notes about the CI/CD pipeline", {"query": "CI/CD pipeline notes"}),
            ("Find all our past notes about the billing API redesign", {"query": "billing API redesign notes"}),
            ("Have we discussed rate limiting before?", {"query": "rate limiting discussion"}),
            ("What did we conclude about the caching strategy?", {"query": "caching strategy conclusion"}),
            ("Look up our previous work on error handling", {"query": "error handling work"}),
            ("I remember we discussed something about edge functions. What was it?", {"query": "edge functions discussion"}),
            ("Search for deployment-related decisions from last month", {"query": "deployment decisions last month"}),
            ("Any past work on the notification system?", {"query": "notification system past work"}),
        ]
    },
    "session_save_handoff": {
        "default_args": {"project": "prism-mcp"},
        "prompts": [
            ("Create a handoff for the synalux-portal project", {"project": "synalux-portal"}),
            ("I'm handing this off to the night shift. Save the state for prism-mcp.", {"project": "prism-mcp"}),
            ("Save handoff notes for the next session", {"project": "prism-mcp"}),
            ("Prepare the handoff — someone else will pick this up tomorrow", {"project": "synalux-portal"}),
            ("Create transition notes for bcba-private", {"project": "bcba-private"}),
            ("Pass the baton — save where we are on synalux-docs", {"project": "synalux-docs"}),
            ("I need to switch contexts. Save the handoff state.", {"project": "prism-mcp"}),
            ("Make sure the next person knows where we left off", {"project": "prism-mcp"}),
            ("Handoff time. Save the current project state.", {"project": "synalux-portal"}),
            ("End my shift. Create handoff notes.", {"project": "prism-mcp"}),
            ("Save handoff.", {"project": "prism-mcp"}),
            ("Write handoff for the prism project", {"project": "prism-mcp"}),
            ("Archive current state so the next agent can resume", {"project": "prism-mcp"}),
            ("Freeze the state and prepare handoff notes", {"project": "synalux-portal"}),
            ("Wrap up and hand off to the next session", {"project": "prism-mcp"}),
        ]
    },
    "session_forget_memory": {
        "default_args": {"memory_id": "entry-id", "reason": "No longer relevant"},
        "prompts": [
            ("Delete the memory entry for the broken config change", {"memory_id": "broken_config_entry"}),
            ("That memory about the old deployment script is totally wrong. Nuke it.", {"memory_id": "old_deploy_script"}),
            ("Remove the memory about the failed deploy last Friday", {"memory_id": "failed_deploy_friday"}),
            ("Forget the session entry about the old API design", {"memory_id": "old_api_design"}),
            ("Erase that incorrect note about the database schema", {"memory_id": "wrong_db_schema"}),
            ("This entry is outdated. Please delete it.", {"memory_id": "outdated_entry"}),
            ("Purge the stale memory about our old auth flow", {"memory_id": "old_auth_flow"}),
            ("Remove memory ID abc-123 — it's wrong information", {"memory_id": "abc-123"}),
            ("That ledger entry is no longer accurate. Remove it please.", {"memory_id": "inaccurate_entry"}),
            ("Delete that bad note we saved about the Stripe integration", {"memory_id": "bad_stripe_note"}),
            ("Clean out the memory about the deprecated endpoint", {"memory_id": "deprecated_endpoint"}),
            ("Forget entry xyz-789", {"memory_id": "xyz-789"}),
            # Targeted: diverse "nuke/delete" phrasings to prevent knowledge_forget hallucination
            ("That memory entry about the old deployment script is totally wrong. Nuke it.", {"memory_id": "old_deploy_entry"}),
            ("Get rid of that wrong memory about the API keys", {"memory_id": "wrong_api_keys"}),
            ("Wipe out the session memory for the failed migration", {"memory_id": "failed_migration"}),
            ("Kill that bad entry in the ledger", {"memory_id": "bad_entry"}),
            ("Trash the memory about our broken CI config", {"memory_id": "broken_ci_config"}),
            ("Remove that incorrect session note", {"memory_id": "incorrect_note"}),
            ("Delete this memory — it's completely wrong", {"memory_id": "wrong_memory"}),
            ("Destroy that stale entry about the old webhook", {"memory_id": "stale_webhook"}),
        ]
    },
    "session_health_check": {
        "default_args": {},
        "prompts": [
            ("Check if the memory database has any integrity issues", {}),
            ("Is everything OK with the memory backend? Run diagnostics.", {}),
            ("Run a health check on the memory system", {}),
            ("Is our memory system healthy? Auto-fix if not.", {"auto_fix": True}),
            ("Diagnose the memory storage", {}),
            ("Are there any issues with our session database?", {}),
            ("Run integrity checks on the knowledge base", {}),
            ("Check for orphaned entries or missing embeddings", {}),
            ("Is the memory layer working correctly?", {}),
            ("System health check please", {}),
            ("Any problems with the database?", {}),
            ("Scan for memory issues", {}),
        ]
    },
    "knowledge_search": {
        "default_args": {"query": "search query"},
        "prompts": [
            ("Search knowledge base for anything about CORS policies", {"query": "CORS policies"}),
            ("Any institutional knowledge about how we handle rate limiting?", {"query": "rate limiting handling"}),
            ("What's in our knowledge base about Supabase RLS policies?", {"query": "Supabase RLS policies"}),
            ("I need to know if our knowledge base has anything on Kubernetes pod autoscaling.", {"query": "Kubernetes pod autoscaling"}),
            ("Any documented patterns for error handling?", {"query": "error handling patterns"}),
            ("What do we know about edge function cold starts?", {"query": "edge function cold starts"}),
            ("Look up knowledge about OAuth2 flows", {"query": "OAuth2 flows"}),
            ("Is there documented guidance on our database indexing strategy?", {"query": "database indexing strategy"}),
            ("Search KB for JWT best practices", {"query": "JWT best practices"}),
            ("Any knowledge items about deployment procedures?", {"query": "deployment procedures"}),
            ("What's recorded about our caching architecture?", {"query": "caching architecture"}),
            ("Check knowledge for HIPAA compliance notes", {"query": "HIPAA compliance"}),
            # Extra examples with "knowledge" keyword to prevent false negatives
            ("Do we have any knowledge about handling WebSocket reconnections?", {"query": "WebSocket reconnections"}),
            ("Search our knowledge for database migration best practices", {"query": "database migration best practices"}),
            ("What knowledge do we have about error boundary patterns?", {"query": "error boundary patterns"}),
            ("Any knowledge about rate limiting strategies?", {"query": "rate limiting strategies"}),
            ("Check our knowledge base for API versioning notes", {"query": "API versioning"}),
        ]
    },
    "session_compact_ledger": {
        "default_args": {"project": "prism-mcp"},
        "prompts": [
            ("Compact old ledger entries for the prism-mcp project", {"project": "prism-mcp"}),
            ("The ledger is getting huge. Summarize and archive the old stuff.", {"project": "prism-mcp"}),
            ("Clean up the session history, compact entries", {"project": "synalux-portal"}),
            ("Archive old entries and roll them up", {"project": "prism-mcp"}),
            ("The session log is too long. Compact it.", {"project": "prism-mcp"}),
            ("Compress the older ledger entries into summaries", {"project": "prism-mcp"}),
            ("Run compaction on the prism-mcp ledger", {"project": "prism-mcp"}),
            ("Tidy up the session history", {"project": "synalux-portal"}),
            ("Compact ledger.", {"project": "prism-mcp"}),
            ("Roll up old sessions into summaries", {"project": "prism-mcp"}),
        ]
    },
    "session_export_memory": {
        "default_args": {"output_dir": "/tmp/export", "project": "prism-mcp", "format": "json"},
        "prompts": [
            ("Export prism-mcp memory to /tmp/export", {"output_dir": "/tmp/export", "project": "prism-mcp"}),
            ("Dump everything to a file so I can back it up. JSON format, save to /tmp/prism-backup.", {"output_dir": "/tmp/prism-backup", "format": "json"}),
            ("Export all memory data for bcba-private to my desktop", {"output_dir": "/Users/admin/Desktop", "project": "bcba-private"}),
            ("Back up the knowledge base to JSON", {"output_dir": "/tmp/export", "format": "json"}),
            ("Save a snapshot of all session data to /tmp/dump", {"output_dir": "/tmp/dump"}),
            ("Create a portable export of the prism project", {"output_dir": "/tmp/export", "project": "prism-mcp"}),
            ("I need an offline copy of our memory. Export it.", {"output_dir": "/tmp/export"}),
            ("Export memory.", {"output_dir": "/tmp/export"}),
            ("Download all session history as markdown", {"output_dir": "/tmp/export", "format": "markdown"}),
            ("Archive the project memory to disk", {"output_dir": "/tmp/archive"}),
        ]
    },
    "session_task_route": {
        "default_args": {"task_description": "task to route"},
        "prompts": [
            ("Route this task: refactoring the auth middleware", {"task_description": "refactoring the auth middleware", "estimated_scope": "refactor"}),
            ("Should I handle this CSS grid refactor myself or punt it to the local model?", {"task_description": "CSS grid refactor", "estimated_scope": "refactor"}),
            ("Is this task complex enough for the cloud or can local AI handle it?", {"task_description": "task complexity evaluation"}),
            ("Route: fixing a typo in the README", {"task_description": "fixing a typo in the README", "estimated_scope": "minor_edit"}),
            ("Should this bug fix go to the local model or cloud?", {"task_description": "bug fix routing", "estimated_scope": "bug_fix"}),
            ("Determine if this new feature needs cloud processing", {"task_description": "new feature evaluation", "estimated_scope": "new_feature"}),
            ("Can the local agent handle adding unit tests?", {"task_description": "adding unit tests"}),
            ("Route this coding task appropriately", {"task_description": "coding task routing"}),
        ]
    },
}

# === REASONING PROMPTS (diverse, NO tool) ===
REASONING_PROMPTS = [
    # Standard technical questions
    "What is the difference between TCP and UDP?",
    "How does React's virtual DOM work?",
    "Write a Python function to reverse a linked list",
    "Explain JWT tokens and how they work in authentication",
    "What are the pros and cons of microservices architecture?",
    "How do I save state in React with useState?",
    "Explain how session tokens work in web authentication",
    "What is knowledge distillation in machine learning?",
    "How do I save data to localStorage in the browser?",
    "What is task routing in distributed systems like Celery?",
    "How do I implement a session manager in Express.js?",
    "Explain memory management in Rust — ownership and borrowing",
    "What's the best way to save user preferences in React Native?",
    "Write a function that searches through a knowledge graph using BFS",
    "How does garbage collection work in Go?",
    "Can you explain the compact representation of sparse matrices?",
    "What is the health check endpoint pattern in microservices?",
    "How do I export data from PostgreSQL to a CSV file?",
    "Write a bash one-liner to find files larger than 100MB",
    "How do you implement a search algorithm for a graph?",
    "Explain how load balancing works across multiple servers",
    "What is the difference between stack and heap memory?",
    "How does session replication work in distributed systems?",
    "What are database connection pooling strategies?",
    "Explain the circuit breaker pattern in microservices",
    "How do I implement WebSocket authentication?",
    "What is the difference between SQL and NoSQL databases?",
    "How do you implement pagination in a REST API?",
    "Explain the concept of eventual consistency",
    "What is the difference between monorepo and polyrepo?",
    # Targeted: meta-questions that should NOT trigger tools (false positive traps)
    "What tools do you have available?",
    "Tell me about yourself.",
    "What can you do?",
    "Who are you?",
    "What are your capabilities?",
    "Help me understand what you can do.",
    "List your available features.",
    "What kind of assistant are you?",
    "Are you an AI?",
    "What model are you based on?",
    "Hi there, how are you?",
    "Hello!",
    "What's your name?",
    "How does your memory system work?",
    "Explain how Prism MCP tools work.",
    "What is session memory?",
    "What is the knowledge search feature?",
    "How do I use the health check?",
    "Describe the export functionality.",
    "What tools are available for session management?",
    # More keyword traps
    "Explain the CAP theorem in simple terms",
    "What is knowledge representation in AI?",
    "How do I handle session expiry in a web app?",
    "What are the best practices for saving state in Redux?",
    "How do you implement a health check endpoint in Express?",
    "What is task routing in Apache Airflow?",
    "How do I export modules in TypeScript?",
    "What is memory-safe programming?",
    "How do I compact a MongoDB collection?",
    "Explain context switching in operating systems",
    # Targeted: "session manager" / "session" in framework context (FP fix)
    "How do I implement a session manager in Express.js with Redis?",
    "What is session management in Django?",
    "How do I set up session middleware in Flask?",
    "How do I create a session in PHP using session_start()?",
    "What is session affinity in load balancing?",
    "How do I manage user sessions in a microservices architecture?",
    "What is the difference between session-based and token-based auth?",
    # Targeted: conversational closings (FP fix) 
    "Thanks, that's all for now.",
    "Goodbye!",
    "I'm done, thanks for the help.",
    "That's it for today, thank you.",
    "OK, we're finished.",
    # Targeted: "context manager" / "context" in programming context (FP fix — HEAVY)
    "Write me a Python context manager for database connections.",
    "How do I create a context manager using __enter__ and __exit__?",
    "Explain Python's contextlib module.",
    "How does context switching work in operating systems?",
    "What is execution context in JavaScript?",
    "Write a Python context manager that handles file locking.",
    "How do I use contextlib.contextmanager decorator?",
    "Explain the difference between a context manager and a decorator in Python.",
    "Write a context manager for managing database transactions.",
    "How do I create an async context manager in Python?",
    "What is a browser rendering context?",
    "Explain React context API vs Redux for state management.",
    "Write a context manager that times code execution.",
    "How does OpenGL rendering context work?",
    "What is the difference between context manager and try/finally in Python?",
    # Targeted: "forget gate" / "forget" in ML/academic context (FP fix — HEAVY)
    "What is the forget gate in an LSTM neural network?",
    "Explain how LSTM forget gates control information flow.",
    "What is catastrophic forgetting in neural networks?",
    "How do I implement an LSTM with forget bias in PyTorch?",
    "Explain the role of the forget gate in GRU vs LSTM.",
    "What is continual learning and how does it address catastrophic forgetting?",
    "How do I tune the forget bias in TensorFlow LSTM layers?",
    "What is the forget gate activation function in an LSTM cell?",
    "Explain elastic weight consolidation for preventing forgetting.",
    "How do LSTM forget gates differ from attention mechanisms?",
    "Write a PyTorch LSTM cell with a custom forget gate.",
    "What is progressive neural network approach to avoiding forgetting?",
    "Explain the forget gate equation: f_t = sigmoid(W_f * [h_t-1, x_t] + b_f)",
    "How does the forget gate interact with the cell state in LSTM?",
    "What is knowledge distillation for preventing catastrophic forgetting?",
]


# Disambiguation reasoning templates per tool
DISAMBIGUATION_THINK = {
    "session_forget_memory": [
        "The user wants to delete a SPECIFIC memory entry. For deleting individual entries by ID, I use session_forget_memory. knowledge_forget is for BULK deletion by project/category/age — that's not what the user wants here.",
        "This is about removing one particular entry. session_forget_memory handles specific entry deletion. knowledge_forget is for wiping entire categories — wrong tool for this.",
        "The user wants to nuke a specific entry. I need session_forget_memory (deletes by memory_id), NOT knowledge_forget (deletes by project/category).",
    ],
    "knowledge_forget": [
        "The user wants to bulk delete entries by project or category or age. knowledge_forget handles bulk deletion. session_forget_memory is for deleting ONE specific entry by ID — that's not what's needed here.",
    ],
    "session_save_ledger": [
        "The user wants to save/log/record what was accomplished. session_save_ledger is for logging work done. session_save_handoff is for transferring state to the next agent — that's not what's needed.",
        "This is a request to record session work. I use session_save_ledger for logging accomplishments, NOT session_save_handoff which is for handoff/transfer.",
        "The user wants to save what we did. session_save_ledger logs work. session_save_handoff passes state to another agent. This is clearly a save/log request.",
    ],
    "session_save_handoff": [
        "The user wants to hand off work to another agent/session. session_save_handoff transfers state. session_save_ledger just logs — the user explicitly wants a handoff.",
        "This is about passing the baton to another session or agent. I need session_save_handoff, not session_save_ledger which is just for recording work.",
    ],
    "session_load_context": [
        "The user wants to load/resume context for a project. I should use session_load_context.",
        "This is a request to start a session and load previous project state. session_load_context is the right tool.",
    ],
    "session_search_memory": [
        "The user wants to search through past session history. session_search_memory does semantic search over past conversations.",
        "This is a memory search request — the user wants to find past discussions or decisions. session_search_memory is correct.",
    ],
    "knowledge_search": [
        "The user wants to search the knowledge base for accumulated insights. knowledge_search queries curated knowledge items, not raw session history.",
        "This is about searching institutional knowledge. knowledge_search looks through curated KIs, which is what's needed.",
    ],
}

# Generic fallback think templates for tools without specific disambiguation
GENERIC_THINK = [
    "The user wants me to use the {tool_name} tool. This is the right tool for this request.",
    "I should use {tool_name} for this request.",
    "This requires calling the {tool_name} tool.",
]

# Reasoning think templates with keyword awareness
REASONING_THINK = [
    "This is a general knowledge question. I should answer directly without using any tools.",
    "This doesn't require any Prism MCP tools. I'll answer from my knowledge.",
    "No tools needed — this is a coding/knowledge question I can answer directly.",
    "I should answer this directly. No tool call is appropriate here.",
    "This is a standard technical question. No MCP tools needed.",
    # Keyword-aware disambiguation (prevents false positives)
    "Although this mentions 'session', it's asking about a general programming concept, not about Prism session management. I answer directly.",
    "The word 'memory' here refers to computer memory/RAM, not Prism's session memory system. No tool needed.",
    "This mentions 'save' but it's about saving data in an application, not about saving Prism session state. No tool.",
    "This mentions 'search' but it's about implementing search algorithms, not about searching Prism's memory. No tool.",
    "This mentions 'health check' but it's about the microservices pattern, not Prism's memory health check. No tool.",
    "This is asking ABOUT my tools/capabilities — it's a meta-question. I describe them, I don't call them.",
    "The user is greeting me or asking who I am. This is a conversational exchange, not a tool request.",
    "This mentions 'export' but it's about exporting data in a programming context, not Prism memory export. No tool.",
    "The user asks about 'knowledge' in an AI/academic context. This is not a request to search Prism's knowledge base.",
    "This mentions 'compact' but it's about data structures, not Prism ledger compaction. No tool.",
    "This mentions 'context' but it's about OS context switching, not loading Prism project context. No tool.",
    "This mentions 'task routing' but it's about distributed systems, not Prism's session_task_route. No tool.",
    # Pattern: session manager / session in framework context
    "This asks about implementing a session manager in a web framework like Express.js, Django, or Flask. This is general web development — NOT a request to use Prism's session_load_context tool. I answer directly.",
    # Pattern: conversational closings
    "The user is saying goodbye, thanking me, or indicating they're done. This is a conversational closing — NOT a request to save a handoff or ledger entry. No tool needed.",
    # Pattern: context manager (Python) — not Prism context loading
    "This asks about Python context managers (__enter__/__exit__, contextlib). This is about Python language features, NOT about loading Prism project context. No tool.",
    # Pattern: forget gate (LSTM/ML) — not Prism memory deletion
    "This asks about the forget gate in LSTM networks or catastrophic forgetting in ML. This is machine learning theory, NOT a request to delete Prism memory entries. No tool.",
]


def build_completion(tool_name, args, default_args):
    """Build a properly formatted tool-call completion with disambiguation CoT."""
    merged = {**default_args, **args}
    # Always include required fields for specific tools
    if tool_name == "session_load_context":
        merged.setdefault("level", "deep")
        merged.setdefault("toolAction", "Loading context")
        merged.setdefault("toolSummary", "Loading project context")
    elif tool_name == "session_save_ledger":
        merged.setdefault("conversation_id", "current")
    
    tool_json = json.dumps({"name": tool_name, "arguments": merged})
    
    # Use disambiguation reasoning if available, else generic
    if tool_name in DISAMBIGUATION_THINK:
        think = random.choice(DISAMBIGUATION_THINK[tool_name])
    else:
        think = random.choice(GENERIC_THINK).format(tool_name=tool_name)
    
    return f"<think>\n{think}\n</think>\n\n{TOOL_CALL_OPEN}\n{tool_json}\n{TOOL_CALL_CLOSE}"


def build_reasoning_completion(prompt):
    """Build a reasoning-only completion with keyword-aware CoT."""
    prompt_lower = prompt.lower()
    
    # Select keyword-aware think block when prompt contains tool-like keywords
    keyword_thinks = []
    if 'session' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[5])  # session disambiguation
    if 'memory' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[6])  # memory disambiguation
    if 'save' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[7])  # save disambiguation
    if 'search' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[8])  # search disambiguation
    if 'health check' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[9])  # health check disambiguation
    if any(w in prompt_lower for w in ['tool', 'capability', 'feature', 'available', 'can you do']):
        keyword_thinks.append(REASONING_THINK[10])  # meta-question
    if any(w in prompt_lower for w in ['hello', 'hi ', 'who are you', 'your name', 'about yourself', 'are you']):
        keyword_thinks.append(REASONING_THINK[11])  # greeting
    if 'export' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[12])  # export disambiguation
    if 'knowledge' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[13])  # knowledge disambiguation
    if 'compact' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[14])  # compact disambiguation
    if 'context' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[15])  # context disambiguation
    if 'task rout' in prompt_lower and 'prism' not in prompt_lower:
        keyword_thinks.append(REASONING_THINK[16])  # task routing disambiguation
    if any(w in prompt_lower for w in ['session manager', 'session_start', 'session middleware', 'session affinity']):
        keyword_thinks.append(REASONING_THINK[17])  # session-in-framework
    if any(w in prompt_lower for w in ['thanks', 'goodbye', 'done', 'finished', 'that\'s all', 'that\'s it']):
        keyword_thinks.append(REASONING_THINK[18])  # conversational closing
    if any(w in prompt_lower for w in ['context manager', 'contextlib', '__enter__', '__exit__', 'execution context']):
        keyword_thinks.append(REASONING_THINK[19])  # context manager (Python)
    if any(w in prompt_lower for w in ['forget gate', 'lstm', 'catastrophic forgetting', 'forget bias']):
        keyword_thinks.append(REASONING_THINK[20])  # forget gate (ML)
    
    if keyword_thinks:
        think = random.choice(keyword_thinks)
    else:
        think = random.choice(REASONING_THINK[:5])  # generic no-tool
    
    return f"<think>\n{think}\n</think>\n\nI'll answer this directly.\n\n"



def main():
    random.seed(42)
    data = []
    
    # Generate tool examples
    tool_count = 0
    for tool_name, config in TOOL_PROMPTS.items():
        for prompt, extra_args in config["prompts"]:
            completion = build_completion(tool_name, extra_args, config["default_args"])
            data.append({
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
            })
            tool_count += 1
    
    # Generate reasoning examples
    reasoning_count = 0
    for prompt in REASONING_PROMPTS:
        completion = build_reasoning_completion(prompt)
        data.append({
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
        })
        reasoning_count += 1
    
    print(f"Generated {len(data)} unique examples:")
    print(f"  Tool examples: {tool_count} ({len(TOOL_PROMPTS)} tools)")
    for tool_name, config in TOOL_PROMPTS.items():
        print(f"    {tool_name}: {len(config['prompts'])} phrasings")
    print(f"  Reasoning examples: {reasoning_count}")
    
    # Repeat for stronger signal (80x for better coverage)
    repeated = data * 80
    random.shuffle(repeated)
    
    split = int(len(repeated) * 0.95)
    train = repeated[:split]
    valid = repeated[split:]
    
    out_dir = "/Users/admin/synalux-private/data/grpo"
    with open(f"{out_dir}/train.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    
    with open(f"{out_dir}/valid.jsonl", "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n  Repeated 50x → {len(repeated)} total")
    print(f"  Train: {len(train)}, Valid: {len(valid)}")
    print(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
