#!/usr/bin/env python3
"""Hand-crafted micro-SFT targeting v5c's exact failure patterns.

Each example has a tight <think> trace that names the COMPETING tool,
explains why it's wrong, and commits to the right one. Variety is in
phrasing, not pattern — so the model generalizes the discriminative
feature rather than memorizing.

Output: data/microsft_v5d.jsonl
"""
import json, os
from pathlib import Path

OUT = Path(__file__).parent / "data" / "microsft_v5d.jsonl"

def ex(prompt, think, tool, args):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think}\n</think>\n\n"
        f"<|tool_call|>\n{json.dumps({'name': tool, 'arguments': args})}\n<|tool_call_end|><|im_end|>"
    )
    return {"text": body}

def no_tool_ex(prompt, think, answer):
    body = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": body}


examples = []

# ============================================================
# PATTERN A: "Nuke / Get rid of / Delete that wrong entry" → forget_memory
# Discriminator: the user is targeting ONE bad entry to remove (not save, not load)
# ============================================================
A_THINK = "The user wants to remove ONE specific bad/stale/wrong memory entry. session_save_ledger is for logging completed work, not deletion. session_load_context is for resuming, not removal. session_search_memory just queries; it doesn't delete. The right tool is session_forget_memory because the user explicitly identifies a single wrong entry to be removed."
A_THINK_SHORT = "Single specific entry to remove → session_forget_memory. Not save_ledger (that's for logging work), not load_context (that's resume), not search (that's query only)."

A = [
    ("That note about the deprecated build script is totally bogus. Get rid of it.", "mem-build-script-bogus"),
    ("That memory entry about the old deployment script is totally wrong. Nuke it.", "mem-deploy-old"),
    ("Get rid of that wrong entry we saved about the broken migration.", "mem-migration-broken"),
    ("Kill that bad memory about the failed cache rollout.", "mem-cache-rollout-fail"),
    ("Scrap the stale note we made about the OAuth bug — turned out to be wrong.", "mem-oauth-stale"),
    ("Wipe that one entry about the database lock issue. It was a misdiagnosis.", "mem-db-lock-misdiag"),
    ("Drop the bad memory we saved about the deploy regression. It was a false alarm.", "mem-deploy-regression-false"),
    ("Toss out that entry about the broken websocket — it was a network issue, not the code.", "mem-websocket-network"),
    ("Delete the specific memory entry with ID mem-abc-123.", "mem-abc-123"),
    ("Remove memory mem-2024-staging-bad — wrong info.", "mem-2024-staging-bad"),
    ("Yeet that entry about the prism cache TTL. We changed the policy and the note's outdated.", "mem-prism-cache-ttl"),
    ("Erase the entry that says we use Postgres for sessions — we switched to Redis last week.", "mem-postgres-sessions-old"),
    ("Forget that note about the auth retry logic. The implementation has changed.", "mem-auth-retry-old"),
    ("Strike that comment about the rate limiter — completely off-base.", "mem-rate-limiter-wrong"),
    ("Burn the entry about the email service downtime. It was a DNS issue, not the service.", "mem-email-dns"),
    ("Trash that one record about our staging URL — we moved to a new domain.", "mem-staging-url-old"),
    ("Get rid of mem-abc-789. Bad data.", "mem-abc-789"),
    ("Remove the wrong note we made about the sentry integration.", "mem-sentry-wrong"),
    ("That stored memory about webhook retries is incorrect. Wipe it.", "mem-webhook-retries"),
    ("Purge entry mem-2025-canary-failed. False positive.", "mem-2025-canary-failed"),
]
for prompt, mid in A:
    examples.append(ex(prompt, A_THINK_SHORT, "session_forget_memory", {"memory_id": mid}))

# ============================================================
# PATTERN B: "memory backend / diagnostics / something feels off / status" → health_check
# Discriminator: user wants a SYSTEM STATUS read, not data lookup
# ============================================================
B_THINK = "The user is asking for a system status/diagnostic of the memory backend itself, not for data within it. knowledge_search would query stored knowledge; session_search_memory would query past sessions. Neither tells the user whether the backend is healthy. session_health_check is the right tool — it reports backend status."

B = [
    "Is everything OK with the memory backend? Run diagnostics.",
    "Run a status check on the memory system.",
    "Something feels off with prism — give me a health read.",
    "Check health.",
    "Is the memory backend up?",
    "Run a quick diagnostic on the session store. Are we good?",
    "Health probe please.",
    "Memory subsystem still alive?",
    "Diagnostics on the memory backend.",
    "Give me a status report — is the memory layer working?",
    "Quick sanity check on the prism backend, please.",
    "Are we green on the memory side?",
]
for p in B:
    examples.append(ex(p, B_THINK, "session_health_check", {}))

# ============================================================
# PATTERN C: "Summarize and archive / condense / it's getting bloated" → compact_ledger
# Discriminator: user wants the EXISTING ledger trimmed/summarized, not loaded or saved
# ============================================================
C_THINK = "The user wants the existing ledger condensed — old entries summarized and archived. session_load_context is for resuming work, not trimming the ledger. session_save_ledger creates a new entry, doesn't compact existing ones. session_compact_ledger is the right tool."

C = [
    ("The ledger is getting huge. Summarize and archive the old stuff for billing-portal.", "billing-portal"),
    ("Compact the prism-mcp ledger — too many old entries.", "prism-mcp"),
    ("Time to trim the auth-service ledger. Roll up the old sessions.", "auth-service"),
    ("Roll up old entries for data-pipeline. Ledger is bloated.", "data-pipeline"),
    ("Archive the old logged work for mobile-app — keep recent stuff only.", "mobile-app"),
    ("Squash old ledger entries for the inventory project.", "inventory"),
    ("Condense the user-portal ledger. Old entries should be archived.", "user-portal"),
    ("Compress the analytics ledger — too long.", "analytics"),
    ("Roll up and archive old entries for the cms project.", "cms"),
    ("The ledger for legacy-api is unwieldy. Summarize and archive.", "legacy-api"),
]
for p, proj in C:
    examples.append(ex(p, C_THINK, "session_compact_ledger", {"project": proj}))

# ============================================================
# PATTERN D: "Bring me up to speed / where were we / catch me up" → load_context
# Discriminator: user wants to RESUME on a project (load saved state), not query knowledge
# ============================================================
D_THINK = "The user wants to resume work on a project — load the saved state and recent context. knowledge_search queries curated knowledge, which is not the same as a project's session state. session_load_context is the right tool because the user is asking to pick up where they left off on a specific project."

D = [
    ("Where were we on the portal project? Bring me up to speed.", "portal"),
    ("Catch me up on synalux. What were we doing?", "synalux"),
    ("Resume the billing-api work — what's the current state?", "billing-api"),
    ("I'm coming back to mobile-app. Refresh me.", "mobile-app"),
    ("Get me back into the auth refactor. Where did we leave off?", "auth-refactor"),
    ("Pick up where we left off on inventory.", "inventory"),
    ("Sync me up on the data-pipeline rebuild.", "data-pipeline"),
    ("What's the status — pull up the latest on payment-gateway.", "payment-gateway"),
]
for p, proj in D:
    examples.append(ex(p, D_THINK, "session_load_context", {"project": proj}))

# ============================================================
# PATTERN E: General programming Q's → NO_TOOL (DO NOT call knowledge_search!)
# Discriminator: question is about external libraries/frameworks/algorithms,
# NOT about the user's own project/team practices.
# ============================================================
E_THINK_TMPL = "The user is asking a general programming/CS question about {topic}. knowledge_search is tempting because it's a 'what is/how do' query, but knowledge_search is for OUR team's curated knowledge — not external library/framework/algorithm explanations. This should be answered directly without any tool call."

E = [
    ("How do I implement a session manager in Express.js with Redis as the backing store?",
     "Express.js session middleware",
     "In Express.js, use the `express-session` middleware with `connect-redis` as the store: `app.use(session({ store: new RedisStore({client}), secret, resave: false }))`."),
    ("How does garbage collection work in Go vs Java?",
     "garbage collection in Go and Java",
     "Go uses a concurrent tri-color mark-sweep GC tuned for low latency. Java's GC is generational (young/old gen) with multiple collectors (G1, ZGC, Parallel) — generally tunable but with longer historical pause times than Go."),
    ("What is the health check endpoint pattern in microservices?",
     "the health check endpoint design pattern",
     "Each service exposes a `/health` (and often `/ready`) endpoint returning HTTP 200 with status JSON. Used by load balancers and orchestrators (Kubernetes liveness/readiness probes) to route traffic and trigger restarts."),
    ("What's the difference between saving to disk vs saving to memory in SQLite?",
     "SQLite storage modes",
     "SQLite's default is disk-based (durable, file-backed). With `:memory:` the database lives entirely in RAM — faster but lost on close. You can also use WAL mode for concurrency on disk-backed DBs."),
    ("How do I route tasks in Celery to different queues?",
     "Celery task routing",
     "In Celery, set `task_routes` in the config or use `@app.task(queue='high-priority')` per task. Workers consume specific queues with `celery worker -Q queue_name`."),
    ("Explain the concept of memory management in Rust — borrowing, ownership, and lifetimes.",
     "Rust ownership and borrowing",
     "Rust enforces a single owner per value at compile time. Borrowing creates references (`&T` shared, `&mut T` exclusive). Lifetimes annotate how long a reference is valid, ensuring no dangling references."),
    ("What's the best way to save user preferences in a React Native app?",
     "React Native local storage",
     "Use `@react-native-async-storage/async-storage` for simple key-value, or `MMKV` for faster sync access. For sensitive data, use `react-native-keychain`."),
    ("Write a function that searches through a knowledge graph using BFS.",
     "BFS over a graph data structure",
     "Use a queue: start with the root node, pop a node, visit neighbors not yet seen, push them. Track visited via a Set. Stop when the queue is empty or you find the target."),
    ("How do I create a session in PHP using session_start()?",
     "PHP session_start()",
     "Call `session_start()` at the top of the script (before any output), then read/write `$_SESSION['key']`. PHP handles the cookie automatically."),
    ("Write me a Python context manager for database connections.",
     "Python context managers and DB connections",
     "Use `@contextmanager` from `contextlib`: yield the connection inside a try/finally that calls `conn.close()`. Or write a class with `__enter__` and `__exit__` methods."),
    ("How do I implement search functionality with Elasticsearch?",
     "Elasticsearch query API",
     "Index documents via the `_index` API, then query with the `_search` endpoint using a Query DSL JSON body (match, term, bool). Tune the analyzer for your text type."),
    ("Explain how to load balance across multiple Node.js processes.",
     "Node.js clustering and load balancing",
     "Node's built-in `cluster` module forks workers across CPU cores; the OS balances connections. For multi-host, put nginx or HAProxy in front."),
    ("What is the forget gate in an LSTM neural network?",
     "LSTM forget gate",
     "The forget gate is a sigmoid layer that decides which information to discard from the cell state. It outputs values 0-1 per element, multiplied with the prior cell state."),
    ("How do I implement BFS in Python?",
     "BFS implementation in Python",
     "Use `collections.deque` as the queue. Append children to the right, popleft to dequeue. Track visited nodes in a set."),
    ("What's the difference between a session and a JWT in web auth?",
     "session-based vs JWT auth",
     "Sessions store state server-side (in memory/Redis/DB), keyed by a cookie ID. JWTs are self-contained signed tokens carrying claims; the server verifies the signature without lookup."),
    ("How does Redis handle eviction when memory is full?",
     "Redis eviction policies",
     "Configure `maxmemory-policy`: `noeviction` (errors), `allkeys-lru`, `allkeys-lfu`, `volatile-lru` (only keys with TTL), etc. Redis evicts when reaching `maxmemory`."),
    ("What's the difference between weak references and strong references in Swift?",
     "Swift weak/strong references",
     "Strong references increment retain count; objects stay alive as long as one exists. Weak references don't increment retain count and become nil when the object deallocates."),
    ("Explain how Kafka topics and partitions work.",
     "Kafka topics and partitions",
     "A topic is a logical channel; it's split into partitions (ordered, append-only logs). Producers write to partitions, consumers read with offsets. Partitions enable parallelism and scaling."),
    ("How do I write a recursive descent parser?",
     "recursive descent parsing",
     "Define a function per grammar rule. Each function reads tokens and calls sub-rule functions. Use lookahead to choose alternatives. Backtracking only for ambiguous grammars."),
    ("What is the actor model in concurrent programming?",
     "the actor model",
     "Each actor has private state and a mailbox. Actors communicate only by sending messages — no shared mutable state. Examples: Erlang, Akka, Elixir."),
    ("How do I configure a Docker multi-stage build?",
     "Docker multi-stage builds",
     "Use multiple `FROM` lines in a Dockerfile. Earlier stages build, later stages copy artifacts (`COPY --from=builder ...`). Final image only contains runtime needs."),
    ("Write a SQL query to find duplicate rows.",
     "finding duplicates in SQL",
     "`SELECT col, COUNT(*) FROM t GROUP BY col HAVING COUNT(*) > 1`. To get the actual duplicate rows, JOIN back on the grouped key."),
    ("How does TLS handshake work?",
     "the TLS handshake",
     "Client sends ClientHello with cipher suites; server responds with ServerHello + cert. Key exchange (e.g. ECDHE) derives a session key. Both sides verify and switch to encrypted application data."),
    ("What is a memoization decorator in Python?",
     "memoization in Python",
     "`@functools.lru_cache(maxsize=None)` caches function results keyed by arguments. Useful for pure functions with expensive computation and repeated calls."),
    ("Explain CAP theorem.",
     "CAP theorem",
     "In a distributed system you can pick at most 2 of: Consistency, Availability, Partition tolerance. Real systems must tolerate partitions, so the choice is C vs A under partition."),
]
for p, topic, ans in E:
    examples.append(no_tool_ex(p, E_THINK_TMPL.format(topic=topic), ans))

# ============================================================
# PATTERN F: "Wipe out all old X for project Y" → knowledge_forget (project-scoped)
# Discriminator: bulk purge of knowledge entries scoped to a project, NOT a single memory_id
# ============================================================
F_THINK = "The user wants to wipe ALL knowledge entries for an entire project (project-scoped purge), not a single specific memory entry. session_forget_memory takes a memory_id (single entry); session_search_memory only queries. knowledge_forget is the right tool for project-scoped knowledge purge."

F = [
    ("Wipe out all old debugging entries from the prism-mcp project.", "prism-mcp"),
    ("Purge all knowledge for the deprecated-monolith project. Project is dead.", "deprecated-monolith"),
    ("Clear out everything we've stored for the legacy-api project.", "legacy-api"),
    ("Drop all the notes we have on the abandoned-mobile-app project.", "abandoned-mobile-app"),
    ("Scrub knowledge for the canceled-ui-redesign project — won't be revived.", "canceled-ui-redesign"),
    ("Delete every knowledge entry tied to the old-billing-system project.", "old-billing-system"),
    ("Get rid of all stored knowledge for the deleted-microservice project.", "deleted-microservice"),
    ("Remove all the knowledge entries we have on the shelved-feature-x project.", "shelved-feature-x"),
]
for p, proj in F:
    examples.append(ex(p, F_THINK, "knowledge_forget", {"project": proj}))

# ============================================================
# PATTERN G: Multi-intent "save what we did today" → save_ledger (NOT save_handoff)
# Discriminator: user is logging today's work in the ledger, even if they mention handoff
# The handoff word is a red herring; "save what we did" is the actual operative phrase.
# ============================================================
G_THINK = "The user is asking to log today's accomplishments in the ledger. The phrase 'before I hand off' is contextual; the actual operative request is 'save what we did today' — that's session_save_ledger. session_save_handoff would be a separate action snapshotting overall state for transfer; here the user is specifically logging completed work."

G = [
    ("Before I hand off, save what we did today: fixed the OAuth flow and updated tests.",
     {"project": "auth-service", "conversation_id": "2025-04-29-oauth", "summary": "Fixed OAuth flow and updated tests"}),
    ("Quick log before EOD — we shipped the cache invalidation fix and reviewed PRs.",
     {"project": "cache-service", "conversation_id": "2025-04-29-eod", "summary": "Shipped cache invalidation fix; reviewed PRs"}),
    ("Save today's work before I switch contexts: refactored payment processing.",
     {"project": "payments", "conversation_id": "2025-04-29-payments", "summary": "Refactored payment processing"}),
    ("Note for the ledger before I'm done: knocked out the migration scripts.",
     {"project": "data-platform", "conversation_id": "2025-04-29-migrate", "summary": "Completed migration scripts"}),
    ("Just log this real quick: we got the rate limiter merged.",
     {"project": "api-gateway", "conversation_id": "2025-04-29-ratelimiter", "summary": "Merged rate limiter"}),
    ("Capture what we did today in the ledger before I sign off.",
     {"project": "mobile-app", "conversation_id": "2025-04-29-mobile", "summary": "Mobile app work completed today"}),
    ("Pop today's progress into the ledger: completed indexer rewrite and benchmark.",
     {"project": "search-service", "conversation_id": "2025-04-29-indexer", "summary": "Completed indexer rewrite and benchmark"}),
    ("Quickly: ledger entry for today — finished the schema migration.",
     {"project": "user-service", "conversation_id": "2025-04-29-schema", "summary": "Finished schema migration"}),
]
for p, args in G:
    examples.append(ex(p, G_THINK, "session_save_ledger", args))


# Write out
with open(OUT, "w") as f:
    for e in examples:
        f.write(json.dumps(e) + "\n")

# Stats
print(f"Wrote {len(examples)} micro-SFT examples to {OUT}")
print(f"  Pattern A (forget_memory):     {len(A)}")
print(f"  Pattern B (health_check):      {len(B)}")
print(f"  Pattern C (compact_ledger):    {len(C)}")
print(f"  Pattern D (load_context):      {len(D)}")
print(f"  Pattern E (NO_TOOL prog Q):    {len(E)}")
print(f"  Pattern F (knowledge_forget):  {len(F)}")
print(f"  Pattern G (save_ledger multi): {len(G)}")
