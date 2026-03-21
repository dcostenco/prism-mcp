"""
Prism Research Agent — Tools
=============================
Tools the agent can use during research. Includes both a built-in
Prism knowledge base and web search via Brave API (if available).

Architecture:
  - search_prism_knowledge: Vector-like search over embedded Prism docs
  - search_web: Live web search via Brave API (optional)
  - lookup_mcp_concept: Quick lookup for MCP/agentic AI terminology
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.parse
from typing import Optional


# ---------------------------------------------------------------------------
# Built-in Prism Knowledge Base (no external dependencies)
# ---------------------------------------------------------------------------
# This simulates a RAG knowledge store with curated Prism MCP documentation.
# In production, this would query a ChromaDB/FAISS vector store.

PRISM_KNOWLEDGE_BASE = [
    {
        "topic": "progressive_context_loading",
        "content": (
            "Prism MCP supports three levels of context loading: "
            "'quick' (~50 tokens, just TODOs and keywords), "
            "'standard' (~200 tokens, recommended default with summaries and decisions), "
            "and 'deep' (~1000+ tokens, full session history). "
            "This saves tokens and speeds up boot time."
        ),
        "tags": ["memory", "context", "tokens", "performance"],
    },
    {
        "topic": "time_travel",
        "content": (
            "Every successful handoff save creates a versioned snapshot. "
            "memory_history browses all versions; memory_checkout reverts to any version. "
            "This is non-destructive like git revert — no history is ever lost. "
            "Uses optimistic concurrency control (OCC) with version numbers."
        ),
        "tags": ["versioning", "OCC", "state", "history", "git"],
    },
    {
        "topic": "agent_telepathy",
        "content": (
            "Multi-agent sync: when Agent A (Cursor) saves a handoff, "
            "Agent B (Claude Desktop) gets notified instantly. "
            "Local mode uses file-based IPC via SQLite polling. "
            "Cloud mode uses Supabase Realtime (Postgres CDC). "
            "No configuration needed."
        ),
        "tags": ["sync", "multi-agent", "realtime", "collaboration"],
    },
    {
        "topic": "storage_architecture",
        "content": (
            "Prism uses a storage abstraction layer with two backends: "
            "SQLite (local, libSQL + F32_BLOB vectors + FTS5 full-text search) "
            "and Supabase (cloud, PostgreSQL + pgvector). "
            "The factory in storage/index.ts auto-selects based on PRISM_STORAGE env var. "
            "All handlers route through getStorage() for backend-agnostic operation."
        ),
        "tags": ["storage", "sqlite", "supabase", "vectors", "architecture"],
    },
    {
        "topic": "mind_palace_dashboard",
        "content": (
            "A glassmorphism dashboard served at localhost:3000. "
            "Features: current state & TODOs, Git drift detection, "
            "Morning Briefing (Gemini-synthesized), Time Travel timeline, "
            "Visual Memory vault, Session Ledger audit trail, and Neural Graph "
            "for visualizing project relationships via Vis.js force-directed layout."
        ),
        "tags": ["dashboard", "UI", "visualization", "neural_graph"],
    },
    {
        "topic": "security_scanning",
        "content": (
            "Prompt Injection Shield: Gemini-powered security scan in "
            "session_health_check. Detects system override attempts, jailbreaks, "
            "and data exfiltration hidden in agent memory. "
            "Tuned to minimize false positives on normal dev commands. "
            "Brain health check also detects missing embeddings, duplicates, "
            "orphaned handoffs, and stale rollups."
        ),
        "tags": ["security", "health", "injection", "fsck"],
    },
    {
        "topic": "fact_merger",
        "content": (
            "Async LLM contradiction resolution on every handoff save. "
            "If old context says 'Postgres' and new says 'MySQL', "
            "Gemini silently merges the facts in the background. "
            "Zero latency impact — uses fire-and-forget pattern. "
            "Maintains consistency in long-running project memory."
        ),
        "tags": ["consistency", "LLM", "merge", "async"],
    },
    {
        "topic": "knowledge_accumulation",
        "content": (
            "Every session_save_ledger and session_save_handoff automatically "
            "extracts keywords using lightweight in-process NLP (~0.020ms/call). "
            "No LLM calls needed. Extracted keywords enable knowledge_search "
            "and session_search_memory for semantic retrieval across sessions."
        ),
        "tags": ["NLP", "keywords", "search", "semantic"],
    },
    {
        "topic": "mcp_protocol",
        "content": (
            "Model Context Protocol (MCP) is a standard that connects AI systems "
            "with external tools and data sources. Prism MCP communicates via "
            "stdio transport with clients like Claude Desktop, Cursor, and Windsurf. "
            "It exposes Tools, Prompts, and Resources as MCP primitives. "
            "Tools are callable functions; Resources are readable data sources."
        ),
        "tags": ["MCP", "protocol", "stdio", "tools", "resources"],
    },
    {
        "topic": "auto_compaction",
        "content": (
            "session_compact_ledger auto-compacts old session entries by rolling "
            "them up into Gemini-generated summaries. Prevents the ledger from "
            "growing indefinitely. Keeps recent entries intact (configurable). "
            "Uses threshold-based triggering with dry_run preview support."
        ),
        "tags": ["compaction", "ledger", "Gemini", "optimization"],
    },
    {
        "topic": "reality_drift_detection",
        "content": (
            "Prism captures Git state (branch + commit SHA) on every handoff save. "
            "On context load, it compares saved vs current Git state. "
            "If files changed outside the agent's view, it warns: "
            "'REALITY DRIFT DETECTED'. Prevents stale-context bugs."
        ),
        "tags": ["git", "drift", "stale", "detection"],
    },
    {
        "topic": "code_mode_templates",
        "content": (
            "8 pre-built QuickJS extraction templates for zero-reasoning-token "
            "data transformation: github_issues, github_prs, jira_tickets, "
            "dom_links, dom_headings, api_endpoints, slack_messages, csv_summary. "
            "Usage: pass template name instead of custom JavaScript code. "
            "Runs in a sandboxed QuickJS executor for security."
        ),
        "tags": ["templates", "QuickJS", "sandbox", "extraction"],
    },
]


def search_prism_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """Search the Prism knowledge base using keyword matching.

    In a production RAG system, this would use vector embeddings
    and cosine similarity. Here we use tag/content matching for
    simplicity and zero-dependency operation.

    Args:
        query: The search query.
        top_k: Number of top results to return.

    Returns:
        List of matching knowledge entries with relevance scores.
    """
    query_terms = set(query.lower().split())
    scored_results = []

    for entry in PRISM_KNOWLEDGE_BASE:
        # Score based on tag matches + content keyword overlap
        tag_score = len(query_terms & set(t.lower() for t in entry["tags"]))
        content_words = set(entry["content"].lower().split())
        content_score = len(query_terms & content_words) * 0.5
        topic_score = 2.0 if any(t in entry["topic"].lower() for t in query_terms) else 0
        total_score = tag_score + content_score + topic_score

        if total_score > 0:
            scored_results.append({
                "topic": entry["topic"],
                "content": entry["content"],
                "score": round(total_score, 2),
            })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


def search_web(query: str) -> Optional[str]:
    """Search the web using Brave Search API.

    Requires BRAVE_API_KEY environment variable.
    Returns a summary of search results, or None if unavailable.

    Args:
        query: The search query.

    Returns:
        Formatted search results string, or None.
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return None

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&count=3"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        })

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []
        for item in data.get("web", {}).get("results", [])[:3]:
            results.append(f"• {item.get('title', 'N/A')}: {item.get('description', 'N/A')}")

        return "\n".join(results) if results else None

    except Exception as e:
        return f"[Web search error: {e}]"


MCP_GLOSSARY = {
    "MCP": "Model Context Protocol — an open standard connecting AI models to external tools and data sources via a client-server architecture.",
    "stdio": "Standard I/O transport — MCP servers communicate with clients through stdin/stdout, enabling process-based tool execution.",
    "RAG": "Retrieval-Augmented Generation — technique where an AI retrieves relevant documents from a knowledge base before generating an answer, reducing hallucinations.",
    "OCC": "Optimistic Concurrency Control — version-based conflict detection where each write includes an expected_version to prevent overwrites.",
    "embeddings": "Dense vector representations of text that capture semantic meaning. Similar texts have similar vectors, enabling similarity search.",
    "vector_search": "Finding the most similar items in a database by comparing embedding vectors using cosine similarity or dot product.",
    "FTS5": "SQLite's Full-Text Search extension (version 5). Enables keyword-based search with ranking across text columns.",
    "pgvector": "PostgreSQL extension for storing and querying vector embeddings. Supports indexing methods like IVFFlat and HNSW.",
    "agentic_ai": "AI systems that can autonomously plan, use tools, make decisions, and loop until a task is completed — going beyond single-shot responses.",
    "LangGraph": "A Python framework by LangChain for building stateful, multi-step AI agents using directed graphs with nodes, edges, and conditional routing.",
    "handoff": "In Prism MCP, the latest project state saved for the next session to consume. Contains summary, TODOs, context, and branch info.",
    "ledger": "An append-only audit trail in Prism MCP that records what was accomplished in each session — immutable for accountability.",
    "telepathy": "Prism's multi-agent sync feature — when one agent saves state, other connected agents get notified in real-time.",
}


def lookup_mcp_concept(term: str) -> Optional[str]:
    """Look up an MCP or agentic AI concept.

    Args:
        term: The concept to look up.

    Returns:
        Definition string, or None if not found.
    """
    term_lower = term.lower().replace(" ", "_").replace("-", "_")
    # Try exact match first
    if term_lower in MCP_GLOSSARY:
        return f"**{term}**: {MCP_GLOSSARY[term_lower]}"

    # Try partial match
    for key, value in MCP_GLOSSARY.items():
        if term_lower in key.lower() or key.lower() in term_lower:
            return f"**{key}**: {value}"

    return None


# ---------------------------------------------------------------------------
# Prism Ledger — Persistent Agentic Memory
# ---------------------------------------------------------------------------
# Saves research results to a local JSON ledger file (standalone mode)
# or to Prism MCP's session_save_ledger tool (MCP-connected mode).

LEDGER_FILE = os.path.join(os.path.dirname(__file__), "research_ledger.json")


def save_to_prism_ledger(
    query: str,
    answer: str,
    iterations: int,
    search_results: list[str] = None,
) -> dict:
    """Save a completed research session to the persistent ledger.

    In standalone mode: writes to a local JSON file.
    In MCP-connected mode: routes to Prism's session_save_ledger tool
    (handled by the agent's save_session_node).

    This demonstrates AGENTIC MEMORY — the agent doesn't just answer
    and forget. It explicitly persists its findings for future retrieval.

    Args:
        query: The original research question.
        answer: The generated answer.
        iterations: Number of search-analyze loops performed.
        search_results: Raw search results used.

    Returns:
        The ledger entry that was saved.
    """
    import datetime
    import uuid

    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer_length": len(answer),
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
        "iterations": iterations,
        "sources_count": len(search_results) if search_results else 0,
        "tags": ["langgraph_research"],
    }

    # Append to local ledger file
    try:
        if os.path.exists(LEDGER_FILE):
            with open(LEDGER_FILE, "r") as f:
                ledger = json.load(f)
        else:
            ledger = []

        ledger.append(entry)

        with open(LEDGER_FILE, "w") as f:
            json.dump(ledger, f, indent=2)

        print(f"   💾 Saved to ledger: {LEDGER_FILE} ({len(ledger)} entries)")
    except Exception as e:
        print(f"   ⚠️ Ledger save failed: {e}")

    return entry
