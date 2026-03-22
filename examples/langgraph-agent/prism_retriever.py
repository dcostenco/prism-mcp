"""
Prism Memory Retriever — LangChain BaseRetriever Integration
=============================================================
Phase 3: Framework Integration for the Agent Memory System.

This module provides standard LangChain BaseRetriever subclasses that
bridge Prism MCP's search tools into the LangChain/LangGraph ecosystem.
Trace metadata from Phase 1 flows directly into Document.metadata,
making it automatically visible in LangSmith.

Architecture:
  ┌───────────────────────┐       ┌──────────────────────┐
  │  LangChain Chain      │       │  Prism MCP Server    │
  │  (create_retrieval_   │       │  (Node.js + TS)      │
  │   chain, RAG, etc.)   │       │                      │
  │                       │       │  session_search_     │
  │  PrismMemoryRetriever │──────→│  memory              │
  │  (BaseRetriever)      │  MCP  │  (enable_trace=True) │
  │                       │       │                      │
  │  Returns:             │       │  Returns:            │
  │  Document(            │←──────│  content[0]: results │
  │    page_content=text, │  JSON │  content[1]: trace   │
  │    metadata={trace}   │  RPC  │                      │
  │  )                    │       │                      │
  └───────────────────────┘       └──────────────────────┘

Why Two Retrievers?
  - PrismMemoryRetriever  → Semantic search (embeddings, pgvector)
  - PrismKnowledgeRetriever → Keyword search (FTS5, full-text)

  LangChain's EnsembleRetriever can combine both for hybrid search.

The Async Event Loop Gotcha (CRITICAL DESIGN DECISION):
  I built the LangChain adapter using _aget_relevant_documents because
  the underlying Model Context Protocol operates on asynchronous JSON-RPC
  streams. Forcing it into a synchronous _get_relevant_documents wrapper
  would risk event-loop collisions in high-concurrency LangGraph deployments.

  Specifically: the official Python MCP SDK (mcp package) is fundamentally
  async — it relies on asyncio to manage JSON-RPC streams over stdio/SSE.
  If we only implement the sync _get_relevant_documents(), calling
  asyncio.run() inside an already-running async LangGraph node throws:
      RuntimeError: This event loop is already running.

  This is the #1 bug that plagues LangGraph developers integrating
  external async services. The fix:

  1. _aget_relevant_documents() is the PRIMARY method (async-native)
  2. It uses asyncio.to_thread() to wrap the sync PrismMCPBridge,
     which is subprocess-based and performs blocking I/O
  3. asyncio.to_thread() delegates to the thread pool executor,
     avoiding event loop collisions entirely
  4. _get_relevant_documents() is a SYNC FALLBACK for scripts/notebooks
     where no event loop is running — it calls the bridge directly

  This pattern is production-safe for:
    - LangGraph async nodes (uses _aget_relevant_documents)
    - LangChain sync chains (uses _get_relevant_documents)
    - Jupyter notebooks (uses _get_relevant_documents)
    - FastAPI endpoints (uses _aget_relevant_documents)

Dependencies:
  pip install langchain-core pydantic
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

# Phase 3 logger — separate from Prism's server-side logging.
# Consumers can enable via: logging.getLogger("prism_retriever").setLevel(logging.DEBUG)
logger = logging.getLogger("prism_retriever")

# ---------------------------------------------------------------------------
# Constants — MemoryTrace content block markers
# ---------------------------------------------------------------------------
# These must match the markers defined in src/utils/tracing.ts
# (traceToContentBlock function). If the TypeScript side changes
# these markers, update them here too.

TRACE_MARKER = "=== MEMORY TRACE ==="


# ---------------------------------------------------------------------------
# Helper: Parse the MemoryTrace from a raw MCP content block
# ---------------------------------------------------------------------------

def _parse_trace_block(content_blocks: list[dict]) -> dict:
    """Extract MemoryTrace JSON from the second content block (content[1]).

    Phase 1 Design Recap:
      content[0] = human-readable search results (text)
      content[1] = MemoryTrace JSON, prefixed with "=== MEMORY TRACE ==="

    This function safely extracts and parses the trace, returning an
    empty dict if the block is missing, malformed, or doesn't contain
    the expected marker.

    Args:
        content_blocks: The raw MCP content block list from call_tool_raw().

    Returns:
        Parsed MemoryTrace dict, or empty dict if unavailable.
    """
    if len(content_blocks) < 2:
        logger.debug("No content[1] block — trace not available (enable_trace=false?)")
        return {}

    trace_block = content_blocks[1]
    trace_text = trace_block.get("text", "")

    # Verify the marker is present (guards against non-trace content[1])
    if TRACE_MARKER not in trace_text:
        logger.warning("content[1] exists but missing MEMORY TRACE marker")
        return {}

    # Strip the marker line to get pure JSON
    raw_json = trace_text.replace(TRACE_MARKER, "").strip()

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse MemoryTrace JSON: %s", e)
        return {"parse_error": str(e), "raw": raw_json[:200]}


# ===========================================================================
# PrismMemoryRetriever — Semantic Search via session_search_memory
# ===========================================================================

class PrismMemoryRetriever(BaseRetriever):
    """LangChain retriever that searches Prism MCP's semantic memory.

    Calls session_search_memory (pgvector cosine similarity) and returns
    results as standard LangChain Documents. When enable_trace=True,
    the Phase 1 MemoryTrace (latency breakdown, scores, strategy) is
    embedded in Document.metadata["trace"] — automatically visible in
    LangSmith run traces.

    Usage:
        from mcp_client import PrismMCPBridge
        from prism_retriever import PrismMemoryRetriever

        bridge = PrismMCPBridge()
        bridge.connect()

        retriever = PrismMemoryRetriever(
            mcp_client=bridge,
            project="my-project",
            enable_trace=True
        )

        # Sync usage
        docs = retriever.invoke("RAG architecture patterns")

        # Async usage (preferred in LangGraph)
        docs = await retriever.ainvoke("RAG architecture patterns")

        # Access trace metadata (from Phase 1)
        print(docs[0].metadata["trace"]["latency"]["embedding_ms"])

    LangSmith Integration:
        When this retriever is used inside a LangChain chain, LangSmith
        automatically captures the Document.metadata. This means the
        Phase 1 MemoryTrace (embedding_ms, storage_ms, top_score) appears
        in the LangSmith run trace with zero additional configuration.
    """

    # Allow PrismMCPBridge (non-Pydantic class) as a field type.
    # Without this, Pydantic v2 raises: "arbitrary_types_allowed not set"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Fields ────────────────────────────────────────────────────────
    mcp_client: Any = Field(
        description="Connected PrismMCPBridge instance. Must have call_tool_raw() method."
    )
    project: Optional[str] = Field(
        default=None,
        description="Optional project filter. Limits search to a specific project's memory."
    )
    enable_trace: bool = Field(
        default=True,
        description=(
            "Phase 1 Explainability: when True, requests a MemoryTrace content block "
            "from the server and embeds it in Document.metadata['trace']. "
            "Adds ~0ms overhead (trace generation is in-process, no extra API calls)."
        ),
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity score (0-1). Higher = fewer, more relevant results."
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return."
    )

    # ── Async Primary Method ──────────────────────────────────────────
    # DESIGN DECISION: _aget_relevant_documents is the PRIMARY method.
    #
    # I built this adapter async-first because the Model Context Protocol
    # operates on asynchronous JSON-RPC streams. Forcing it into a
    # synchronous wrapper would risk event-loop collisions in
    # high-concurrency LangGraph deployments.
    #
    # LangGraph calls .ainvoke() → routes to _aget_relevant_documents().
    # The underlying PrismMCPBridge is synchronous (subprocess stdio),
    # so we use asyncio.to_thread() to avoid blocking the event loop.
    #
    # WHY asyncio.to_thread() INSTEAD OF asyncio.run()?
    #   asyncio.run() → creates a NEW event loop → CRASHES if one
    #     is already running (LangGraph, FastAPI, any async context)
    #   asyncio.to_thread() → delegates to ThreadPoolExecutor → safe
    #     in any context, never touches the running event loop
    #
    # This is the #1 async integration gotcha in LangGraph. Getting
    # it wrong causes RuntimeError in production under load.

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async retrieval — the primary code path for LangGraph.

        Wraps the synchronous PrismMCPBridge.call_tool_raw() in
        asyncio.to_thread() to avoid blocking the event loop.

        Args:
            query: Natural language search query.
            run_manager: LangChain callback manager (for LangSmith tracing).

        Returns:
            List of Documents with page_content (results) and metadata (trace).
        """
        # Delegate the blocking MCP call to a thread pool worker
        raw_response = await asyncio.to_thread(
            self._call_mcp_search, query
        )
        return self._parse_response(raw_response, query)

    # ── Sync Fallback ─────────────────────────────────────────────────
    # For developers calling retriever.invoke() outside of async context.
    # This is safe because there's no running event loop to collide with.
    #
    # USE CAREFULLY: If you're inside a LangGraph async node or a
    # FastAPI endpoint, use .ainvoke() instead. Calling .invoke()
    # from an async context will block the event loop thread.

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Sync retrieval — fallback for non-async contexts.

        Calls the MCP bridge directly (it's already synchronous).
        Safe to use in scripts, notebooks, and sync LangChain chains.

        Args:
            query: Natural language search query.
            run_manager: LangChain callback manager (for LangSmith tracing).

        Returns:
            List of Documents with page_content (results) and metadata (trace).
        """
        raw_response = self._call_mcp_search(query)
        return self._parse_response(raw_response, query)

    # ── Internal: MCP Bridge Call ─────────────────────────────────────

    def _call_mcp_search(self, query: str) -> dict:
        """Call session_search_memory via the MCP bridge.

        Constructs the MCP tool arguments and delegates to
        call_tool_raw() which preserves the content block structure.

        Args:
            query: The search query string.

        Returns:
            Raw MCP response dict with "content" list and "isError" bool.
        """
        arguments = {
            "query": query,
            "limit": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "enable_trace": self.enable_trace,
        }

        # Add optional project filter
        if self.project:
            arguments["project"] = self.project

        logger.debug("Calling session_search_memory with args: %s", arguments)

        return self.mcp_client.call_tool_raw(
            "session_search_memory", arguments
        )

    # ── Internal: Response Parser ─────────────────────────────────────

    def _parse_response(self, raw_response: dict, query: str) -> List[Document]:
        """Parse the raw MCP response into LangChain Documents.

        Handles three cases:
          1. Error response → returns Document with error message
          2. Normal response (no trace) → Document with results only
          3. Traced response → Document with results + MemoryTrace metadata

        Args:
            raw_response: Dict from call_tool_raw() with "content" and "isError".
            query: Original query (included in metadata for debugging).

        Returns:
            List containing a single Document (or empty list on error).
        """
        # Handle MCP errors
        if raw_response.get("isError"):
            error_text = "Unknown MCP error"
            content = raw_response.get("content", [])
            if content:
                error_text = content[0].get("text", error_text)
            logger.error("MCP tool call failed: %s", error_text)
            return [
                Document(
                    page_content=f"[MCP Error] {error_text}",
                    metadata={"source": "prism_mcp", "error": True},
                )
            ]

        content_blocks = raw_response.get("content", [])

        if not content_blocks:
            logger.warning("Empty content blocks from session_search_memory")
            return []

        # content[0] = human-readable search results
        text_results = content_blocks[0].get("text", "")

        # content[1] = MemoryTrace JSON (if enable_trace was True)
        trace_metadata = _parse_trace_block(content_blocks)

        # Build the metadata dict
        # The trace data is nested under "trace" so it doesn't pollute
        # the top-level metadata namespace used by other LangChain components.
        metadata = {
            "source": "prism_mcp",
            "tool": "session_search_memory",
            "query": query,
        }

        if trace_metadata:
            metadata["trace"] = trace_metadata

        return [
            Document(
                page_content=text_results,
                metadata=metadata,
            )
        ]


# ===========================================================================
# PrismKnowledgeRetriever — Keyword Search via knowledge_search
# ===========================================================================

class PrismKnowledgeRetriever(BaseRetriever):
    """LangChain retriever that searches Prism MCP's keyword-based knowledge.

    Calls knowledge_search (FTS5 full-text search) and returns results
    as standard LangChain Documents. Complements PrismMemoryRetriever:
      - PrismMemoryRetriever: semantic (meaning-based, uses embeddings)
      - PrismKnowledgeRetriever: keyword (exact term overlap, uses FTS5)

    For hybrid search, use LangChain's EnsembleRetriever:
        from langchain.retrievers import EnsembleRetriever
        hybrid = EnsembleRetriever(
            retrievers=[memory_retriever, knowledge_retriever],
            weights=[0.6, 0.4]
        )

    Usage:
        retriever = PrismKnowledgeRetriever(
            mcp_client=bridge,
            project="my-project",
            enable_trace=True
        )
        docs = retriever.invoke("debugging authentication")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mcp_client: Any = Field(description="Connected PrismMCPBridge instance.")
    project: Optional[str] = Field(
        default=None,
        description="Optional project filter."
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (e.g., 'debugging', 'architecture')."
    )
    enable_trace: bool = Field(
        default=True,
        description="Phase 1: request MemoryTrace in content[1]."
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results."
    )

    # ── Async Primary (same pattern as PrismMemoryRetriever) ──────────
    # Uses asyncio.to_thread() to avoid event loop collisions.
    # See PrismMemoryRetriever docstrings for the full rationale.

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async retrieval for knowledge_search."""
        raw_response = await asyncio.to_thread(
            self._call_mcp_search, query
        )
        return self._parse_response(raw_response, query)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Sync fallback for knowledge_search."""
        raw_response = self._call_mcp_search(query)
        return self._parse_response(raw_response, query)

    def _call_mcp_search(self, query: str) -> dict:
        """Call knowledge_search via the MCP bridge."""
        arguments = {
            "query": query,
            "limit": self.max_results,
            "enable_trace": self.enable_trace,
        }
        if self.project:
            arguments["project"] = self.project
        if self.category:
            arguments["category"] = self.category

        logger.debug("Calling knowledge_search with args: %s", arguments)
        return self.mcp_client.call_tool_raw("knowledge_search", arguments)

    def _parse_response(self, raw_response: dict, query: str) -> List[Document]:
        """Parse MCP response into Documents (same pattern as PrismMemoryRetriever)."""
        if raw_response.get("isError"):
            error_text = raw_response.get("content", [{}])[0].get("text", "Unknown error")
            return [
                Document(
                    page_content=f"[MCP Error] {error_text}",
                    metadata={"source": "prism_mcp", "error": True},
                )
            ]

        content_blocks = raw_response.get("content", [])
        if not content_blocks:
            return []

        text_results = content_blocks[0].get("text", "")
        trace_metadata = _parse_trace_block(content_blocks)

        metadata = {
            "source": "prism_mcp",
            "tool": "knowledge_search",
            "query": query,
        }
        if trace_metadata:
            metadata["trace"] = trace_metadata

        return [
            Document(
                page_content=text_results,
                metadata=metadata,
            )
        ]
