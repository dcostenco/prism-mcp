"""
Prism Research Agent — State Definition
========================================
Defines the agent's state that flows through the LangGraph.
Each node reads and writes to this shared state.

─────────────────────────────────────────────────────────────────
Q&A #3:
"If we wanted to deploy this to 1,000 users, what would break first?"
─────────────────────────────────────────────────────────────────
RIGHT NOW: State lives in-memory during graph execution. Each
`agent.invoke(initial_state)` holds the full AgentState dict in
the Python process. This means:

  1. NO PERSISTENCE — if the process crashes mid-loop, all research
     is lost. There's no way to resume from the analyze step.
  2. NO CONCURRENCY — each invocation is synchronous. 1,000 users
     would need 1,000 concurrent Python processes.
  3. NO STATE INSPECTION — you can't see what the agent is doing
     mid-execution from an external monitoring tool.

TO SCALE, add LangGraph's built-in CHECKPOINTER:

  from langgraph.checkpoint.postgres import PostgresSaver
  checkpointer = PostgresSaver(conn_string="postgresql://...")
  app = graph.compile(checkpointer=checkpointer)

This gives you:
  • Mid-execution persistence — crash recovery, pause/resume
  • Async execution — FastAPI endpoint + async checkpointer
  • State inspection — query Postgres to see agent progress
  • Human-in-the-loop — interrupt at any node, wait for approval

For the MCP server: scale with multiple Prism instances behind a
load balancer. Each handles its own JSON-RPC stdio connection.
Supabase (cloud backend) handles concurrent ledger writes natively.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import TypedDict, Literal
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph.

    Attributes:
        query: The user's original research question.
        plan: The agent's research plan (what to search for).
        search_results: Raw results from knowledge/web search.
        analysis: The agent's analysis of the search results.
        answer: The final synthesized answer.
        iterations: How many research loops the agent has done.
        max_iterations: Safety cap to prevent infinite loops.
        status: Current phase of the research pipeline.
        messages: Full conversation history for the LLM.
    """
    query: str
    plan: str
    search_results: list[str]
    analysis: str
    answer: str
    iterations: int
    max_iterations: int
    status: Literal["planning", "searching", "analyzing", "complete"]
    messages: list[BaseMessage]
