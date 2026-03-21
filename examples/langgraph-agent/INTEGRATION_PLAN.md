# Prism MCP + LangGraph Integration Plan

> How to connect this LangGraph research agent directly to Prism MCP's tools for production-grade agentic workflows.

## Current Architecture (Standalone)

```
┌────────────────────┐      ┌──────────────────┐
│  LangGraph Agent   │ ───→ │  Built-in KB     │
│  (Python)          │      │  (tools.py)      │
│                    │ ───→ │  Brave API       │
│  plan → search →   │      │  (web search)    │
│  analyze → decide  │      └──────────────────┘
└────────────────────┘
```

## Target Architecture (Prism-Integrated)

```
┌────────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  LangGraph Agent   │      │  MCP Client      │      │  Prism MCP       │
│  (Python)          │ ───→ │  (mcp SDK)       │ ───→ │  Server          │
│                    │      │                  │      │                  │
│  plan → search →   │      │  • call_tool()   │      │  • knowledge_    │
│  analyze → decide  │      │  • read_resource │      │    search        │
│  (LangGraph graph) │      │  • list_tools()  │      │  • session_      │
│                    │      │                  │      │    search_memory │
│  STATE:            │      └──────────────────┘      │  • brave_web_    │
│  • query           │                                │    search        │
│  • search_results  │                                │  • session_      │
│  • analysis        │                                │    save_ledger   │
│  • answer          │                                │  • gemini_       │
└────────────────────┘                                │    research_     │
                                                      │    paper_analysis│
                                                      └──────────────────┘
```

## Integration Steps

### Phase 1: MCP Client Connection

Install the Python MCP SDK and connect to Prism:

```python
# Install
# pip install mcp

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_to_prism():
    """Connect to Prism MCP server via stdio."""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "prism-mcp-server"],
        env={
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
            "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", ""),
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Connected! {len(tools.tools)} tools available")

            return session
```

### Phase 2: Replace Built-in Tools with MCP Calls

Update `tools.py` to call Prism MCP instead of the built-in knowledge base:

```python
async def search_prism_knowledge_mcp(session, query: str) -> list[dict]:
    """Search via Prism's session_search_memory tool."""
    result = await session.call_tool(
        "session_search_memory",
        arguments={"query": query, "limit": 5}
    )
    return result.content

async def search_knowledge_mcp(session, query: str) -> list[dict]:
    """Search via Prism's knowledge_search tool."""
    result = await session.call_tool(
        "knowledge_search",
        arguments={"query": query, "limit": 10}
    )
    return result.content

async def search_web_mcp(session, query: str) -> str:
    """Search via Prism's brave_web_search tool."""
    result = await session.call_tool(
        "brave_web_search",
        arguments={"query": query, "count": 5}
    )
    return result.content
```

### Phase 3: Add Session Memory to the Agent

Make the agent save its research sessions to Prism's ledger:

```python
async def save_research_session(session, query: str, answer: str, iterations: int):
    """Save the research session to Prism's ledger."""
    await session.call_tool(
        "session_save_ledger",
        arguments={
            "project": "langgraph-research-agent",
            "conversation_id": str(uuid.uuid4()),
            "summary": f"Researched: {query}. Completed in {iterations} iterations.",
            "decisions": [f"Answer generated for: {query[:100]}"],
            "todos": [],
        }
    )
```

### Phase 4: Add to the LangGraph as New Nodes

```python
# In agent.py, add a "save_session" node after "answer"

def save_session_node(state: AgentState) -> dict:
    """Save research results to Prism MCP memory."""
    # This creates a persistent research trail
    save_research_session(
        mcp_session,
        state["query"],
        state["answer"],
        state["iterations"]
    )
    return state

# Update graph
graph.add_node("save_session", save_session_node)
graph.add_edge("answer", "save_session")
graph.add_edge("save_session", END)
```

## Integration Benefits

| Feature | Standalone | Prism-Integrated |
|---|---|---|
| Knowledge base | 12 static entries | Full Prism session memory (growing) |
| Web search | Direct Brave API | Via Prism (with code_mode templates) |
| Research memory | Lost between runs | Persisted in Prism ledger |
| Cross-agent access | None | Other agents can read research results |
| Time travel | None | Can revert to any research state |
| Security | None | Prompt injection scanning |

## File Changes Summary

| File | Change |
|---|---|
| `tools.py` | Add `PrismMCPClient` class wrapping MCP SDK calls |
| `agent.py` | Add `save_session` node; make search/analyze async |
| `state.py` | Add `mcp_session` field to `AgentState` |
| `main.py` | Add `--prism` flag to enable MCP integration |
| `requirements.txt` | Add `mcp` package |
