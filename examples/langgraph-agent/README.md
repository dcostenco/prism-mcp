# 🔬 Prism Research Agent — LangGraph Portfolio

> A multi-step AI research agent built with **LangGraph** + **Gemini**, integrated with the **Prism MCP** agentic memory server. Demonstrates autonomous planning, tool use, decision-making, and persistent memory.

## Architecture

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐
│  Plan   │ ──→ │  Search  │ ──→ │ Analyze  │     │ Save Session │
└─────────┘     └──────────┘     └────┬─────┘     └──────┬───────┘
                     ↑                │                   │
                     │           ╔════╧════╗              │
                     │           ║ Done?   ║           ┌──┴──┐
                     │           ╚════╤════╝           │ END │
                     │      No       │    Yes          └─────┘
                     └───────────────┘     ↓
                                      ┌─────────┐
                                      │ Answer  │ ──→ Save Session ──→ END
                                      └─────────┘
```

**5 nodes** · **1 conditional edge** · **3 tools** · **Agentic Memory**

## Key Agentic AI Patterns Demonstrated

| Pattern | Implementation | Talking Point |
|---|---|---|
| **StateGraph** | `build_research_agent()` in `agent.py` | Directed graph of processing nodes with typed shared state |
| **Conditional Edges** | `should_continue()` — agent decides to loop or answer | Autonomous decision-making based on research quality |
| **Tool Use** | 3 tools: Prism KB, MCP glossary, Brave web search | Multi-source information gathering |
| **Autonomous Looping** | Searches again if results are INCOMPLETE | Self-evaluating research quality |
| **Agentic Memory** | `save_session_node` persists to ledger | Agent doesn't just answer and forget — it commits findings |
| **MCP Integration** | `PrismMCPBridge` — raw JSON-RPC over stdio | Deep protocol understanding, decoupled architecture |
| **Dual-Mode Design** | Standalone (built-in KB) or MCP-connected | Architecture is tool-source agnostic |

## Quick Start

```bash
# Prerequisites
pip install langgraph langchain langchain-google-genai

# Set API key
export GOOGLE_API_KEY="your-gemini-key"

# Run a query (standalone mode)
python main.py "How does time travel work in Prism MCP?"

# Interactive mode
python main.py

# MCP-connected mode (requires local Prism server)
python main.py --prism "How does agent telepathy work?"
```

## Project Structure

```
examples/langgraph-agent/
├── state.py           # AgentState TypedDict — shared state schema
├── tools.py           # 3 agent tools + persistent ledger
├── agent.py           # LangGraph StateGraph (5 nodes, conditional edges)
├── mcp_client.py      # Raw JSON-RPC MCP client bridge (Python 3.9+)
├── main.py            # CLI entry point (--prism flag for MCP mode)
├── research_ledger.json  # Auto-generated: persistent research memory
├── README.md
└── INTEGRATION_PLAN.md
```

## How It Works

### 1. Plan → Search → Analyze → Decide → Answer → Save

Each query flows through the agent's reasoning pipeline:

1. **Plan** — LLM analyzes the query and creates a research strategy
2. **Search** — Executes tools (Prism KB, glossary, web search)
3. **Analyze** — LLM evaluates if findings are COMPLETE or INCOMPLETE
4. **Decide** — Conditional edge: loop back to search, or proceed to answer
5. **Answer** — LLM synthesizes all research into a comprehensive response
6. **Save Session** — Persists findings to memory ledger (agentic memory)

### 2. MCP Client Bridge — Transport Lesson Learned

```
When bridging LangGraph with Prism MCP, I initially tried invoking the
server via npx. However, the Python client would drop the connection.
I realized npx acts as a wrapper that disrupts the continuous stdio
piping required for MCP's JSON-RPC heartbeat. By pointing directly
at the compiled Node binary (node dist/server.js), we maintain the
persistent transport lifecycle — exactly how MCP Hosts like Claude
Desktop manage their connections.
```

### 3. Dual-Mode Architecture

The agent's `search_node` is **tool-source agnostic**:

- **Standalone**: `python main.py "query"` → uses built-in Prism KB (12 entries) + Brave API
- **MCP-connected**: `python main.py --prism "query"` → routes through live Prism MCP tools

This proves the architecture is **decoupled** — the LangGraph agent orchestrates reasoning (Plan → Analyze → Decide), while tool execution is abstracted by the MCP protocol.

## Technology Stack

- **Agent Framework**: LangGraph (Python)
- **LLM**: Google Gemini 2.5 Flash Lite (`langchain-google-genai`)
- **MCP Protocol**: Raw JSON-RPC 2.0 over stdio (no SDK, Python 3.9+)
- **Knowledge Base**: Curated Prism MCP documentation (12 entries, 13 glossary terms)
- **Web Search**: Brave Search API (optional, via `BRAVE_API_KEY`)
- **Memory**: Local JSON ledger + Prism MCP `session_save_ledger`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key (free tier available) |
| `BRAVE_API_KEY` | Optional | Brave Search API key for web search |

## Integration with Prism MCP

This agent lives inside the [Prism MCP](https://github.com/yourusername/bcba) repository as a showcase of how LangGraph agents can integrate with MCP servers. See [INTEGRATION_PLAN.md](./INTEGRATION_PLAN.md) for the 4-phase integration roadmap.

## License

MIT — Part of the Prism MCP project.
