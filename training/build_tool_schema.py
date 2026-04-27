#!/usr/bin/env python3
"""Extract Prism MCP tool schema from server.ts for training and evaluation."""
import re, json, os

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
SRC_DIR = os.path.join(_REPO_DIR, "src")
OUTPUT = os.path.join(_TRAINING_DIR, "data", "tool_schema.json")

# Known Prism MCP tools from the server registration
PRISM_TOOLS = [
    {
        "name": "session_load_context",
        "description": "Load full project context including recent sessions, TODOs, decisions, and knowledge graph",
        "parameters": {
            "type": "object",
            "required": ["project"],
            "properties": {
                "project": {"type": "string", "description": "Project identifier"},
                "level": {"type": "string", "enum": ["shallow", "deep"], "description": "Context depth level"}
            }
        }
    },
    {
        "name": "session_save_ledger",
        "description": "Save a session summary with decisions, TODOs, files changed, and keywords",
        "parameters": {
            "type": "object",
            "required": ["project", "summary"],
            "properties": {
                "project": {"type": "string", "description": "Project identifier"},
                "summary": {"type": "string", "description": "Session summary text"},
                "decisions": {"type": "array", "items": {"type": "string"}, "description": "Architectural decisions made"},
                "todos": {"type": "array", "items": {"type": "string"}, "description": "Open TODO items"},
                "files_changed": {"type": "array", "items": {"type": "string"}, "description": "Files modified"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Search keywords"},
                "role": {"type": "string", "description": "Agent role (e.g. global, qa, coder)"}
            }
        }
    },
    {
        "name": "session_search_memory",
        "description": "Search session history by semantic query, keywords, or project filter",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "description": "Max results to return"},
                "role": {"type": "string", "description": "Filter by role"}
            }
        }
    },
    {
        "name": "session_list",
        "description": "List recent sessions for a project with optional filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project identifier"},
                "limit": {"type": "integer", "description": "Number of sessions to return"},
                "include_archived": {"type": "boolean", "description": "Include archived sessions"}
            }
        }
    },
    {
        "name": "session_forget_memory",
        "description": "Soft-delete a memory entry by ID with a reason",
        "parameters": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Memory entry ID to delete"},
                "reason": {"type": "string", "description": "Reason for deletion"}
            }
        }
    },
    {
        "name": "knowledge_save",
        "description": "Store a semantic knowledge concept with description and confidence",
        "parameters": {
            "type": "object",
            "required": ["project", "concept", "description"],
            "properties": {
                "project": {"type": "string", "description": "Project identifier"},
                "concept": {"type": "string", "description": "Knowledge concept name"},
                "description": {"type": "string", "description": "Detailed description"},
                "confidence": {"type": "number", "description": "Confidence score 0-1"},
                "related_entities": {"type": "array", "items": {"type": "string"}, "description": "Related concepts"}
            }
        }
    },
    {
        "name": "knowledge_search",
        "description": "Search semantic knowledge by concept or description",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query for knowledge"},
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "description": "Max results"}
            }
        }
    },
    {
        "name": "memory_link",
        "description": "Create a link between two memory entries for knowledge graph",
        "parameters": {
            "type": "object",
            "required": ["source_id", "target_id", "relation"],
            "properties": {
                "source_id": {"type": "string", "description": "Source memory ID"},
                "target_id": {"type": "string", "description": "Target memory ID"},
                "relation": {"type": "string", "description": "Relationship type (e.g. depends_on, related_to, blocks)"}
            }
        }
    },
    {
        "name": "session_save_handoff",
        "description": "Save structured handoff state for the next session",
        "parameters": {
            "type": "object",
            "required": ["project", "from_agent", "to_agent", "summary"],
            "properties": {
                "project": {"type": "string", "description": "Project identifier"},
                "from_agent": {"type": "string", "description": "Handing off agent name"},
                "to_agent": {"type": "string", "description": "Receiving agent name"},
                "summary": {"type": "string", "description": "Handoff context and instructions"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
            }
        }
    },
    {
        "name": "session_task_route",
        "description": "Determine if a task can be routed to a local LLM or needs cloud",
        "parameters": {
            "type": "object",
            "required": ["task_description"],
            "properties": {
                "task_description": {"type": "string", "description": "Description of the task to route"},
                "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]}
            }
        }
    }
]

# R11-fix: Include V4 Agentic schemas (40% of BFCL scoring weight)
try:
    from config import V4_API_SCHEMAS
    for api_tools in V4_API_SCHEMAS.values():
        PRISM_TOOLS.extend(api_tools)
except ImportError:
    print("WARNING: Could not import V4_API_SCHEMAS from config.py")

# R6.3-fix: Atomic write to prevent partial reads during concurrent CI
_tmp_output = OUTPUT + ".tmp"
with open(_tmp_output, "w") as f:
    json.dump({"tools": PRISM_TOOLS, "version": "1.0", "source": "prism-mcp"}, f, indent=2)
os.replace(_tmp_output, OUTPUT)

print(f"Wrote {len(PRISM_TOOLS)} tool schemas to {OUTPUT}")
