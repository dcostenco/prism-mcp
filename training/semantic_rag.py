#!/usr/bin/env python3
"""
Semantic RAG for Tool Selection (R5-4)

Generates and retrieves tool embeddings for Top-K injection.
Instead of passing ALL tools in the system prompt (2000+ tokens),
retrieve only the 5 most relevant tools per query (~300 tokens).

Usage:
    # Generate embeddings (one-time)
    python semantic_rag.py generate

    # Retrieve top-5 tools for a query
    python semantic_rag.py retrieve "Load my project context"

    # Test retrieval accuracy
    python semantic_rag.py test "Load context for my project"
"""
import json
import math
import os
import sys
import urllib.request

TOOL_SCHEMA_PATH = os.environ.get(
    "PRISM_TOOL_SCHEMA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "tool_schema.json")
)
EMBEDDINGS_PATH = os.environ.get(
    "PRISM_EMBEDDINGS_PATH",
    os.path.join(os.path.dirname(__file__), "data", "tool_embeddings.json")
)
OLLAMA_EMBEDDING_MODEL = os.environ.get("PRISM_EMBED_MODEL", "nomic-embed-text")
OLLAMA_API = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def get_embedding(text: str) -> list:
    """Get embedding vector from Ollama nomic-embed-text."""
    req = urllib.request.Request(
        f"{OLLAMA_API}/api/embeddings",
        data=json.dumps({"model": OLLAMA_EMBEDDING_MODEL, "prompt": text}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        res = urllib.request.urlopen(req, timeout=30)
        return json.loads(res.read().decode('utf-8'))['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_embeddings():
    """Generate and save Nomic embeddings for all MCP tools."""
    print("Generating Nomic embeddings for MCP tools...")

    if not os.path.exists(TOOL_SCHEMA_PATH):
        print(f"ERROR: Tool schema not found at {TOOL_SCHEMA_PATH}")
        print("Create data/tool_schema.json with your tool definitions first.")
        return

    with open(TOOL_SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    tools = schema.get("tools", [])
    print(f"Found {len(tools)} tools.")

    embeddings = {}
    for t in tools:
        name = t["name"]
        desc = t.get("description", "")
        # Include param names in embedding text for better matching
        params = t.get("parameters", {}).get("properties", {})
        param_text = ", ".join(params.keys()) if params else ""
        embed_text = f"{name}: {desc}. Parameters: {param_text}"

        print(f"  Embedding: {name}")
        emb = get_embedding(embed_text)
        if emb:
            embeddings[name] = {
                "schema": t,
                "embedding": emb,
                "embed_text": embed_text,
            }

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")


def load_embeddings() -> dict:
    """Load precomputed tool embeddings."""
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"WARNING: No embeddings at {EMBEDDINGS_PATH}. Run 'python semantic_rag.py generate' first.")
        return {}
    with open(EMBEDDINGS_PATH, "r") as f:
        return json.load(f)


def retrieve_top_k(query: str, k: int = 5) -> list:
    """R5-4: Retrieve the top-k most relevant tools for a query.

    Args:
        query: User's natural language query.
        k: Number of tools to return (default 5).

    Returns:
        List of tool schema dicts sorted by relevance (highest first).
    """
    embeddings = load_embeddings()
    if not embeddings:
        return []

    query_emb = get_embedding(query)
    if not query_emb:
        return []

    # Score all tools
    scores = []
    for tool_name, data in embeddings.items():
        sim = cosine_similarity(query_emb, data["embedding"])
        scores.append((sim, tool_name, data["schema"]))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[0], reverse=True)

    # Return top-k schemas
    results = []
    for sim, name, schema in scores[:k]:
        results.append(schema)
        print(f"  RAG: {name} (sim={sim:.4f})")

    return results


def build_rag_system_prompt(query: str, k: int = 5) -> str:
    """R5-4: Build a system prompt with only the top-k relevant tools.

    This reduces system prompt from ~2000 tokens to ~300 tokens.
    """
    from config import format_system_prompt

    tools = retrieve_top_k(query, k=k)
    if not tools:
        print("WARNING: No tools retrieved. Using empty tool list.")
    return format_system_prompt(tools)


def test_retrieval(query: str):
    """Test retrieval accuracy for a query."""
    print(f"\nQuery: {query}")
    print("-" * 50)
    tools = retrieve_top_k(query, k=5)
    print(f"\nRetrieved {len(tools)} tools:")
    for i, t in enumerate(tools):
        print(f"  {i+1}. {t['name']}: {t.get('description', '')[:80]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python semantic_rag.py generate       # Generate embeddings")
        print("  python semantic_rag.py retrieve 'query'  # Retrieve top-5 tools")
        print("  python semantic_rag.py test 'query'      # Test retrieval")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "generate":
        generate_embeddings()
    elif cmd in ("retrieve", "test"):
        if len(sys.argv) < 3:
            print("ERROR: Provide a query string.")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        if cmd == "retrieve":
            tools = retrieve_top_k(query)
            print(json.dumps([t["name"] for t in tools], indent=2))
        else:
            test_retrieval(query)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
