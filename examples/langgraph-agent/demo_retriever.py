"""
Prism Memory Retriever Demo — Phase 3 Integration Proof
========================================================
Demonstrates the PrismMemoryRetriever working as a standard
LangChain retriever with Phase 1 MemoryTrace metadata.

This script:
  1. Connects to a running Prism MCP server via PrismMCPBridge
  2. Creates a PrismMemoryRetriever (BaseRetriever subclass)
  3. Invokes a semantic search query
  4. Prints the Document results + embedded trace metadata
  5. Shows what LangSmith would automatically capture

Prerequisites:
  export GOOGLE_API_KEY="your_gemini_key"
  export SUPABASE_URL="your_supabase_project_url"
  export SUPABASE_SERVICE_KEY="your_supabase_service_role_key"

Usage:
  cd examples/langgraph-agent
  python demo_retriever.py

  # Or with a custom query:
  python demo_retriever.py "RAG architecture patterns"
"""

from __future__ import annotations

import json
import sys

from mcp_client import PrismMCPBridge
from prism_retriever import PrismMemoryRetriever, PrismKnowledgeRetriever


def demo(query: str = "debugging authentication issues"):
    """Run the PrismMemoryRetriever demo.

    Steps:
      1. Connect to Prism MCP server (subprocess stdio)
      2. Create both semantic and keyword retrievers
      3. Run .invoke(query) on each
      4. Pretty-print results + trace metadata
    """
    print("=" * 70)
    print("🧠 Prism Memory Retriever — Phase 3 Demo")
    print("=" * 70)
    print(f"   Query: \"{query}\"")
    print()

    # ── Step 1: Connect to Prism MCP ──────────────────────────────────
    bridge = PrismMCPBridge()

    try:
        bridge.connect()
        print()

        # ── Step 2: Semantic Search (PrismMemoryRetriever) ────────────
        print("-" * 70)
        print("🔍 Semantic Search via PrismMemoryRetriever")
        print("-" * 70)

        memory_retriever = PrismMemoryRetriever(
            mcp_client=bridge,
            enable_trace=True,  # Phase 1: request MemoryTrace
            similarity_threshold=0.5,
            max_results=3,
        )

        # .invoke() calls _get_relevant_documents() (sync path)
        memory_docs = memory_retriever.invoke(query)

        if memory_docs:
            for i, doc in enumerate(memory_docs):
                print(f"\n📄 Document {i + 1}:")
                print(f"   Content: {doc.page_content[:200]}...")
                print(f"   Source:  {doc.metadata.get('source', 'N/A')}")
                print(f"   Tool:    {doc.metadata.get('tool', 'N/A')}")

                # Phase 1: Display trace metadata
                trace = doc.metadata.get("trace", {})
                if trace:
                    print(f"\n   🔬 MemoryTrace (Phase 1):")
                    print(f"      Strategy:     {trace.get('strategy', 'N/A')}")
                    print(f"      Top Score:    {trace.get('top_score', 'N/A')}")
                    print(f"      Result Count: {trace.get('result_count', 'N/A')}")

                    latency = trace.get("latency", {})
                    if latency:
                        print(f"      Latency:")
                        print(f"        embedding_ms: {latency.get('embedding_ms', 'N/A')}")
                        print(f"        storage_ms:   {latency.get('storage_ms', 'N/A')}")
                        print(f"        total_ms:     {latency.get('total_ms', 'N/A')}")
                else:
                    print(f"\n   ⚠️ No trace metadata (enable_trace might be False)")
        else:
            print("   No results returned from semantic search.")

        print()

        # ── Step 3: Keyword Search (PrismKnowledgeRetriever) ──────────
        print("-" * 70)
        print("📚 Keyword Search via PrismKnowledgeRetriever")
        print("-" * 70)

        knowledge_retriever = PrismKnowledgeRetriever(
            mcp_client=bridge,
            enable_trace=True,
            max_results=5,
        )

        knowledge_docs = knowledge_retriever.invoke(query)

        if knowledge_docs:
            for i, doc in enumerate(knowledge_docs):
                print(f"\n📄 Document {i + 1}:")
                print(f"   Content: {doc.page_content[:200]}...")

                trace = doc.metadata.get("trace", {})
                if trace:
                    print(f"   🔬 Trace Strategy: {trace.get('strategy', 'N/A')}")
                    latency = trace.get("latency", {})
                    print(f"   ⏱️  Total: {latency.get('total_ms', 'N/A')}ms")
        else:
            print("   No results returned from keyword search.")

        # ── Step 4: Show what LangSmith sees ──────────────────────────
        print()
        print("-" * 70)
        print("📊 What LangSmith Would See (Document.metadata):")
        print("-" * 70)
        if memory_docs:
            print(json.dumps(memory_docs[0].metadata, indent=2, default=str))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure environment variables are set:")
        print("     export GOOGLE_API_KEY=...")
        print("     export SUPABASE_URL=...")
        print("     export SUPABASE_SERVICE_KEY=...")
        import traceback
        traceback.print_exc()

    finally:
        bridge.close()

    print()
    print("=" * 70)
    print("✅ Demo complete. The trace data above is what LangSmith captures")
    print("   automatically when this retriever is used in a LangChain chain.")
    print("=" * 70)


if __name__ == "__main__":
    # Accept optional query from command line
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "debugging authentication issues"
    demo(user_query)
