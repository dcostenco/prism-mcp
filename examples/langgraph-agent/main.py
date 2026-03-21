"""
Prism Research Agent — CLI Entry Point
=======================================
Run the LangGraph research agent from the command line.

Usage:
    # Standalone mode (built-in knowledge base)
    python main.py "How does time travel work in Prism MCP?"

    # MCP-connected mode (connects to running Prism MCP server)
    python main.py --prism "How does time travel work in Prism MCP?"

    # Interactive mode
    python main.py
    python main.py --prism    # interactive + MCP
"""

import sys
from agent import run_research, set_mcp_bridge


def main():
    args = sys.argv[1:]
    use_prism = False
    query = None

    # Parse arguments
    if "--prism" in args:
        use_prism = True
        args.remove("--prism")

    if args:
        query = " ".join(args)

    # ── MCP-Connected Mode ──
    if use_prism:
        try:
            from mcp_client import PrismMCPBridge
            bridge = PrismMCPBridge()
            bridge.connect()
            set_mcp_bridge(bridge)
            print()
        except Exception as e:
            print(f"⚠️  Could not connect to Prism MCP: {e}")
            print("   Falling back to standalone mode.\n")

    # ── Single query mode ──
    if query:
        run_research(query)
        return

    # ── Interactive REPL mode ──
    mode = "MCP-connected" if use_prism else "standalone"
    print(f"\n🔬 Prism Research Agent — Interactive Mode ({mode})")
    print("   Type a question and press Enter. Type 'quit' to exit.\n")

    example_queries = [
        "How does time travel work in Prism MCP?",
        "What is session telepathy?",
        "Compare Prism MCP to other MCP servers",
        "How does progressive context loading work?",
    ]
    print("   Example queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"     {i}. {q}")
    print()

    while True:
        try:
            user_input = input("❓ Ask: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            # Allow picking example by number
            if user_input.isdigit() and 1 <= int(user_input) <= len(example_queries):
                user_input = example_queries[int(user_input) - 1]
                print(f"   → {user_input}")
            print()
            run_research(user_input)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

    # Cleanup MCP connection
    if use_prism:
        try:
            from mcp_client import PrismMCPBridge
            # Bridge cleanup is handled by the module
        except Exception:
            pass


if __name__ == "__main__":
    main()
