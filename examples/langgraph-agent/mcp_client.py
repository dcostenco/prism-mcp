"""
Prism MCP Client Bridge — Raw JSON-RPC over stdio
===================================================
Connects the LangGraph agent to a running Prism MCP server via the
Model Context Protocol (JSON-RPC 2.0 over stdio).

This implements the MCP client protocol WITHOUT the official SDK,
demonstrating deep understanding of the underlying protocol.

MCP Protocol Flow:
  1. Start Prism MCP server as a subprocess
  2. Send JSON-RPC requests via stdin
  3. Read JSON-RPC responses from stdout
  4. Wrap MCP tools as LangChain StructuredTool objects

Architecture:
  ┌─────────────────┐  stdin (JSON-RPC)  ┌──────────────────┐
  │  LangGraph Agent │ ───────────────── →│  Prism MCP       │
  │  (Python)        │                    │  Server (Node.js)│
  │                  │← ─────────────── ─│                  │
  │  PrismMCPBridge  │  stdout (JSON-RPC) │  20+ tools       │
  └─────────────────┘                    └──────────────────┘
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Any, Optional
from langchain_core.tools import StructuredTool


class PrismMCPBridge:
    """MCP Client that connects to Prism MCP server via stdio.

    Implements the Model Context Protocol (JSON-RPC 2.0) directly,
    without requiring the official MCP SDK (which needs Python 3.10+).

    Usage:
        bridge = PrismMCPBridge()
        bridge.connect()
        tools = bridge.get_langchain_tools()
        result = bridge.call_tool("knowledge_search", {"query": "RAG"})
        bridge.close()
    """

    def __init__(
        self,
        command: str = "node",
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ):
        """Initialize the MCP bridge.

        TRANSPORT LESSON LEARNED:
        Using `npx` as a wrapper disrupts the continuous stdio piping
        required for MCP's JSON-RPC heartbeat. npx acts as a middleman
        process that can close the pipes prematurely. By pointing
        directly at the compiled Node binary (`node dist/server.js`),
        we maintain the persistent transport lifecycle — exactly how
        MCP Hosts like Claude Desktop manage their connections.

        Args:
            command: Command to start the MCP server (default: "node").
            args: Arguments for the server command.
            env: Environment variables for the server process.
        """
        self.command = command
        self.args = args or ["dist/server.js"]
        self.env = {**os.environ, **(env or {})}
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._reader_thread: Optional[threading.Thread] = None
        self._responses: dict[int, Any] = {}
        self._notifications: list[dict] = []
        self._tools_cache: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Phase 1: MCP Connection — Establish stdio transport
    # ------------------------------------------------------------------

    def connect(self) -> dict:
        """Start the Prism MCP server and initialize the MCP session.

        Returns:
            The server's initialization response (capabilities, version).
        """
        print(f"🔌 Starting Prism MCP server: {self.command} {' '.join(self.args)}")

        # Start the server as a subprocess with stdio pipes
        #
        # KNOWN LIMITATION — Windows terminal popup:
        # On Windows, subprocess.Popen opens a visible console for each child
        # process. The standard fix is CREATE_NO_WINDOW (0x08000000), which
        # we apply below for our direct Popen call. However, this does NOT
        # fix popups from the official Anthropic `mcp` SDK (StdioServerParameters),
        # because the SDK delegates process creation to `anyio`, which strips
        # out creationflags. The only SDK-level fix is to migrate from
        # stdio_client → sse_client (HTTP transport, zero subprocesses).
        # See README.md → Known Limitations for the full explanation.
        CREATE_NO_WINDOW = 0x08000000
        creation_flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            bufsize=0,  # Unbuffered for real-time communication
            creationflags=creation_flags,
        )

        # Start background thread to read stderr (server logs)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        # MCP initialization handshake (JSON-RPC 2.0)
        init_response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
            },
            "clientInfo": {
                "name": "langgraph-research-agent",
                "version": "1.0.0",
            },
        })

        # Send initialized notification (completes the handshake)
        self._send_notification("notifications/initialized", {})

        server_info = init_response.get("result", {}).get("serverInfo", {})
        server_name = server_info.get("name", "Unknown")
        server_version = server_info.get("version", "Unknown")
        print(f"✅ Connected to {server_name} v{server_version}")

        capabilities = init_response.get("result", {}).get("capabilities", {})
        print(f"   Capabilities: {list(capabilities.keys())}")

        return init_response

    def close(self):
        """Gracefully shut down the MCP server."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            print("🔌 Disconnected from Prism MCP server")

    # ------------------------------------------------------------------
    # Phase 2: Tool Discovery — List and convert MCP tools
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Fetch all available tools from the Prism MCP server.

        Returns:
            List of tool definitions with name, description, and inputSchema.
        """
        if self._tools_cache:
            return self._tools_cache

        response = self._send_request("tools/list", {})
        tools = response.get("result", {}).get("tools", [])
        self._tools_cache = tools
        print(f"🛠️  Discovered {len(tools)} MCP tools")
        return tools

    def get_langchain_tools(self) -> list[StructuredTool]:
        """Convert all Prism MCP tools into LangChain StructuredTool objects.

        This is the key integration point — LangGraph can now use
        any Prism MCP tool as if it were a native LangChain tool.

        Returns:
            List of LangChain StructuredTool wrappers.
        """
        mcp_tools = self.list_tools()
        langchain_tools = []

        for tool_def in mcp_tools:
            tool_name = tool_def["name"]
            tool_desc = tool_def.get("description", f"MCP tool: {tool_name}")

            # Create a closure that calls the MCP tool
            def make_executor(name: str):
                def executor(**kwargs) -> str:
                    return self.call_tool(name, kwargs)
                return executor

            lc_tool = StructuredTool.from_function(
                func=make_executor(tool_name),
                name=tool_name,
                description=tool_desc[:1024],  # LangChain has a description limit
            )
            langchain_tools.append(lc_tool)

        print(f"🔗 Converted {len(langchain_tools)} MCP tools → LangChain format")
        return langchain_tools

    # ------------------------------------------------------------------
    # Phase 3: Tool Execution — Call Prism MCP tools
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """Execute a Prism MCP tool and return the result.

        Args:
            tool_name: Name of the MCP tool (e.g., "knowledge_search").
            arguments: Tool arguments as a dictionary.

        Returns:
            Tool result as a string.
        """
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })

        result = response.get("result", {})

        # Handle MCP tool response format
        if result.get("isError"):
            return f"[MCP Error] {result.get('content', [{}])[0].get('text', 'Unknown error')}"

        # Extract text content from MCP response
        content_parts = result.get("content", [])
        texts = []
        for part in content_parts:
            if part.get("type") == "text":
                texts.append(part["text"])
            elif part.get("type") == "resource":
                texts.append(f"[Resource: {part.get('resource', {}).get('uri', 'N/A')}]")

        return "\n".join(texts) if texts else str(result)

    # ------------------------------------------------------------------
    # Phase 3 Enhancement: Raw Tool Execution — Preserves content blocks
    # ------------------------------------------------------------------
    # Why this exists:
    # call_tool() above flattens all content blocks into a single string.
    # This loses the Phase 1 MemoryTrace data returned as content[1].
    #
    # call_tool_raw() preserves the full MCP response structure, allowing
    # consumers (like PrismMemoryRetriever) to access content[0] (results)
    # and content[1] (trace metadata) independently.
    #
    # BACKWARD COMPATIBILITY: call_tool() is NOT modified. Existing
    # consumers (agent.py, search_node, demo scripts) continue to work
    # exactly as before. Only new code that needs trace data uses this.

    def call_tool_raw(self, tool_name: str, arguments: dict = None) -> dict:
        """Execute a Prism MCP tool and return the FULL structured response.

        Unlike call_tool() which flattens content blocks into a string,
        this method preserves the MCP response structure so that callers
        can inspect individual content blocks (e.g., content[1] for trace).

        Phase 1 Context:
            When enable_trace=True is passed to session_search_memory or
            knowledge_search, the handler returns TWO content blocks:
              content[0] → human-readable search results (text)
              content[1] → MemoryTrace JSON (latency, scores, strategy)

            call_tool() would concatenate both into one string.
            call_tool_raw() keeps them separate for programmatic access.

        Args:
            tool_name: Name of the MCP tool (e.g., "session_search_memory").
            arguments: Tool arguments as a dictionary.

        Returns:
            dict with keys:
              - "content": list of content block dicts, each with "type" and "text"
              - "isError": bool indicating if the tool call failed

        Example:
            raw = bridge.call_tool_raw("session_search_memory", {
                "query": "RAG patterns",
                "enable_trace": True
            })
            results_text = raw["content"][0]["text"]      # Human-readable
            trace_json   = raw["content"][1]["text"]       # MemoryTrace JSON
        """
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })

        result = response.get("result", {})

        # Return the full structured result — content blocks intact
        return {
            "content": result.get("content", []),
            "isError": result.get("isError", False),
        }

    # ------------------------------------------------------------------
    # Phase 4: MCP Resources — Read Prism memory/context
    # ------------------------------------------------------------------

    def read_resource(self, uri: str) -> str:
        """Read a Prism MCP resource by URI.

        Args:
            uri: Resource URI (e.g., "memory://prism-mcp/handoff").

        Returns:
            Resource content as string.
        """
        response = self._send_request("resources/read", {"uri": uri})
        contents = response.get("result", {}).get("contents", [])
        return "\n".join(c.get("text", "") for c in contents)

    def list_resources(self) -> list[dict]:
        """List all available Prism MCP resources.

        Returns:
            List of resource definitions with uri, name, description.
        """
        response = self._send_request("resources/list", {})
        return response.get("result", {}).get("resources", [])

    # ------------------------------------------------------------------
    # Internal: JSON-RPC 2.0 transport over stdio
    # ------------------------------------------------------------------

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC 2.0 request and wait for the response.

        Args:
            method: The RPC method name (e.g., "tools/list").
            params: Method parameters.

        Returns:
            The JSON-RPC response as a dictionary.
        """
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        # Send as a single line of JSON followed by newline
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode("utf-8"))
        self.process.stdin.flush()

        # Read lines until we get a response with our ID
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise ConnectionError("MCP server closed the connection")

            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip non-JSON lines (server logs, etc.)

            # Check if this is our response
            if response.get("id") == self._request_id:
                if "error" in response:
                    error = response["error"]
                    print(f"   ⚠️ MCP Error [{error.get('code')}]: {error.get('message')}")
                return response

            # Store notifications for later processing
            if "method" in response and "id" not in response:
                self._notifications.append(response)

    def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC 2.0 notification (no response expected).

        Args:
            method: The notification method name.
            params: Notification parameters.
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode("utf-8"))
        self.process.stdin.flush()

    def _read_stderr(self):
        """Background thread: read server stderr for logs/errors."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stderr.readline()
                if line:
                    pass  # Silently consume server logs
                    # Uncomment for debugging: print(f"[MCP stderr] {line.decode().strip()}")
            except Exception:
                break


# ------------------------------------------------------------------
# Convenience functions for quick usage
# ------------------------------------------------------------------

def connect_to_prism(env: Optional[dict] = None) -> PrismMCPBridge:
    """Quick-connect to a local Prism MCP server.

    Args:
        env: Optional environment variables (GOOGLE_API_KEY, BRAVE_API_KEY, etc.)

    Returns:
        Connected PrismMCPBridge instance.

    Example:
        bridge = connect_to_prism({"BRAVE_API_KEY": "your-key"})
        tools = bridge.get_langchain_tools()
        bridge.close()
    """
    bridge = PrismMCPBridge(env=env)
    bridge.connect()
    return bridge


def demo():
    """Demo: connect to Prism MCP and list all available tools."""
    print("=" * 60)
    print("🔌 Prism MCP Client Bridge — Demo")
    print("=" * 60)

    bridge = PrismMCPBridge()

    try:
        bridge.connect()

        # List all tools
        tools = bridge.list_tools()
        print(f"\n📋 Available Prism MCP Tools ({len(tools)}):")
        for tool in tools:
            print(f"   • {tool['name']}: {tool.get('description', 'N/A')[:80]}")

        # List resources
        resources = bridge.list_resources()
        print(f"\n📦 Available Resources ({len(resources)}):")
        for res in resources:
            print(f"   • {res.get('uri', 'N/A')}: {res.get('name', 'N/A')}")

        # Test a tool call
        print("\n🧪 Test: calling knowledge_search...")
        result = bridge.call_tool("knowledge_search", {"query": "RAG"})
        print(f"   Result: {result[:200]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure Prism MCP server is available via 'npx -y prism-mcp-server'")

    finally:
        bridge.close()


if __name__ == "__main__":
    demo()
