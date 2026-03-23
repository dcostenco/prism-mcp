#!/bin/bash

# Silently kill stale processes (redirecting BOTH stdout AND stderr to /dev/null)
pkill -15 -f "node.*prism/dist/server.js" >/dev/null 2>&1
sleep 0.5
pkill -9 -f "node.*prism/dist/server.js" >/dev/null 2>&1

# Start the actual server — exec keeps stdio pipes attached for MCP
exec /usr/bin/env node /Users/admin/prism/dist/server.js "$@"
