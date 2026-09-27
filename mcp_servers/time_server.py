# mcp_servers/time_server.py
from mcp.server.mcpserver import MCPServer
from datetime import datetime, timezone

mcp = MCPServer("time-service")

@mcp.tool()
def current_time() -> dict:
    """Get the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return {"utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003)