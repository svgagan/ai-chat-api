# mcp_servers/weather_server.py
from mcp.server.mcpserver import MCPServer
import random

mcp = MCPServer("weather-service")

@mcp.tool()
def weather(city: str) -> dict:
    """Get the current weather for a given city."""
    conditions = ["sunny", "partly cloudy", "rainy", "clear"]
    return {
        "city": city,
        "temperature_celsius": random.randint(18, 35),
        "condition": random.choice(conditions)
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)