# mcp_servers/currency_server.py
from mcp.server.mcpserver import MCPServer
import random

mcp = MCPServer("currency-service")

@mcp.tool()
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another using current exchange rates."""
    rate = round(random.uniform(0.5, 90), 4)
    return {
        "amount": amount,
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "converted": round(amount * rate, 2)
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004)