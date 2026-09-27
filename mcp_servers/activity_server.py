# mcp_servers/activity_server.py
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("activity-service")

@mcp.tool()
def activity_suggester(weather_condition: str, budget_inr: float) -> dict:
    """Suggest an activity based on current weather condition and budget in INR."""
    indoor_options = [
        {"name": "Escape room", "cost_inr": 800},
        {"name": "Movie theater", "cost_inr": 400},
        {"name": "Board game cafe", "cost_inr": 600},
    ]
    outdoor_options = [
        {"name": "Trekking trail", "cost_inr": 300},
        {"name": "City walking tour", "cost_inr": 500},
        {"name": "Park picnic", "cost_inr": 200},
    ]
    pool = indoor_options if weather_condition in ("rainy", "cloudy") else outdoor_options
    affordable = [opt for opt in pool if opt["cost_inr"] <= budget_inr]
    return {
        "weather_condition": weather_condition,
        "budget_inr": budget_inr,
        "affordable_options": affordable if affordable else "none within budget"
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005)