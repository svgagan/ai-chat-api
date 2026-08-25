# app/tools/time_tool.py
from datetime import datetime, timezone

def get_current_time() -> dict:
    now = datetime.now(timezone.utc)
    return {"utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

TIME_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time in UTC. Use this whenever the user asks what time or date it is right now.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}