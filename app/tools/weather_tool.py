# app/tools/weather_tool.py
import random

def get_weather(city: str) -> dict:
    """
    Mocked weather function — no real API needed for learning
    the tool-calling mechanism itself.
    """
    conditions = ["sunny", "partly cloudy", "rainy", "clear"]
    return {
        "city": city,
        "temperature_celsius": random.randint(18, 35),
        "condition": random.choice(conditions)
    }


WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city. Use this whenever the user asks about weather, temperature, or conditions in a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. Bangalore, London"
                }
            },
            "required": ["city"]
        }
    }
}