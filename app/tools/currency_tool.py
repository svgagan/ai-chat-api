# app/tools/currency_tool.py
import random

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Mocked exchange rate — no real API, focus is tool selection not accuracy."""
    rate = round(random.uniform(0.5, 90), 4)
    return {
        "amount": amount,
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "converted": round(amount * rate, 2)
    }

CURRENCY_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates. Use this for any currency conversion request.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. INR"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    }
}