# app/services/tool_service.py
import json
from app.services.ai_service import ai_service
from app.tools.weather_tool import get_weather, WEATHER_TOOL_DEFINITION
from app.tools.calculator_tool import calculate, CALCULATOR_TOOL_DEFINITION
from app.tools.time_tool import get_current_time, TIME_TOOL_DEFINITION
from app.tools.currency_tool import convert_currency, CURRENCY_TOOL_DEFINITION
from app.tools.knowledge_base_tool import search_knowledge_base, KNOWLEDGE_BASE_TOOL_DEFINITION


# Registry mapping tool names to actual callable functions
# This indirection is the key architectural piece: the model
# only ever sees names and schemas, never actual code
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "convert_currency": convert_currency,
    "search_knowledge_base": search_knowledge_base,
}

AVAILABLE_TOOLS = [
    WEATHER_TOOL_DEFINITION,
    CALCULATOR_TOOL_DEFINITION,
    TIME_TOOL_DEFINITION,
    CURRENCY_TOOL_DEFINITION,
    KNOWLEDGE_BASE_TOOL_DEFINITION,
]

class ToolService:

    def execute_tool_call(self, tool_call) -> str:
        """
        Executes exactly ONE tool call the model requested.
        Returns a JSON string — tool results must be strings
        when fed back into the conversation.
        """
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name not in TOOL_REGISTRY:
            # Model hallucinated a tool name that doesn't exist —
            # this happens, and must be handled, not trusted
            return json.dumps({"error": f"Unknown tool: {function_name}"})

        try:
            function = TOOL_REGISTRY[function_name]
            result = function(**arguments)
            return json.dumps(result)
        except Exception as e:
            # Execution failure — model SEES this, per your Q2 reasoning
            return json.dumps({"error": str(e)})

    def chat_with_tools(self, user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Full tool-calling round trip. Loops until the model
        gives a direct text answer instead of requesting more tools.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        max_iterations = 5  # safety cap — never loop forever
        for _ in range(max_iterations):
            response = ai_service.chat_with_tools(
                messages=messages,
                tools=AVAILABLE_TOOLS
            )
            message = response.choices[0].message

            if not message.tool_calls:
                # Model gave a direct answer — done
                return message.content

            # Model wants to call one or more tools
            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                result = self.execute_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "I wasn't able to complete this request after multiple tool calls."


tool_service = ToolService()