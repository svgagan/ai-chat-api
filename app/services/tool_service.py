# app/services/tool_service.py
import json
import asyncio
from app.services.ai_service import ai_service
from app.services.mcp_client_service import mcp_client_service


class ToolService:

    async def chat_with_tools(self, user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        tools = await mcp_client_service.get_tool_definitions()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        max_iterations = 5
        for _ in range(max_iterations):
            response = await asyncio.to_thread(
                ai_service.chat_with_tools,
                messages=messages,
                tools=tools
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result = await mcp_client_service.call_tool(tool_call.function.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        return "I wasn't able to complete this request after multiple tool calls."


tool_service = ToolService()