# app/services/agent_service.py
"""
Hand-built ReAct agent loop, calling MCP tools directly.

FROZEN as of Project 10. This file will not receive further
features, this is intentional, not neglect. It exists as the
verified baseline used to compare against LangChain's
create_agent (see langchain_agent_service.py), documented in
the Project 10 blog post. All new agent capability, starting
with Project 11's long-term memory and Jev integration, is
built on top of langchain_agent_service.py going forward.
"""

import json
import asyncio
from app.services.ai_service import ai_service
from app.services.mcp_client_service import mcp_client_service

REACT_SYSTEM_PROMPT = """You are a reasoning agent that solves problems step by step.

CRITICAL: Every single response you give, without exception, must include
a short explanation in your message text of your current reasoning, even
when you are also calling a tool. Never send a tool call with empty message
content. Before EVERY tool call, write one sentence explaining what you know
and why you're calling this specific tool right now.

When you have enough information to fully answer the user's goal, do not call
any more tools. Instead, respond with your reasoning followed by a clear final
answer, starting with "Final Answer:".

Always call only ONE tool at a time, even if you can see you'll need more than
one eventually, so you can incorporate each result before deciding the next step."""


class AgentService:

    async def _execute_tool(self, tool_call, max_retries: int = 1) -> str:
        """
        Routes execution through mcp_client_service, which itself
        decides whether the tool is remote (MCP server) or local
        (direct Python call). This service does not know or care
        which, that decision is fully owned by mcp_client_service.
        """
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        for attempt in range(max_retries + 1):
            try:
                result = await mcp_client_service.call_tool(function_name, arguments)
                return json.dumps(result)
            except Exception as e:
                if attempt < max_retries:
                    continue
                return json.dumps({
                    "error": f"Tool failed after {max_retries + 1} attempts: {str(e)}"
                })

    async def run_agent(self, goal: str, max_iterations: int = 6) -> dict:
        tools = await mcp_client_service.get_tool_definitions()

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": goal}
        ]

        trace = []

        for step in range(1, max_iterations + 1):
            # litellm.completion() is synchronous and blocking; running it
            # directly in this async function would freeze the event loop
            # for every other request FastAPI is handling concurrently.
            # asyncio.to_thread offloads it to a worker thread instead.
            response = await asyncio.to_thread(
                ai_service.chat_with_tools,
                messages=messages,
                tools=tools
            )
            message = response.choices[0].message

            reasoning = getattr(message, "reasoning", None) or message.content
            step_record = {
                "step": step,
                "reasoning_text": reasoning,
                "tool_call": None,
                "observation": None
            }

            if not message.tool_calls:
                step_record["final"] = True
                trace.append(step_record)
                return {
                    "answer": message.content,
                    "steps_taken": step,
                    "trace": trace
                }

            messages.append(message.model_dump())

            tool_call = message.tool_calls[0]
            step_record["tool_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }

            result = await self._execute_tool(tool_call)
            step_record["observation"] = result
            trace.append(step_record)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        return {
            "answer": "I wasn't able to reach a final answer within the step limit.",
            "steps_taken": max_iterations,
            "trace": trace
        }


agent_service = AgentService()