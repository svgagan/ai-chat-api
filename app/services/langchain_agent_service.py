# app/services/langchain_agent_service.py
"""
LangChain-based agent, using the SAME mcp_client_service
discovery and routing as agent_service.py, adapted into
StructuredTool objects that LangChain's create_agent expects.

This exists alongside the hand-built agent_service.py
deliberately, for direct comparison, not as a replacement yet.
Full migration, retiring the hand-built loop and tool_service.py
in favor of this path everywhere, is planned for a future project.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from pydantic import create_model
from app.services.mcp_client_service import mcp_client_service
from app.config import ai_config
from langgraph.errors import GraphRecursionError

def _json_schema_to_pydantic(tool_name: str, json_schema: dict):
    """
    Converts a JSON schema (from MCP's list_tools()) into a
    Pydantic model, so LangChain's StructuredTool knows the
    real parameter names and types, instead of an opaque
    **kwargs signature it cannot introspect.
    """
    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))

    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }

    fields = {}
    for field_name, field_schema in properties.items():
        field_type = type_map.get(field_schema.get("type", "string"), str)
        default = ... if field_name in required else None
        fields[field_name] = (field_type, default)

    return create_model(f"{tool_name}Args", **fields)


async def _build_langchain_tools():
    tool_definitions = await mcp_client_service.get_tool_definitions()
    langchain_tools = []

    for definition in tool_definitions:
        func_info = definition["function"]
        tool_name = func_info["name"]
        tool_description = func_info["description"]
        parameters_schema = func_info["parameters"]

        args_model = _json_schema_to_pydantic(tool_name, parameters_schema)

        def make_tool_function(name):
            async def tool_function(**kwargs):
                result = await mcp_client_service.call_tool(name, kwargs)
                return str(result)
            return tool_function

        wrapped = StructuredTool.from_function(
            coroutine=make_tool_function(tool_name),
            name=tool_name,
            description=tool_description,
            args_schema=args_model
        )
        langchain_tools.append(wrapped)

    return langchain_tools


class LangChainAgentService:

    def __init__(self):
        self._agent = None

    def _to_langchain_model_string(self, litellm_model: str) -> str:
        provider, _, model_name = litellm_model.partition("/")
        return f"{provider}:{model_name}"

    async def _get_agent(self):
        if self._agent is None:
            langchain_tools = await _build_langchain_tools()
            model = init_chat_model(self._to_langchain_model_string(ai_config.MODEL))
            self._agent = create_agent(
                model=model,
                tools=langchain_tools,
                system_prompt="You are a reasoning agent that solves problems step by step using the available tools."
            )
        return self._agent

    async def run_agent(self, goal: str, recursion_limit: int = 25) -> dict:
        try:
            agent = await self._get_agent()
            result = await agent.ainvoke({"messages": [{"role": "user", "content": goal}]}, 
                config={"recursion_limit": recursion_limit})

            messages = result["messages"]
            final_message = messages[-1]

            return {
                "answer": final_message.content,
                "message_count": len(messages),
                "raw_messages": [
                    {
                        "role": m.type,
                        "content": m.content,
                        "reasoning": m.additional_kwargs.get("reasoning_content")
                    }
                    for m in messages
                ]
            }
        except GraphRecursionError as e:
            # GraphRecursionError specifically, if the limit is hit
            return {
                "answer": f"Agent stopped without reaching a final answer: {str(e)}",
                "message_count": 0,
                "raw_messages": []
            }
        except Exception as e:
            return {
                "answer": f"Exception occurred during processing: {str(e)}",
                "message_count": 0,
                "raw_messages": []
            }


langchain_agent_service = LangChainAgentService()