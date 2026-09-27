# app/services/mcp_client_service.py
import json
from mcp import Client
from app.config import ai_config
from app.tools.knowledge_base_tool import search_knowledge_base, KNOWLEDGE_BASE_TOOL_DEFINITION

# Local tools: genuinely coupled to this app's database and RAG
# pipeline, never routed through MCP, called directly instead
LOCAL_TOOLS = {
    "knowledge_base": search_knowledge_base,
}
LOCAL_TOOL_DEFINITIONS = [
    KNOWLEDGE_BASE_TOOL_DEFINITION,
]


class MCPClientService:

    def __init__(self):
        self._tool_to_server: dict[str, str] = {}
        self._tool_definitions: list[dict] = []
        self._discovered = False

    async def discover_tools(self):
        """
        Builds one merged routing table and definitions list from
        two sources: remote MCP servers (discovered live via
        list_tools()) and local tools (hand-registered, tightly
        coupled to this app). The model sees one unified tool set,
        with no visibility into which source each came from.
        """
        self._tool_to_server = {}
        self._tool_definitions = []

        for server_url in ai_config.KNOWN_MCP_SERVERS:
            async with Client(server_url) as client:
                tools_result = await client.list_tools()
                for tool in tools_result.tools:
                    self._tool_to_server[tool.name] = server_url
                    self._tool_definitions.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    })

        for definition in LOCAL_TOOL_DEFINITIONS:
            name = definition["function"]["name"]
            self._tool_to_server[name] = "local"
            self._tool_definitions.append(definition)

        self._discovered = True

    async def get_tool_definitions(self) -> list[dict]:
        if not self._discovered:
            await self.discover_tools()
        return self._tool_definitions

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if not self._discovered:
            await self.discover_tools()

        destination = self._tool_to_server.get(tool_name)
        if not destination:
            return {"error": f"No known server or local tool named: {tool_name}"}

        if destination == "local":
            function = LOCAL_TOOLS[tool_name]
            return function(**arguments)

        async with Client(destination) as client:
            result = await client.call_tool(tool_name, arguments)

            if result.structured_content is not None:
                return result.structured_content

            # Fall back to parsing the text content block, since our
            # tools return plain dicts without a declared output schema,
            # and the SDK serializes those as text, not structured_content
            if result.content:
                text = result.content[0].text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw_text": text}

            return {"error": "Tool returned no content"}


mcp_client_service = MCPClientService()