import logging
from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


class HttpMCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        logger.debug(
            "HttpMCPClient instance created", extra={"server_url": mcp_server_url}
        )

    @classmethod
    async def create(cls, mcp_server_url: str) -> "HttpMCPClient":
        """Async factory method to create and connect MCPClient"""
        # TODO:
        # 1. Create instance `cls(mcp_server_url)`
        # 2. Connect to MCP Server (method `connect`)
        # 3. Return created instance

        mcp_instance = cls(mcp_server_url)
        await mcp_instance.connect()
        return mcp_instance

    async def connect(self):
        """Connect to MCP server"""
        # TODO:
        # 1. Set `self._streams_context` as `streamablehttp_client(self.server_url)`
        # 2. Create `read_stream, write_stream, _` variables from result if execution of `await self._streams_context.__aenter__()`
        # 3. Set `self._session_context` as `ClientSession(read_stream, write_stream)`
        # 4. Set `self.session: ClientSession` as `await self._session_context.__aenter__()`
        # 5. Call session initialization (initialize method) and assign results to `init_result` variable (initialize is async)
        # 6. Log the `init_result` to see in logs MCP server capabilities

        self._streams_context = streamable_http_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self._session_context.__aenter__()

        init_result = await self.session.initialize()
        logger.info(
            "MCP session initialized",
            extra={
                "server_url": self.server_url,
                "init_result": init_result.model_dump(),
            },
        )

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""
        # TODO:
        # 1. Check if session is present, if not then raise an error with message that MCP client is not connected to MCP server
        # 2. Through the session get list tools (it is async method, await it)
        # 3. Retrieved tools are returned according MCP (Anthropic) spec. You need to covert it to the DIAL (OpenAI compatible)
        #    tool format https://dialx.ai/dial_api#operation/sendChatCompletionRequest (see tools param)
        # 4. Log retrieved tools
        # 5. Return tools dicts list

        if not self.session:
            raise RuntimeError("MCP client not connected.")

        tools = await self.session.list_tools()

        tool_list = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools.tools
        ]

        logger.info(
            "Retrieved tools from MCP server",
            extra={
                "server_url": self.server_url,
                "tool_count": len(tool_list),
                "tool_names": [tool["function"]["name"] for tool in tool_list],
            },
        )

        return tool_list

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""
        # TODO:
        # 1. Check if session is present, if not then raise an error with message that MCP client is not connected to MCP server
        # 2. Log the call to MCP Server (tool name, tool args, url)
        # 3. Make tool call through session (it is async, don't forget to await)
        # 4. Get tool execution content
        # 5. Get first element from content (it is array with `ContentBlock`)
        # 6. Check if element is instance of TextContent, if yes then return its text, otherwise return retrieved content

        if not self.session:
            raise RuntimeError("MCP client not connected.")

        logger.info(
            "Calling MCP tool",
            extra={
                "server_url": self.server_url,
                "tool_name": tool_name,
                "tool_args": tool_args,
            },
        )

        tool_result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        content = tool_result.content


        if len(content) == 1 and isinstance(content[0], TextContent):
            return content[0].text

        return content
