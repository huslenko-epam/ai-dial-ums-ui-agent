import json
import logging
from collections import defaultdict
from typing import Any, AsyncGenerator

from openai import AsyncAzureOpenAI

from agent.clients.stdio_mcp_client import StdioMCPClient
from agent.models.message import Message, Role
from agent.clients.http_mcp_client import HttpMCPClient

logger = logging.getLogger(__name__)


class DialClient:
    """Handles AI model interactions and integrates with MCP client"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        tools: list[dict[str, Any]],
        tool_name_client_map: dict[str, HttpMCPClient | StdioMCPClient],
    ):
        # TODO:
        # 1. set tools, tool_name_client_map and model
        # 2. Create AsyncAzureOpenAI as `async_openai` with:
        #   - api_key=api_key
        #   - azure_endpoint=endpoint
        #   - api_version=""

        self.tools = tools
        self.tool_name_client_map = tool_name_client_map
        self.model = model
        self.async_openai = AsyncAzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=""
        )

    async def response(self, messages: list[Message]) -> Message:
        """Non-streaming completion with tool calling support"""
        # TODO:
        # 1. Create chat completions request (self.async_openai.chat.completions.create) and get it as `response`
        # 2. Create message `ai_message`
        # 3. Check if message contains tool_calls, if yes, then add them as tool_calls
        # 4. If `ai_message` contains tool calls then:
        #       - add `ai_message` to messages
        #       - call `_call_tools(ai_message, messages)`
        #       - make recursive call with messages to process further
        # 5. return ai_message

        response = await self.async_openai.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=self.tools,
            temperature=0.0,
            stream=False,
        )

        message = Message(
            role=Role.ASSISTANT,
            content=response.choices[0].message.content,
        )

        if tool_calls := response.choices[0].message.tool_calls:
            message.tool_calls = tool_calls

        if message.tool_calls:
            messages.append(message)
            await self._call_tools(message, messages)
            return await self.response(messages)

        return message

    async def stream_response(
        self, messages: list[Message]
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion with tool calling support.
        Yields SSE-formatted chunks.
        """
        # TODO:
        # 1. Create chat completions request (self.async_openai.chat.completions.create) and get it as `stream`
        # 2. Create empty sting and assign it to `content_buffer` variable (we will collect content while streaming)
        # 3. Create empty array with `tool_deltas` variable name
        # 4. Make async loop through `stream` (async for chunk in stream):
        #       - get delta chunk
        #       - if delta contains content
        #           - create dict:{"choices": [{"delta": {"content": delta.content}, "index": 0, "finish_reason": None}]} as `chunk_data`
        #           - `yield f"data: {json.dumps(chunk_data)}\n\n"`
        #           - concat `content_buffer` with delta content
        #       - if delta has tool calls then extend `tool_deltas` with `delta.tool_calls`
        # 5. If `tool_deltas` are present:
        #       - collect tool calls with `_collect_tool_calls` method and assign to the `tool_calls` variable
        #       - create assistant message with collected content and tool calls
        #       - add created assistant message to `messages`
        #       - call `_call_tools(ai_message, messages)` (its async, don't forget about await)
        #       - make recursive call with messages to process further:
        #           `async for chunk in self.stream_response(messages):
        #               yield chunk
        #            return`
        # 6. Add assistant message with collected content
        # 7. Create final chunk dict: {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
        # 8. yield f"data: {json.dumps(final_chunk)}\n\n"
        # 9. yield "data: [DONE]\n\n"

        stream = await self.async_openai.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=self.tools,
            temperature=0.0,
            stream=True,
        )

        content_buffer = ""
        tool_deltas = []

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                chunk_data = {
                    "choices": [
                        {
                            "delta": {"content": delta.content},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                content_buffer += delta.content

            if delta.tool_calls:
                tool_deltas.extend(delta.tool_calls)

        if tool_deltas:
            tool_calls = self._collect_tool_calls(tool_deltas)

            message = Message(
                role=Role.ASSISTANT, content=content_buffer, tool_calls=tool_calls
            )

            messages.append(message)
            await self._call_tools(message, messages)

            async for chunk in self.stream_response(messages):
                yield chunk
            return

        messages.append(Message(role=Role.ASSISTANT, content=content_buffer))

        final_chunk = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def _collect_tool_calls(self, tool_deltas):
        """Convert streaming tool call deltas to complete tool calls"""
        # TODO:
        # 1. Create `tool_dict` with `defaultdict(lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None})`
        # 2. Iterate through tool_deltas and:
        #       - get delta index
        #       - if delta has id then add it to `tool_dict[idx]["id"]`
        #       - if delta has name (function.name) the add to `tool_dict[idx]["function"]["name"]`
        #       - if delta has arguments (function.arguments) the add to `tool_dict[idx]["function"]["arguments"]`
        #       - if delta has type then add it to `tool_dict[idx]["type"]`
        # 3. Create list from `tool_dict` values and return it

        tool_dict = defaultdict(
            lambda: {
                "id": None,
                "function": {"arguments": "", "name": None},
                "type": None,
            }
        )

        for delta in tool_deltas:
            idx = delta.index
            if delta.id:
                tool_dict[idx]["id"] = delta.id
            if delta.function.name:
                tool_dict[idx]["function"]["name"] = delta.function.name
            if delta.function.arguments:
                tool_dict[idx]["function"]["arguments"] += delta.function.arguments
            if delta.type:
                tool_dict[idx]["type"] = delta.type

        collected_tools = list(tool_dict.values())
        return collected_tools

    async def _call_tools(
        self, ai_message: Message, messages: list[Message], silent: bool = False
    ):
        """Execute tool calls using MCP client"""
        # TODO:
        # Iterate through ai_message tool_calls:
        # 1. Get tool name from tool call (function.name)
        # 2. Load tool arguments from tool call (function.arguments) through `json.loads`
        # 3. Get MCP client from `tool_name_client_map` via tool name
        # 4. If no MCP Client found then create tool message with info in content that such tool is absent, add it to
        #    `messages`, and `continue`
        # 5. Make tool call with MCP client (its async!)
        # 6. Add tool message with content with tool execution result to `messages`

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            client = self.tool_name_client_map.get(tool_name)
            if not client:
                error_msg = f"Unable to call {tool_name}. MCP client not found."

                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if not silent:
                logger.info(
                    "Calling tool",
                    extra={"tool_name": tool_name, "tool_args": tool_args},
                )

            tool_result = await client.call_tool(tool_name, tool_args)

            messages.append(
                Message(
                    role=Role.TOOL,
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )
