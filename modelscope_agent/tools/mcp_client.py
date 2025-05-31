from contextlib import AsyncExitStack
from typing import Any, Dict, Literal, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from modelscope_agent.config.env import Env
from modelscope_agent.tools.base import Tool
from modelscope_agent.utils import get_logger

logger = get_logger()

EncodingErrorHandler = Literal['strict', 'ignore', 'replace']

DEFAULT_ENCODING = 'utf-8'
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = 'strict'

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5


class MCPClient(Tool):

    def __init__(self, config, mcp_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.mcp_config = mcp_config

    async def call_tool(self, server_name: str, tool_name: str,
                        tool_args: dict):
        response = await self.sessions[server_name].call_tool(
            tool_name, tool_args)
        texts = []
        if response.isError:
            return f'execute error: {'\n\n'.join(response.content)}'
        for content in response.content:
            if content.type == 'text':
                texts.append(content.text)
        return '\n\n'.join(texts)

    async def get_tools(self) -> Dict:
        tools = {}
        for key, session in self.sessions.items():
            tools[key] = []
            response = await session.list_tools()
            tools[key].extend(response.tools)
        return tools

    @staticmethod
    def print_tools(server_name: str, tools: Dict):
        if len(tools) > 10:
            tools = [tool.name for tool in tools][:10]
            logger.info(f'\nConnected to server "{server_name}" '
                        f'with tools: \n{"\n".join(tools)}\nOnly list first 10 of them.')
        else:
            tools = [tool.name for tool in tools]
            logger.info(f'\nConnected to server "{server_name}" '
                        f'with tools: \n{"\n".join(tools)}.')

    async def connect_to_server(self, server_name: str, **kwargs):
        logger.info(f'connect to {server_name}')
        command = kwargs.get('command')
        url = kwargs.get('url')
        session_kwargs = kwargs.get('session_kwargs')
        if url:
            # transport: 'sse'
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url, kwargs.get('headers'),
                    kwargs.get('timeout', DEFAULT_HTTP_TIMEOUT),
                    kwargs.get('sse_read_timeout', DEFAULT_SSE_READ_TIMEOUT)))
            read, write = sse_transport
            session_kwargs = session_kwargs or {}
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write, **session_kwargs))

            await session.initialize()
            # Store session
            self.sessions[server_name] = session
            # List available tools
            self.print_tools(server_name, await session.list_tools())
        elif command:
            # transport: 'stdio'
            args = kwargs.get('args')
            if not args:
                raise ValueError(
                    "'args' parameter is required for stdio connection")
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=kwargs.get('env'),
                encoding=kwargs.get('encoding', DEFAULT_ENCODING),
                encoding_error_handler=kwargs.get(
                    'encoding_error_handler', DEFAULT_ENCODING_ERROR_HANDLER),
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params))

            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write))

            await session.initialize()

            # Store session
            self.sessions[server_name] = session
            self.print_tools(server_name, await session.list_tools())
        else:
            raise ValueError(
                "'url' or 'command' parameter is required for connection")
        return server_name

    async def connect(self):
        assert self.mcp_config, 'MCP config is required'
        envs = Env.load_env()
        for name, server in self.mcp_config.items():
            cmd = server['cmd']
            env_dict = cmd.pop('env', {})
            env_dict = {
                key: value if value else envs.get(key, '')
                for key, value in env_dict.items()
            }
            await self.connect_to_server(server_name=name, env=env_dict, **cmd)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
