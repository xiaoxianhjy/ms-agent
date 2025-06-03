import asyncio
import json
from typing import List, Tuple, Dict, Any

from modelscope_agent.tools.base import ToolBase
from modelscope_agent.tools.loop_tool import LoopTool
from modelscope_agent.tools.mcp_client import MCPClient


class ToolManager:

    def __init__(self, config):
        self.config = config
        self.mcp_client = MCPClient(config)
        self.loop_tool = LoopTool(config)
        self.extra_tools: List[ToolBase] = []
        self._tool_index = {}

    async def connect(self):
        await self.mcp_client.connect()
        await self.loop_tool.connect()
        for tool in self.extra_tools:
            await tool.connect()
        await self.reindex_tool()

    async def cleanup(self):
        await self.mcp_client.cleanup()
        await self.loop_tool.cleanup()
        for tool in self.extra_tools:
            await tool.cleanup()

    async def reindex_tool(self):

        def extend_tool(tool_ins: ToolBase, server_name: str, tool_list: List):
            for tool in tool_list:
                assert tool['tool_name'] not in self._tool_index, f'Tool name duplicated {tool["tool_name"]}'
                self._tool_index[tool['tool_name']] = (tool_ins, server_name, tool)

        mcps = await self.mcp_client.get_tools()
        for server_name, tool_list in mcps.items():
            extend_tool(self.mcp_client, server_name, tool_list)

        loop_tools = await self.loop_tool.get_tools()
        for server_name, tool_list in loop_tools.items():
            extend_tool(self.loop_tool, server_name, tool_list)
        for extra_tool in self.extra_tools:
            tools = await extra_tool.get_tools()
            for server_name, tool_list in tools.items():
                extend_tool(extra_tool, server_name, tool_list)

    async def get_tools(self):
        return [value[2] for value in self._tool_index.values()]

    async def single_call_tool(self, tool_info: Dict[str, Any]):
        tool_name = tool_info['tool_name']
        if '---' in tool_name:
            tool_name = tool_name.split('---')[1]
        tool_args = tool_info['arguments']
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        assert tool_name in self._tool_index, 'Tool name not found'
        tool_ins, server_name, _ = self._tool_index[tool_name]
        return await tool_ins.call_tool(server_name, tool_name, tool_args)

    async def parallel_call_tool(self, tool_list: List[Tuple[str, Dict[str, Any]]]):
        tasks = [self.single_call_tool(tool) for tool in tool_list]
        result = await asyncio.gather(*tasks)
        return result
