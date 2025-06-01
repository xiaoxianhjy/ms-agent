import concurrent
from typing import List, Tuple, Dict, Any

from modelscope_agent.tools.loop_tool import LoopTool
from modelscope_agent.tools.mcp_client import MCPClient


class ToolManager:

    def __init__(self, config):
        self.config = config
        self.mcp_client = MCPClient(config)
        self.loop_tool = LoopTool(config)
        self.extra_tools = []
        self._tool_index = {}
        self.reindex_tool()

    def connect(self):
        self.mcp_client.connect()
        self.loop_tool.connect()
        for tool in self.extra_tools:
            tool.connect()

    def cleanup(self):
        self.mcp_client.cleanup()
        self.loop_tool.cleanup()
        for tool in self.extra_tools:
            tool.cleanup()

    def reindex_tool(self):
        tools = await self.mcp_client.get_tools()
        for tool in tools:
            assert tool.name not in self._tool_index, f'Tool name duplicated {tool.name}'
            self._tool_index[tool.name] = self.mcp_client
        for tool in self.loop_tool.get_tools():
            assert tool.name not in self._tool_index, f'Tool name duplicated {tool.name}'
            self._tool_index[tool.name] = self.loop_tool
        for extra_tool in self.extra_tools:
            for tool in extra_tool.get_tools():
                assert tool.name not in self._tool_index, f'Tool name duplicated {tool.name}'
                self._tool_index[tool.name] = tool

    def get_tools(self):
        return self._tool_index.values()

    def single_call_tool(self, tool_info: Dict[str, Any]):
        tool_name = tool_info['tool_name']
        tool_args = tool_info['tool_args']
        assert tool_name in self._tool_index, 'Tool name not found'
        tool = self._tool_index[tool_name]
        tool.call_tool(tool_name, tool_args)

    def parallel_call_tool(self, tool_list: List[Tuple[str, Dict[str, Any]]]):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_list))
        futures = executor.map(self.single_call_tool, tool_list)
        results = [future.result() for future in futures]
        return results
