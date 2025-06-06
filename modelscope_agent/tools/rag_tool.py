import os

from omegaconf import OmegaConf

from modelscope_agent.tools.base import ToolBase
from modelscope_agent.tools.mcp_client import MCPClient


class RagTool(ToolBase):

    def __init__(self, config):
        super(RagTool, self).__init__(config)
        base_path = os.path.dirname(__file__)
        mcp_file = os.path.join(base_path, 'rag.yaml')
        self.mcp_client = MCPClient(OmegaConf.load(mcp_file))


    async def connect(self):
        await self.mcp_client.connect()

    async def get_tools(self):
        return await self.mcp_client.get_tools()

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        return await self.mcp_client.call_tool(server_name, tool_name, tool_args)
