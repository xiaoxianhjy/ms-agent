import os

from omegaconf import OmegaConf

from modelscope_agent.tools.base import ToolBase
from modelscope_agent.tools.mcp_client import MCPClient


class FileSystemTool(ToolBase):

    def __init__(self, config):
        super(FileSystemTool, self).__init__(config)
        base_path = os.path.dirname(__file__)
        mcp_file = os.path.join(base_path, 'filesystem.yaml')
        self.mcp_client = MCPClient(OmegaConf.load(mcp_file))


    async def connect(self):
        await self.mcp_client.connect()

    async def get_tools(self):
        return await self.mcp_client.get_tools()

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        return await self.mcp_client.call_tool(server_name, tool_name, tool_args)

    async def create_directory(self, path: str):
        return await self.call_tool('filesystem', tool_name='create_directory', tool_args={'path': path})

    async def write_file(self, path: str, content: str):
        return await self.call_tool('filesystem', tool_name='write_file', tool_args={'path': path, 'content': content})

    async def read_file(self, path: str):
        return await self.call_tool('filesystem', tool_name='read_file', tool_args={'path': path})
