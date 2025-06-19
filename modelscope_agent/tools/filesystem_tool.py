import os

from omegaconf import OmegaConf

from modelscope_agent.llm.utils import Tool
from modelscope_agent.tools.base import ToolBase
from modelscope_agent.tools.mcp_client import MCPClient


class FileSystemTool(ToolBase):

    def __init__(self, config):
        super(FileSystemTool, self).__init__(config)
        self.prefix = 'output'
        # base_path = os.path.dirname(__file__)
        # mcp_file = os.path.join(base_path, 'filesystem.yaml')
        # self.mcp_client = MCPClient(OmegaConf.load(mcp_file))


    async def connect(self):
        # await self.mcp_client.connect()
        pass

    async def get_tools(self):
        return {
            'split_task': [Tool(
                tool_name='read_file',
                server_name='file_system',
                description='Read the content of a file',
                parameters= {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file",
                            }
                        },
                        "required": [
                            "path"
                        ],
                        "additionalProperties": False
                    }
        )]
        }

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        return await self.read_file(tool_args['path'])

    async def create_directory(self, path: str):
        os.makedirs(path, exist_ok=True)
        return '<OK>'
        # return await self.call_tool('filesystem', tool_name='create_directory', tool_args={'path': path})

    async def write_file(self, path: str, content: str):
        with open(path, 'w') as f:
            f.write(content)
        return '<OK>'
        # return await self.call_tool('filesystem', tool_name='write_file', tool_args={'path': path, 'content': content})

    async def read_file(self, path: str):
        try:
            if not path.startswith(self.prefix):
                path = os.path.join(self.prefix, path)
            with open(path, 'r') as f:
                return f.read()
        except Exception:
            return 'Code file not found or error, need regenerate.'
        # return await self.call_tool('filesystem', tool_name='read_file', tool_args={'path': path})

    async def list_files(self, path: str = None):
        file_paths = []
        if not path:
            path = self.prefix
        for root, dirs, files in os.walk(path):
            for file in files:
                absolute_path = os.path.join(root, file)
                relative_path = os.path.relpath(absolute_path, path)
                file_paths.append(relative_path)
        return file_paths
