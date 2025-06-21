# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope_agent.llm.utils import Tool
from modelscope_agent.tools.base import ToolBase
from modelscope_agent.utils import get_logger

logger = get_logger()


class FileSystemTool(ToolBase):
    """A file system operation tool

    TODO: This tool now is a simple implementation, sandbox or mcp TBD.
    """

    def __init__(self, config):
        super(FileSystemTool, self).__init__(config)
        file_system_config = getattr(config.tools, 'file_system', None)
        if file_system_config is not None:
            self._exclude_functions = getattr(file_system_config, 'exclude', [])
        else:
            self._exclude_functions = []
        self.output_dir = getattr(config, 'output_dir', 'output')

    async def connect(self):
        logger.warning_once(
            '[IMPORTANT]FileSystemTool is not implemented with sandbox, please consider other similar '
            'tools if you want to run dangerous code.')

    async def get_tools(self):
        tools = {
            'file_system': [
                Tool(
                    tool_name='create_directory',
                    server_name='file_system',
                    description='Create a directory',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type':
                                'string',
                                'description':
                                'The relative path of the directory to create',
                            }
                        },
                        'required': ['path'],
                        'additionalProperties': False
                    }),
                Tool(
                    tool_name='write_file',
                    server_name='file_system',
                    description='Write content into a file',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type': 'string',
                                'description': 'The relative path of the file',
                            }
                        },
                        'required': ['path'],
                        'additionalProperties': False
                    }),
                Tool(
                    tool_name='read_file',
                    server_name='file_system',
                    description='Read the content of a file',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type': 'string',
                                'description': 'The relative path of the file',
                            }
                        },
                        'required': ['path'],
                        'additionalProperties': False
                    }),
                Tool(
                    tool_name='list_files',
                    server_name='file_system',
                    description='List all files in a directory',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type':
                                'string',
                                'description':
                                "The path to list files, if path is None or '' or not given, "
                                'the root dir will be used as path.',
                            }
                        },
                        'required': [],
                        'additionalProperties': False
                    }),
            ]
        }
        return {
            'file_system': [
                t for t in tools['file_system']
                if t['tool_name'] not in self._exclude_functions
            ]
        }

    async def call_tool(self, server_name: str, *, tool_name: str,
                        tool_args: dict) -> str:
        return await self.read_file(tool_args['path'])

    async def create_directory(self, path: str) -> str:
        """Create a directory

        Args:
            path(`str`): The relative directory path to create, a prefix dir will be automatically concatenated.

        Returns:
            <OK> or error message.
        """
        try:
            os.makedirs(os.path.join(self.output_dir, path), exist_ok=True)
            return '<OK>'
        except Exception as e:
            return 'Create directory failed, error: ' + str(e)

    async def write_file(self, path: str, content: str):
        """Write content to a file.

        Args:
            path(`path`): The relative file path to write into, a prefix dir will be automatically concatenated.
            content:

        Returns:
            <OK> or error message.
        """
        try:
            with open(os.path.join(self.output_dir, path), 'w') as f:
                f.write(content)
            return '<OK>'
        except Exception as e:
            return 'Write file failed, error: ' + str(e)

    async def read_file(self, path: str):
        """Read the content of a file.

        Args:
            path(`path`): The relative file path to read, a prefix dir will be automatically concatenated.

        Returns:
            The file content or error message.
        """
        try:
            with open(os.path.join(self.output_dir, path), 'r') as f:
                return f.read()
        except Exception as e:
            return 'Read file failed, error: ' + str(e)

    async def list_files(self, path: str = None):
        """List all files in a directory.

        Args:
            path: The relative path to traverse, a prefix dir will be automatically concatenated.

        Returns:
            The file names concatenated as a string
        """
        file_paths = []
        if not path:
            path = self.output_dir
        else:
            path = os.path.join(self.output_dir, path)
        for root, dirs, files in os.walk(path):
            for file in files:
                if '.' in file or '..' in file:
                    continue
                absolute_path = os.path.join(root, file)
                relative_path = os.path.relpath(absolute_path, path)
                file_paths.append(relative_path)
        return '\n'.join(file_paths)
