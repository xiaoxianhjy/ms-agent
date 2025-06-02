from omegaconf import DictConfig

from modelscope_agent.llm.utils import Tool
from modelscope_agent.tools.base import ToolBase


class LoopTool(ToolBase):

    def __init__(self, config: DictConfig):
        super().__init__(config)

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def get_tools(self):
        return {
            'split_complex_task': [Tool(
                tool_name='split_to_sub_task',
                server_name='split_complex_task',
                description='Split complex task into sub tasks, for example, split a website generation task into sub tasks, '
                               'you plan the framework, include code files and classes and functions, and give the detail '
                               'information to the system and query field of the subtask, then '
                               'let each subtask to write a single file',
                parameters={
                    'system': 'str type, The system prompt to use in this sub task.',
                    'query': 'str type, The specific query in this sub task.'
                }
        )]
        }


    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        system = tool_args['system']
        query = tool_args['query']
        config = DictConfig(self.config)
        config.prompt.system = system
        from modelscope_agent.engine import SimpleEngine
        engine = SimpleEngine(config=config)
        return await engine.run(query)
