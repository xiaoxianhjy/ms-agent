from omegaconf import DictConfig

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
            'split_complex_task': {
                'tool_name': 'split_to_sub_task',
                'tool_args': {
                    'system': 'The system prompt of this sub task',
                    'query': 'The query to solve of this sub task',
                }
            }
        }


    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        system = tool_args['system']
        query = tool_args['query']
        config = DictConfig(self.config)
        config.prompt.system = system
        from modelscope_agent.engine import SimpleEngine
        engine = SimpleEngine(config=config)
        return await engine.run(query)
