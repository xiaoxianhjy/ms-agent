import asyncio

from omegaconf import DictConfig

from modelscope_agent.llm.utils import Tool
from modelscope_agent.tools.base import ToolBase


class SplitTask(ToolBase):

    def __init__(self, config: DictConfig):
        super().__init__(config)

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def get_tools(self):
        return {
            'split_task': [Tool(
                tool_name='split_to_sub_task',
                server_name='split_task',
                description='Split complex task into sub tasks and start them, for example, split a website generation task into sub tasks, '
                               'you plan the framework, include code files and classes and functions, and give the detail '
                               'information to the system and query field of the subtask, then '
                               'let each subtask to write a single file',
                parameters= {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "description": "Each element is a dict, which contains two fields: system and query to start the sub task."
                            }
                        },
                        "required": [
                            "tasks"
                        ],
                        "additionalProperties": False
                    }
        )]
        }


    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        from modelscope_agent.engine import SimpleEngine
        tasks = tool_args.get('tasks')
        sub_tasks = []
        for i, task in enumerate(tasks):
            system = task['system']
            query = task['query']
            config = DictConfig(self.config)
            config.prompt.system = system
            delattr(config.tools, 'split_task')
            trust_remote_code = getattr(config, 'trust_remote_code', False)
            engine = SimpleEngine(config=config, trust_remote_code=trust_remote_code)
            sub_tasks.append(engine.run(query, tag=f'workflow {i}'))
        result = await asyncio.gather(*sub_tasks)
        res = []
        for messages in result:
            res.append(messages[-1].content)
        return '\n\n'.join(res)
