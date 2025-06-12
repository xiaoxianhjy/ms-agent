import inspect
from typing import Any, List

from omegaconf import DictConfig

from modelscope_agent.llm.openai_llm import OpenAI
from modelscope_agent.llm.utils import Tool
from modelscope_agent.utils.llm_utils import retry


class Claude(OpenAI):

    def __init__(self, config: DictConfig):
        super().__init__(config, base_url=config.llm.claude_base_url, api_key=config.llm.claude_api_key)

    def format_tools(self, tools: List[Tool]):
        if tools:
            tools = [
                {
                    'name': f'{tool.get("server_name")}---{tool["tool_name"]}' if tool.get('server_name') else tool[
                        'tool_name'],
                    'description': tool['description'],
                    'input_schema': tool['parameters']
                } for tool in tools
            ]
        else:
            tools = None
        return tools