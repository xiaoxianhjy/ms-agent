import inspect
from typing import Any

from modelscope_agent.utils.llm_utils import retry
from modelscope_agent.llm.llm import LLM


class DashScope(LLM):

    def __init__(self, system):
        self.system = system
        self.client = OpenAI(
            api_key=self.token,
            base_url=self.base_url,
        )

    @retry(max_attempts=5)
    def generate(self, messages, model, tools=None, **kwargs) -> Any:
        _e = None
        parameters = inspect.signature(self.client.chat.completions.create).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=False,
            **kwargs
        )
        return completion

