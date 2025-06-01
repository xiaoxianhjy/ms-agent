import inspect
from typing import Any


class DeepSeek:

    def __init__(self, system):
        self.system = system
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.token,
            base_url=self.base_url,
        )

    def generate(self, messages, model, tools=None, **kwargs) -> Any:
        parameters = inspect.signature(self.client.chat.completions.create).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        return completion

