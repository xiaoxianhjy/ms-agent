import inspect
from typing import Any, List

from omegaconf import DictConfig

from modelscope_agent.utils.utils import assert_package
from .llm import LLM
from .utils import Message


class OpenAI(LLM):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        assert_package('openai')
        import openai
        self.client = openai.OpenAI(
            api_key=config.llm.openai_api_key,
            base_url=config.llm.openai_api_base_url,
        )

    def generate(self, model, messages, tools=None, **kwargs) -> Any:
        parameters = inspect.signature(self.client.chat.completions.create).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        return completion

    def format_message(self, messages: List[Message]):
        openai_messages = []
        for message in messages:
            message = message.to_dict()


