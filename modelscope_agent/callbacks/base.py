from typing import List

from omegaconf import DictConfig

from modelscope_agent.engine.runtime import Runtime
from ..llm.utils import Message


class Callback:

    def __init__(self, config: DictConfig):
        self.config = config

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        pass

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        pass

    async def after_generate_response(self, runtime: Runtime, messages: List[Message]):
        pass

    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        pass

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        pass

    async def on_task_end(self, runtime: Runtime, messages: List[Message]):
        pass