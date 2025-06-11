from typing import List

from omegaconf import DictConfig

from .run_status import RunStatus
from ..llm.utils import Message


class Callback:

    def __init__(self, config: DictConfig):
        self.config = config

    def on_task_begin(self, run_status: RunStatus, messages: List[Message]):
        pass

    def on_generate_response(self, run_status: RunStatus, messages: List[Message]):
        pass

    def after_generate_response(self, run_status: RunStatus, messages: List[Message]):
        pass

    def on_tool_call(self, run_status: RunStatus, messages: List[Message]):
        pass

    def after_tool_call(self, run_status: RunStatus, messages: List[Message]):
        pass

    def on_task_end(self, run_status: RunStatus, messages: List[Message]):
        pass