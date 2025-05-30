from typing import List

from omegaconf import DictConfig

from .run_status import RunStatus
from ..llm.utils import Message


class Callback:

    def on_task_begin(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        config.agents

    def on_generate_response(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        pass

    def after_generate_response(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        pass

    def on_tool_call(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        pass

    def after_tool_call(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        pass

    def on_task_end(self, config: DictConfig, run_status: RunStatus, messages: List[Message]):
        pass