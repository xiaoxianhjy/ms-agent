from abc import abstractmethod
from typing import List

from pydantic import ConfigDict

from modelscope_agent.callbacks import RunStatus
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message


class Planer:

    def __init__(self, config: ConfigDict):
        self.config = config

    @abstractmethod
    def generate_plan(self, llm: LLM, messages: List[Message]):
        pass

    @abstractmethod
    def update_plan(self, llm: LLM, messages: List[Message], run_status: RunStatus):
        pass

