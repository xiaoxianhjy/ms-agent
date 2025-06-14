from abc import abstractmethod
from typing import List

from pydantic import ConfigDict

from modelscope_agent.callbacks import Runtime
from modelscope_agent.llm.utils import Message


class Planer:

    def __init__(self, config: ConfigDict):
        self.config = config

    @abstractmethod
    def generate_plan(self, messages: List[Message], runtime: Runtime):
        pass

    @abstractmethod
    def update_plan(self, messages: List[Message], runtime: Runtime):
        pass

