from typing import List

from pydantic import ConfigDict

from modelscope_agent.callbacks import RunStatus
from modelscope_agent.engine.plan.base import Planer
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message


class ObserverPlaner(Planer):

    def __init__(self, config: ConfigDict):
        super().__init__(config)
        observer_config = self.config.planer.observer
        self.observer = LLM.from_config(observer_config)

    def generate_plan(self, llm: LLM, messages: List[Message]):
        pass

    def update_plan(self, llm: LLM, messages: List[Message], run_status: RunStatus):
        pass