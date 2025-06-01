from abc import abstractmethod
from typing import List

from modelscope_agent.llm.utils import Message


class Memory:

    @abstractmethod
    def refine(self, messages: List[Message]) -> List[Message]:
        pass