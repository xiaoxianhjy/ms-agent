from abc import abstractmethod
from typing import Optional

from modelscope_agent.agent import SimpleEngine
from modelscope_agent.agent.base import Engine
from modelscope_agent.agent.code_engine import CodeEngine


class Workflow:

    def find_engine(self, type: str) -> Optional[Engine]:
        if type == 'SimpleEngine':
            return SimpleEngine
        elif type == 'CodeEngine':
            return CodeEngine
        return None

    @abstractmethod
    async def run(self, inputs, **kwargs):
        pass