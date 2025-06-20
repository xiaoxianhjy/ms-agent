# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
from typing import Optional, Type

from modelscope_agent.agent import Agent, CodeAgent, SimpleLLMAgent


class Workflow:

    @staticmethod
    def find_agent(type: str) -> Optional[Type[Agent]]:
        """Find an agent by name

        Args:
            type(`str`): The type of agent to find

        Returns:
            The Agent class
        """
        if type == 'SimpleEngine':
            return SimpleLLMAgent
        elif type == 'CodeEngine':
            return CodeAgent
        return None

    @abstractmethod
    async def run(self, inputs, **kwargs):
        pass
