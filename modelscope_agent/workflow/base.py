# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
from typing import Optional, Type

from modelscope_agent.agent import Agent, CodeAgent, LLMAgent


class Workflow:

    @staticmethod
    def find_agent(type: str) -> Optional[Type[Agent]]:
        """Find an agent by name

        Args:
            type(`str`): The type of agent to find

        Returns:
            The Agent class
        """
        if type == 'LLMAgent':
            return LLMAgent
        elif type == 'CodeAgent':
            return CodeAgent
        return None

    @abstractmethod
    async def run(self, inputs, **kwargs):
        pass
