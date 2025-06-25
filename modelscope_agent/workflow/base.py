# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import abstractmethod
from typing import List, Optional, Type

import json
from modelscope_agent.agent import Agent, CodeAgent, LLMAgent
from modelscope_agent.config import Config
from modelscope_agent.llm import Message
from modelscope_agent.utils.utils import str_to_md5
from omegaconf import DictConfig, OmegaConf

from modelscope.hub.utils.utils import get_cache_dir


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
