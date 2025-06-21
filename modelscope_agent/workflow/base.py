# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from abc import abstractmethod
from typing import Optional, Type, List

from omegaconf import OmegaConf, DictConfig

from modelscope_agent.agent import Agent, CodeAgent, LLMAgent
from modelscope_agent.llm import Message
from modelscope.hub.utils.utils import get_cache_dir

from modelscope_agent.utils.utils import str_to_md5


class Workflow:

    cache_dir = os.path.join(get_cache_dir(), 'workflow_cache')
    os.makedirs(cache_dir, exist_ok=True)

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

    @staticmethod
    def _save_history(query: str, task: str, config: DictConfig, messages: List[Message]):
        folder = str_to_md5(query)
        os.makedirs(os.path.join(Workflow.cache_dir, folder), exist_ok=True)
        config_file = os.path.join(Workflow.cache_dir, folder, f'{task}.yaml')
        message_file = os.path.join(Workflow.cache_dir, folder, f'{task}.json')
        with open(config_file, 'w') as f:
            OmegaConf.save(config, f)
        with open(message_file, 'w') as f:
            json.dump([message.to_dict() for message in messages], f)

    @staticmethod
    def _read_history(query: str, task: str):
        folder = str_to_md5(query)
        config_file = os.path.join(Workflow.cache_dir, folder, f'{task}.yaml')
        message_file = os.path.join(Workflow.cache_dir, folder, f'{task}.json')
        config = None
        messages = None
        if os.path.exists(config_file):
            config = OmegaConf.load(config_file)
        if os.path.exists(f'{task}.json'):
            with open(message_file, 'r') as f:
                messages = json.load(f)
                messages = [Message(**message) for message in messages]
        return config, messages
