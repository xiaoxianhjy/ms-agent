# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
from typing import Optional, Dict, Union, List

from omegaconf import DictConfig

from modelscope_agent.config import Config
from modelscope_agent.llm import Message


class Agent:
    """The base Agent class.

    Args:
        config_dir_or_id (`Optional[str]`): The directory or id of the config file.
        config (`Optional[DictConfig]`): The configuration object.
        env (`Optional[Dict[str, str]]`): The extra environment variables.
    """

    DEFAULT_TAG = 'Agent-default'

    def __init__(self,
                 config_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None):
        if config_dir_or_id is None:
            self.config: DictConfig = config
        else:
            self.config: DictConfig = Config.from_task(config_dir_or_id, env)

    @abstractmethod
    async def run(self, inputs: Union[str, List[Message]], **kwargs) -> List[Message]:
        """Run the agent.

        Args:
            inputs(`Union[str, List[Message]]`): The inputs can be a prompt string, or a list of messages from the previous agent
        Returns:
            The final messages
        """
        pass