# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import inspect
import os
import sys
from typing import Optional, Dict

from omegaconf import DictConfig

from . import Agent
from .code import Code


class CodeAgent(Agent):

    def __init__(self,
                 config_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None,
                 *,
                 code_file: str):
        super().__init__(config_dir_or_id, config, env)
        self.code_file = code_file

    async def run(self, inputs, **kwargs):
        """Run the agent.

        Args:
            inputs(`Union[str, List[Message]]`): The inputs can be a prompt string, or a list of messages from the previous agent
        Returns:
            The final messages
        """
        base_path = os.path.dirname(self.code_file)
        if sys.path[0] != base_path:
            sys.path.insert(0, base_path)
        code_module = importlib.import_module(self.code_file)
        module_classes = {name: cls for name, cls in inspect.getmembers(code_module, inspect.isclass)}
        for name, cls in module_classes.items():
            if cls.__bases__[0] is Code and cls.__module__ == self.code_file:
                instance = cls(self.config)
                messages = await instance.run(inputs, **kwargs)
                return messages
        return inputs
