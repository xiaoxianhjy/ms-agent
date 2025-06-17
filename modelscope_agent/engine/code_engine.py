import importlib
import inspect
import os
import sys
from typing import Optional, Dict

from omegaconf import DictConfig

from modelscope_agent.engine.base import Engine
from modelscope_agent.engine.code.base import Code


class CodeEngine(Engine):

    def __init__(self,
                 task_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None,
                 *,
                 code_file: str,
                 **kwargs):
        super().__init__(task_dir_or_id, config, env)
        self.code_file = code_file

    async def run(self, inputs, **kwargs):
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
