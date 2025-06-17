from abc import abstractmethod
from typing import Optional, Dict

from omegaconf import DictConfig

from modelscope_agent.config import Config


class Engine:

    def __init__(self,
                 task_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None):
        if task_dir_or_id is None:
            self.config = config
        else:
            self.config = Config.from_task(task_dir_or_id, env)

    @abstractmethod
    async def run(self, inputs, **kwargs):
        pass