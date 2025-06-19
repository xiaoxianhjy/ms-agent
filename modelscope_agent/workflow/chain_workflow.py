from abc import abstractmethod
from typing import Optional, Dict

from omegaconf import DictConfig

from modelscope_agent.config import Config
from modelscope_agent.agent.base import Engine
from modelscope_agent.workflow.base import Workflow


class ChainWorkflow(Workflow):

    def __init__(self,
                 config_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None,
                 **kwargs):
        if config_dir_or_id is None:
            self.config = config
        else:
            self.config = Config.from_task(config_dir_or_id, env)
        self.workflow_chains = []
        self.build_workflow()

    def build_workflow(self):
        if not self.config:
            return []

        has_next = set()
        start_task = None
        for task_name, task_config in self.config.items():
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    has_next.add(next_tasks)
                else:
                    assert len(next_tasks) == 1, 'ChainWorkflow only supports one next task'
                    has_next.update(next_tasks)

        for task_name in self.config.keys():
            if task_name not in has_next:
                start_task = task_name
                break

        if start_task is None:
            raise ValueError("No start task found")

        result = []
        current_task = start_task

        while current_task:
            result.append(current_task)
            next_task = None
            task_config = self.config[current_task]
            if 'next' in task_config:
                next_tasks = task_config['next']
                if isinstance(next_tasks, str):
                    next_task = next_tasks
                else:
                    next_task = next_tasks[0]

            current_task = next_task
        self.workflow_chains = result

    @abstractmethod
    async def run(self, inputs, **kwargs):
        config = None
        for task in self.workflow_chains:
            task_info = getattr(self.config, task)
            engine_cls: Engine = self.find_engine(task_info.engine.name)
            _cfg = getattr(task_info, 'config', config)
            init_args = getattr(task_info.engine, 'kwargs', {})
            if isinstance(_cfg, str):
                engine = engine_cls(config_dir_or_id=_cfg, **init_args)
            else:
                engine = engine_cls(config=_cfg, **init_args)
            inputs = await engine.run(inputs, **kwargs)
            config = engine.config
        return inputs

