from abc import abstractmethod
from typing import Any, Dict, List

from omegaconf import DictConfig

from modelscope_agent.config.config import Config
from modelscope_agent.llm.utils import Message


class LLM:

    def __init__(self, config: DictConfig):
        """Initialize the model.

        Args:
            config: A omegaconf.DictConfig object.
        """
        self.config = config

    @abstractmethod
    def generate(self, model, messages, tools=None, **kwargs) -> Any:
        """Generate response by the given messages.

        Args:
            messages: The previous messages.
            tools: The tools to use.
            **kwargs: Extra generation arguments.

        Returns:
            The response.
        """
        pass

    @classmethod
    def from_task(cls, task_dir_or_id: str, *, env: Dict[str, str] = None) -> Any:
        """Instantiate an LLM instance.

        Args:
            task_dir_or_id: The local task directory or an id in the modelscope repository.
            env: The extra environment variables except ones already been included
                in the environment or in the `.env` file.

        Returns:
            The LLM instance.
        """
        config = Config.from_task(task_dir_or_id, env)
        return cls.from_config(config)


    @classmethod
    def from_config(cls, config: DictConfig) -> Any:
        """Instantiate an LLM instance.

        Args:
            config: The omegaconf.DictConfig object.

        Returns:
            The LLM instance.
        """
        from .model_mapping import all_services_mapping
        assert config.llm.service in all_services_mapping
        return all_services_mapping[config.llm.service](config)
