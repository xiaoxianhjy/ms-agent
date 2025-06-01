import os.path
from typing import Dict, Union, Any

from modelscope import snapshot_download
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.basecontainer import BaseContainer
from modelscope_agent.utils import get_logger
from .env import Env

logger = get_logger()


class Config:

    supported_config_names = ['config.json', 'config.yml', 'config.yaml']

    def from_task(self, task_dir_or_id: str, env: Dict[str, str] = None) -> Union[DictConfig, ListConfig]:
        """Read a task config file and return a config object.

        Args:
            task_dir_or_id: The local task directory or an id in the modelscope repository.
            env: The extra environment variables except ones already been included
                in the environment or in the `.env` file.

        Returns:
            The config object.
        """
        if not os.path.exists(task_dir_or_id):
            task_dir_or_id = snapshot_download(task_dir_or_id)

        config = None
        for name in Config.supported_config_names:
            config_file = os.path.join(task_dir_or_id, name)
            if os.path.exists(config_file):
                config = OmegaConf.load(config_file)

        assert config is not None, (f'Cannot find any config file in {task_dir_or_id} named `config.json`, '
                                    f'`config.yml` or `config.yaml`')
        envs = Env.load_env(env)
        self._update_envs(config, envs)
        config.local_dir = task_dir_or_id
        return config

    @staticmethod
    def _update_envs(config: Union[DictConfig, ListConfig], envs: Dict[str, str]=None):
        if not envs:
            return config

        def traverse_config(_config: Union[DictConfig, ListConfig, Any]):
            if isinstance(_config, DictConfig):
                for name, value in _config.items():
                    if isinstance(value, BaseContainer):
                        traverse_config(value)
                    else:
                        if name in envs:
                            logger.info(f'Replacing {name} with the value in your environment variables.')
                            setattr(_config, name, envs[name])
            elif isinstance(_config, ListConfig):
                for value in _config:
                    if isinstance(value, BaseContainer):
                        traverse_config(value)
        traverse_config(config)
        return None

    def convert_mcp_servers_to_json(self):
        """Convert the mcp servers to json mcp config."""
        servers = {
            'mcpServers': {

            }
        }
        if self.config.servers:
            for server, server_config in self.config.servers.items():
                servers['mcpServers'][server] = server_config.to_json()
        return servers
