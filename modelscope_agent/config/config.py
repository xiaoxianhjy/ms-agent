import argparse
import os.path
from copy import deepcopy
from typing import Dict, Union, Any

from modelscope import snapshot_download
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.basecontainer import BaseContainer
from modelscope_agent.utils import get_logger
from .env import Env

logger = get_logger()


class Config:

    supported_config_names = ['config.json', 'config.yml', 'config.yaml']

    @classmethod
    def from_task(cls, task_dir_or_id: str, env: Dict[str, str] = None) -> Union[DictConfig, ListConfig]:
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
        if os.path.isfile(task_dir_or_id):
            config = OmegaConf.load(task_dir_or_id)
            task_dir_or_id = os.path.dirname(task_dir_or_id)
        else:
            for name in Config.supported_config_names:
                config_file = os.path.join(task_dir_or_id, name)
                if os.path.exists(config_file):
                    config = OmegaConf.load(config_file)

        assert config is not None, (f'Cannot find any config file in {task_dir_or_id} named `config.json`, '
                                    f'`config.yml` or `config.yaml`')
        envs = Env.load_env(env)
        cls._update_config(config, envs)
        _dict_config = cls.parse_args()
        cls._update_config(config, _dict_config)
        config.local_dir = task_dir_or_id
        return config

    @staticmethod
    def parse_args():
        arg_parser = argparse.ArgumentParser()
        args, unknown = arg_parser.parse_known_args()
        _dict_config = {}
        if unknown:
            for idx in range(0, len(unknown), 2):
                key = unknown[idx]
                value = unknown[idx + 1]
                assert key.startswith('--'), f'Parameter not correct: {unknown}'
                _dict_config[key[2:]] = value
        return _dict_config

    @staticmethod
    def _update_config(config: Union[DictConfig, ListConfig], extra: Dict[str, str]=None):
        if not extra:
            return config

        def traverse_config(_config: Union[DictConfig, ListConfig, Any]):
            if isinstance(_config, DictConfig):
                for name, value in _config.items():
                    if isinstance(value, BaseContainer):
                        traverse_config(value)
                    else:
                        if name in extra:
                            logger.info(f'Replacing {name} with extra value.')
                            setattr(_config, name, extra[name])
                        if (isinstance(value, str) and value.startswith('<') and
                                value.endswith('>') and value[1:-1] in extra):
                            logger.info(f'Replacing {value} with extra value.')
                            setattr(_config, name, extra[name])

            elif isinstance(_config, ListConfig):
                for idx in range(len(_config)):
                    value = _config[idx]
                    if isinstance(value, BaseContainer):
                        traverse_config(value)
                    else:
                        if (isinstance(value, str) and value.startswith('<') and
                                value.endswith('>') and value[1:-1] in extra):
                            logger.info(f'Replacing {value} with extra value.')
                            _config[idx] = extra[value[1:-1]]

        traverse_config(config)
        return None

    @staticmethod
    def convert_mcp_servers_to_json(config: Union[DictConfig, ListConfig]):
        """Convert the mcp servers to json mcp config."""
        servers = {
            'mcpServers': {

            }
        }
        if config.servers:
            for server, server_config in config.servers.items():
                servers['mcpServers'][server] = deepcopy(server_config)
        return servers
