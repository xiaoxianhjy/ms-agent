from omegaconf import DictConfig, ListConfig

from modelscope_agent.config.config import ConfigLifecycleHandler


class ConfigHandler(ConfigLifecycleHandler):

    def task_begin(self, config: DictConfig, tag: str) -> DictConfig:
        if tag == 'Architecture':
            # only need arch review
            config.callbacks = ['codes/coding_callback']
        elif tag == 'HumanEvalRefiner':
            config.callbacks = ['codes/human_eval_callback', 'codes/coding_callback']
        elif 'worker' in tag:
            config.callbacks = ['codes/artifact_callback']
            delattr(config.tools, 'split_task')
            config.tools.file_system = DictConfig({'mcp': False,
                                                   'exclude': ['create_directory', 'write_file', 'list_files']})
        return config