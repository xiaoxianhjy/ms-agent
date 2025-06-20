from omegaconf import DictConfig

from modelscope_agent.config.config import ConfigLifecycleHandler


class ConfigHandler(ConfigLifecycleHandler):

    def task_begin(self, config: DictConfig, tag: str) -> DictConfig:
        if tag in ('Architecture', 'Reviewer'):
            pass
        elif tag == 'AutoRefiner':
            config.callbacks = ['artifact_callback', 'prompt_callback']
            config.tools.file_system = DictConfig({'mcp': False,
                                                   'exclude': ['create_directory', 'write_file', 'list_files']})
        elif tag == 'HumanEvalRefiner':
            config.callbacks = ['artifact_callback', 'prompt_callback', 'human_eval_callback']
        else:
            config.callbacks = ['artifact_callback', 'prompt_callback']
            delattr(config.tools, 'split_task')

    def task_end(self, config: DictConfig, tag: str) -> DictConfig:
        pass