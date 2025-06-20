from omegaconf import DictConfig

from modelscope_agent.config.config import ConfigLifecycleHandler


class ConfigHandler(ConfigLifecycleHandler):

    def task_begin(self, config: DictConfig, tag: str) -> DictConfig:
        if tag == 'Architecture':
            # only need arch review
            config.callbacks = ['codes/arch_review_callback']
        elif tag == 'Reviewer':
            # no callbacks needed
            config.callbacks = []
        elif tag == 'AutoRefiner':
            # no callbacks needed
            config.callbacks = []
        elif tag == 'HumanEvalRefiner':
            config.callbacks = ['codes/human_eval_callback']
        elif 'worker' in tag:
            config.callbacks = ['codes/artifact_callback', 'codes/prompt_callback']
            delattr(config.tools, 'split_task')
            config.tools.file_system = DictConfig({'mcp': False,
                                                   'exclude': ['create_directory', 'write_file', 'list_files']})
        return config