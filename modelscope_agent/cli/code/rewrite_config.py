from omegaconf import DictConfig

from modelscope_agent.engine.code.base import Code


class RewriteConfig(Code):

    def __init__(self, config):
        super().__init__(config)

    async def run(self, inputs, **kwargs):
        self.config.callbacks = ['artifact_callback', 'prompt_callback']
        self.config.tools.file_system = DictConfig({'mcp': False})
        return inputs