from omegaconf import DictConfig

from modelscope_agent.utils.utils import assert_package
from .openai import OpenAI


class ModelScope(OpenAI):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        assert_package('openai')
        import openai
        self.client = openai.OpenAI(
            api_key=config.llm.modelscope_api_key,
            base_url=config.llm.modelscope_api_base_url,
        )

