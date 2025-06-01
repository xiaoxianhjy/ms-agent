from omegaconf import DictConfig

from modelscope_agent.utils.utils import assert_package_exist
from modelscope_agent.llm.openai_llm import OpenAI


class ModelScope(OpenAI):

    def __init__(self, config: DictConfig):
        super().__init__(config, base_url=config.llm.modelscope_base_url, api_key=config.llm.modelscope_api_key)
