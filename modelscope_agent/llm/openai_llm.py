import inspect
from typing import Any, List, Dict

from omegaconf import DictConfig

from modelscope_agent.utils.utils import assert_package
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message


class OpenAI(LLM):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        assert_package('openai')
        import openai
        self.model: str = config.llm.model
        self.client = openai.OpenAI(
            api_key=config.llm.openai_api_key,
            base_url=config.llm.openai_base_url,
        )
        exclude_fields = {"model", "base_url", "api_key"}
        self.args: Dict = {k: v for k, v in OmegaConf.to_container(config.llm, resolve=True).items() if k not in exclude_fields}

    def generate(self, messages: List['Message'], model=None, tools=None, **kwargs) -> Any:
        parameters = inspect.signature(self.client.chat.completions.create).parameters
        args = self.args.copy()
        args.update(kwargs)
        args = {key: value for key, value in args.items() if key in parameters}
        messages = self.format_message(messages)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            **args
        )
        return completion

    def format_message(self, messages: List['Message']) -> List[Dict[str, Any]]:
        openai_messages = []
        for message in messages:
            message = message.to_dict()
            openai_messages.append(message)
        return openai_messages


if __name__ == '__main__':
    import os
    from omegaconf import OmegaConf

    # 创建一个嵌套的字典结构
    conf: DictConfig = OmegaConf.create({
        "llm": {
            "model": "Qwen/Qwen3-235B-A22B",
            "openai_base_url": "https://api-inference.modelscope.cn/v1/",
            "openai_api_key": os.getenv("MODELSCOPE_API_KEY"),
            "stream": True,
        }
    })

    messages = [
        Message(role='assistant', content='You are a helpful assistant.'),
        Message(role='user', content='请介绍杭州'),
    ]

    # 打印配置
    print(OmegaConf.to_yaml(conf))

    llm = OpenAI(conf)

    # conf配置
    # res = llm.generate(messages)
    # for chunk in res:
    #     print(chunk)

    # kwargs覆盖conf
    res = llm.generate(messages, stream=False, extra_body={'enable_thinking': False}, max_tokens=5)
    print(res)
