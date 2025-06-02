import inspect
from typing import Any, List, Dict, Optional, Generator

from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message, Tool, ToolCall
from modelscope_agent.utils.utils import assert_package_exist


def _stream_generator() -> Generator[Message, None, None]:
    pass

class OpenAI(LLM):

    def __init__(self, config: DictConfig, base_url: Optional[str] = None,  api_key: Optional[str] = None):
        super().__init__(config)
        assert_package_exist('openai')
        import openai
        self.model: str = config.llm.model
        base_url = base_url or config.llm.openai_base_url
        api_key = api_key or config.llm.openai_api_key
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        exclude_fields = {"model", "base_url", "api_key"}
        self.args: Dict = {k: v for k, v in OmegaConf.to_container(getattr(config, 'generation_config', {}), resolve=True).items() if k not in exclude_fields}

    def generate(self, messages: List[Message], model: Optional[str] = None, tools: List[Tool] = None, **kwargs) -> Message | Generator[Message, None, None]:
        parameters = inspect.signature(self.client.chat.completions.create).parameters
        args = self.args.copy()
        args.update(kwargs)
        stream = args.get('stream', False)

        args = {key: value for key, value in args.items() if key in parameters}

        if tools:
            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': f'{tool.get("server_name")}---{tool["tool_name"]}' if tool.get('server_name') else tool['tool_name'],
                        'description': tool['description'],
                        'parameters': tool['parameters']
                    }
                } for tool in tools
            ]
        completion = self._call_llm(model or self.model, messages, tools, **args)

        if stream:
            return self.stream_continue_generate(completion)
        else:
            return self.continue_generate(messages, completion, tools, **args)

    def _call_llm(self, model, messages, tools, **kwargs):
        messages = self.format_input_message(messages)
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )

    def stream_continue_generate(self, messages: List[Message], completion, tools: List[Tool] = None, **kwargs) -> Generator[Message, None, None]:
        pass

    def format_output_message(self, completion) -> Message:
        content = completion.choices[0].message.content
        reasoning_content = completion.choices[0].message.reasoning_content
        tool_calls = None
        if completion.choices[0].message.tool_calls:
            tool_calls = [ToolCall(
                    id=tool_call.id,
                    index=tool_call.index,
                    type=tool_call.type,
                    arguments=tool_call.function.arguments,
                    tool_name=tool_call.function.name
                ) for tool_call in completion.choices[0].message.tool_calls
            ]
        return Message(role='assistant', content=content, reasoning_content=reasoning_content, tool_calls=tool_calls, id=completion.id)

    def stream_format_output_message(self, completion) -> Generator[Message, None, None]:
        pass

    def _continue_generate(self, messages: List[Message], completion, tools: List[Tool] = None, **kwargs):
        # ref: https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2862210.html&renderType=iframe
        # TODO: 移到dashscope_llm并找到真正openai的续写方式
        if messages[-1].to_dict().get('partial', False):
            new_meessage = self.format_output_message(completion)
            messages[-1].reasoning_content += new_meessage.reasoning_content
            messages[-1].content += new_meessage.content
            if new_meessage.tool_calls:
                if messages[-1].tool_calls:
                    messages[-1].tool_calls += new_meessage.tool_calls
                else:
                    messages[-1].tool_calls = new_meessage.tool_calls
        else:
            messages.append(self.format_output_message(completion))
            messages[-1].partial = True

        messages = self.format_input_message(messages)
        print(f'messages: {messages}')
        return self._call_llm(messages, tools, **kwargs)

    def continue_generate(self, messages: List[Message], completion, tools: List[Tool] = None, **kwargs) -> Message:
        # finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        print(f'finish_reason: {completion.choices[0].finish_reason}')

        if completion.choices[0].finish_reason in ['length', 'null']:
            completion = self._continue_generate(messages, completion, tools, **kwargs)
            return self.continue_generate(messages, completion, tools, **kwargs)
        else:
            return self.format_output_message(completion)

    def format_input_message(self, messages: List[Message]) -> List[Dict[str, Any]]:
        openai_messages = []
        for message in messages:
            if isinstance(message, Message):
                message = message.to_dict()

            if message.get('tool_calls'):
                tool_calls = list()
                for tool_call in message['tool_calls']:
                    function_data: Function = {
                        "name": tool_call['tool_name'],
                        "arguments": tool_call['arguments']
                    }
                    tool_call: ChatCompletionMessageToolCall = {
                        "id": tool_call['id'],
                        "function": function_data,
                        "type": tool_call['type'],
                    }
                    tool_calls.append(tool_call)
                message['tool_calls'] = tool_calls

            input_msg = {'role', 'content', 'tool_calls', 'partial'}
            message = {key: value for key, value in message.items() if key in input_msg and value}

            openai_messages.append(message)


        return openai_messages


if __name__ == '__main__':
    import os
    from omegaconf import OmegaConf

    # 创建一个嵌套的字典结构
    conf: DictConfig = OmegaConf.create({
        "llm": {
            "model": "Qwen/Qwen3-235B-A22B",
            "openai_base_url": "https://api-inference.modelscope.cn/v1",
            "openai_api_key": os.getenv("MODELSCOPE_API_KEY"),
            "stream": True,
        }
    })

    messages = [
        Message(role='assistant', content='You are a helpful assistant.'),
        # Message(role='user', content='经度：116.4074，纬度：39.9042是什么地方。用这个名字作为目录名'),
        Message(role='user', content='请你简单介绍杭州'),

    ]

    # tools = [
    #     Tool(server_name='amap-maps', tool_name='maps_regeocode', description='将一个高德经纬度坐标转换为行政区划地址信息', parameters={'type': 'object', 'properties': {'location': {'type': 'string', 'description': '经纬度'}}, 'required': ['location']}),
    #     Tool(tool_name='mkdir', description='在文件系统创建目录', parameters={'type': 'object', 'properties': {'dir_name': {'type': 'string', 'description': '目录名'}}, 'required': ['location']})
    # ]
    tools = None


    # 打印配置
    print(OmegaConf.to_yaml(conf))

    llm = OpenAI(conf)

    # conf配置
    # res = llm.generate(messages)
    # for chunk in res:
    #     print(chunk)

    # kwargs覆盖conf
    message = llm.generate(messages=messages, tools=tools, stream=False, extra_body={'enable_thinking': False})
    print(message)
    # messages.append(message)
    # messages.append(Message(role='tool', content='北京市朝阳区崔各庄阿里巴巴朝阳科技园'))
    # message = llm.generate(messages=messages, tools=tools, stream=False, extra_body={'enable_thinking': False})
    # print(message)
