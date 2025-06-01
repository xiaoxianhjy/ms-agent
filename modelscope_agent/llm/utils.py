from dataclasses import dataclass
from typing import Literal, Union, List, Dict, Any
from typing_extensions import Literal, Required, TypedDict
from dataclasses import dataclass, asdict


class ToolCall(TypedDict, total=False):
    id: str = 'default_id'
    index: int = 0
    type: str = 'function'
    tool_name: Required[str]
    arguments: str = None


class Tool(TypedDict, total=False):
    server_name: str = None

    tool_name: Required[str]

    description: Required[str]

    parameters: Dict[str, Any] = None

# {'role': 'assistant', 'content': '', 'tool_calls': [
#             ChatCompletionMessageToolCall(id='call_eaa1051b186744ed97f4ef', function=Function(
#                 arguments='{"keywords": "咖啡馆", "location": "120.096834,30.274659", "radius": "1000"}',
#                 name='amap-maps---maps_around_search'), type='function', index=0)]

@dataclass
class Message:
    role: Required[Literal['system', 'user', 'assistant', 'tool']]

    content: Required[Union[str, List[Dict[str, 'Message']]]]

    tool_calls: List[ToolCall] = None

    # 输出需要，输入时pop
    reasoning_content: str = None

    # 记录模型返回的request_id，以便调试排查
    id: str = None

    def to_dict(self):
        return asdict(self)

