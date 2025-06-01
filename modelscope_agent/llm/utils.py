from dataclasses import dataclass
from typing import Literal, Union, List, Dict, Any
from typing_extensions import Literal, Required, TypedDict
from dataclasses import dataclass, asdict



class Tool(TypedDict, total=False):

    server_name: str

    tool_name: Required[str]

    description: Required[str]

    arguments: Dict[str, Any] = None


@dataclass
class Message:
    role: Required[Literal['system', 'user', 'assistant', 'tool']]

    content: Required[Union[str, List[Dict[str, 'Message']]]]

    tools: List[Tool] = None

    def to_dict(self):
        return asdict(self)

