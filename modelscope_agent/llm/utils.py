from dataclasses import dataclass
from typing import Literal, Union, List, Dict
from typing_extensions import Literal, Required, TypedDict


class Message(TypedDict, total=False):

    role: Required[Literal['system', 'user', 'assistant', 'tool']]

    content: Required[Union[str, List[Dict[str, 'Message']]]]

    tools: List[Tool] = None