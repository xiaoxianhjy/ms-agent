from dataclasses import dataclass
from typing import Literal, Union, List, Dict


@dataclass
class Message:

    role: Literal['system', 'user', 'assistant', 'tool'] = None

    content: Union[str, List[Dict[str, 'Message']]] = None

    tools: List[Tool] = None