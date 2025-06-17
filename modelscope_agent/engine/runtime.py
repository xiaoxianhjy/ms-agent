from dataclasses import dataclass
from typing import Optional

from modelscope_agent.llm.llm import LLM


@dataclass
class Runtime:

    should_stop: bool = False

    llm: LLM = None

    tag: Optional[str] = None