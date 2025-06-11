from dataclasses import dataclass

from modelscope_agent.llm.llm import LLM


@dataclass
class RunStatus:

    should_stop: bool = False

    llm: LLM = None