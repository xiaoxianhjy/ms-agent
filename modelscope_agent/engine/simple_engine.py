from typing import List

from modelscope_agent.callbacks.base import Callback
from modelscope_agent.callbacks.run_status import RunStatus
from modelscope_agent.config.config import Config
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message


class SimpleEngine:

    def __init__(self, task_dir_or_id=None, env=None, **kwargs):
        self.config = Config.from_task(task_dir_or_id, env)
        self.llm = LLM.from_config(self.config)
        self.callbacks = []
        self.run_status = RunStatus()
        self.tools = {}

    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def loop_callback(self, point, messages: List[Message]):
        for callback in self.callbacks:
            getattr(callback, point)(self.config, self.run_status, messages)

    def parallel_tool_call(self, messages: List[Message]):
        pass

    def prepare_tools(self):
        pass

    def prepare_messages(self, prompt):
        messages = [
            {'system': self.config.prompt.system},
            {'query': prompt},
        ]
        return messages

    def run(self, prompt, **kwargs):
        messages = self.prepare_messages(prompt)
        self.loop_callback('on_task_begin', messages)
        while not self.run_status.should_stop:
            self.loop_callback('on_generate_response', messages)
            self.llm.generate(messages)
            self.loop_callback('after_generate_response', messages)
            self.loop_callback('on_tool_call', messages)
            self.parallel_tool_call(messages)
            self.loop_callback('after_tool_call', messages)
        self.loop_callback('on_task_end', messages)
