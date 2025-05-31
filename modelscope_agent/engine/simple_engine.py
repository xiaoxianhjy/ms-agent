from typing import List

from modelscope_agent.callbacks.base import Callback
from modelscope_agent.callbacks.run_status import RunStatus
from modelscope_agent.config.config import Config
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools import tool_manager


class SimpleEngine:

    def __init__(self, task_dir_or_id=None, config=None, env=None, **kwargs):
        if task_dir_or_id is None:
            self.config = config
        else:
            self.config = Config.from_task(task_dir_or_id, env)
        self.llm = LLM.from_config(self.config)
        self.callbacks = []
        self.run_status = RunStatus()

    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def loop_callback(self, point, messages: List[Message]):
        for callback in self.callbacks:
            getattr(callback, point)(self.config, self.run_status, messages)

    def parallel_tool_call(self, messages: List[Message]):
        tools = messages[-1]['tools']
        return tool_manager.parallel_tool_call(tools)

    def prepare_tools(self):
        tool_manager.connect()

    def prepare_messages(self, prompt):
        messages = [
            {'system': self.config.prompt.system},
            {'query': prompt},
        ]
        return messages

    async def run(self, prompt, **kwargs):
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
