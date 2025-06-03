import asyncio
import inspect
from typing import List, Optional, Dict
import sys
from omegaconf import DictConfig
import importlib
import atexit
from modelscope_agent.callbacks import callbacks_mapping
from modelscope_agent.callbacks import Callback
from modelscope_agent.callbacks import RunStatus
from modelscope_agent.config import Config
from modelscope_agent.engine.memory import memory_mapping
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.rag.base import Rag
from modelscope_agent.rag.utils import rag_mapping
from modelscope_agent.tools import ToolManager
from modelscope_agent.utils.logger import logger


class SimpleEngine:
    """Engine running for single-agent/multi-agent tasks.

    Args:
        task_dir_or_id (`Optional[str]`): The task directory or ID in modelscope hub.
        config (`Optional[DictConfig]`): The configuration object.
        env (`Optional[Dict[str, str]]`): The extra environment configurations.`)
    """

    DEFAULT_SYSTEM_ZH = """你是一个机器人助手。你会被给予许多工具协助你完成任务，你需要使用它们完成交付给你的任务。

1. 仔细查看工具描述，确保调用正确的工具，并传入正确的参数。
2. 你需要在最开始制定一个详细的任务列表，如果工具中有任务管理工具，你需要依赖它存储、更新任务进度以确定接下来需要做的事情。
3. 在任务比较复杂的情况下，你可以将任务拆分为子任务，这些子任务会由其他机器人助手单独执行，并将结果返回给你，注意每个子任务需要传入恰当的system和query，确保子任务可以正常进行。
4. 如果你被分配了一个任务，请确保最后返回的信息严格按照system和query的需求进行，不要在任务未完成的时候结束你的任务。如果有任务管理工具，留意工具给你的提示，你需要严格遵从这些提示。
5. 你可以并行调用工具，并保证在任务完成之前每一轮对话都会调用工具。
"""

    DEFAULT_SYSTEM_EN = """Here is the English translation:

You are a robot assistant. You will be given many tools to help you complete tasks, and you need to use them to accomplish the tasks assigned to you.

1. Carefully review the tool descriptions to ensure you call the correct tool and pass the correct parameters.
2. You need to create a detailed task list at the beginning. If there are task management tools available, you should rely on them to store and update task progress to determine what needs to be done next.
3. For complex tasks, you can break them down into subtasks. These subtasks will be executed separately by other robot assistants, who will return the results to you. Make sure to provide appropriate system instructions and queries for each subtask to ensure they can proceed normally.
4. If you are assigned a task, ensure that the final information you return strictly follows the requirements in the system instructions and query. Do not end your task before it is completed. If there are task management tools, pay attention to the prompts they give you and strictly follow these prompts.
5. You can call tools in parallel and ensure that you call tools in each round of dialogue before the task is completed.
"""

    def __init__(self,
                 task_dir_or_id: Optional[str]=None,
                 config: Optional[DictConfig]=None,
                 env: Optional[Dict[str, str]]=None,
                 **kwargs):
        if task_dir_or_id is None:
            self.config = config
        else:
            self.config = Config.from_task(task_dir_or_id, env)
        self.llm = LLM.from_config(self.config)
        self.callbacks = []
        self.run_status = RunStatus()
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self._register_callback_from_config()
        self.tool_manager: ToolManager = None
        self.memory_tools = []
        self.rag = None

    def register_callback(self, callback: Callback):
        """Register a callback."""
        self.callbacks.append(callback)

    def _register_callback_from_config(self):
        local_dir = self.config.local_dir if hasattr(self.config, 'local_dir') else None
        if hasattr(self.config, 'callbacks'):
            for _callback in self.config.callbacks:
                if _callback.endswith('.py'):
                    if not self.trust_remote_code:
                        raise AssertionError(f'Your config file contains external code, '
                                             f'instantiate the code may be UNSAFE, if you trust the code, '
                                             f'please pass `trust_remote_code=True` or `--trust_remote_code true`')
                    if sys.path[0] != local_dir:
                        assert local_dir is not None, 'Using external py files, but local_dir cannot be found.'
                        sys.path.insert(0, local_dir)
                    callback_file = importlib.import_module(_callback[:-3])
                    module_classes = {name: cls for name, cls in inspect.getmembers(callback_file, inspect.isclass)}
                    for name, cls in module_classes.items():
                        # Find cls which base class is `Callback`
                        if cls.__base__[0] is Callback:
                            self.callbacks.append(cls())
                else:
                    assert _callback in callbacks_mapping
                    self.callbacks.append(callbacks_mapping[_callback]())

    def _loop_callback(self, point, messages: List[Message]):
        for callback in self.callbacks:
            getattr(callback, point)(self.config, self.run_status, messages)

    async def _parallel_tool_call(self, messages: List[Message]):
        tool_call_result = await self.tool_manager.parallel_call_tool(messages[-1].tool_calls)
        assert len(tool_call_result) == len(messages[-1].tool_calls)
        for tool_call_result, tool_call_query in zip(tool_call_result, messages[-1].tool_calls):
            _new_message = Message(
                role='tool',
                content=tool_call_result,
                tool_call_id=tool_call_query['id'],
                name=tool_call_query['tool_name']
            )
            messages.append(_new_message)

    async def _prepare_tools(self):
        self.tool_manager = ToolManager(self.config)
        await self.tool_manager.connect()

    async def _cleanup_tools(self):
        await self.tool_manager.cleanup()

    def _query_documents(self, query):
        if self.rag is not None:
            return query + self.rag.search_documents(query)
        else:
            return query

    def _prepare_messages(self, prompt):
        messages = [
            Message(role='system', content=self.config.prompt.system or self.DEFAULT_SYSTEM_EN),
            Message(role='user', content=prompt or self.config.prompt.query),
        ]
        messages[1].content = self._query_documents(messages[1].content)
        return messages

    def _prepare_memory(self):
        if hasattr(self.config, 'memory') and self.config.memory:
            for _memory in self.config.memory:
                assert _memory in memory_mapping, (f'{_memory} not in memory_mapping, '
                                                   f'which supports: {list(memory_mapping.keys())}')
                self.memory_tools.append(memory_mapping[_memory]())

    def _prepare_rag(self):
        if hasattr(self.config, 'rag') and self.config.rag:
            assert self.config.rag in rag_mapping
            self.rag: Rag = rag_mapping(self.config.rag)()

    def _refine_memory(self, messages: List[Message]):
        for memory_tool in self.memory_tools:
            messages = memory_tool.refine(messages)
        return messages

    async def run(self, prompt, **kwargs):
        try:
            await self._prepare_tools()
            self._prepare_memory()
            self._prepare_rag()
            messages = self._prepare_messages(prompt)
            self._loop_callback('on_task_begin', messages)
            while not self.run_status.should_stop:
                self._loop_callback('on_generate_response', messages)
                messages = self._refine_memory(messages)
                tools = await self.tool_manager.get_tools()
                _response_message = self.llm.generate(messages, tools=tools)
                messages.append(_response_message)
                self._loop_callback('after_generate_response', messages)
                self._loop_callback('on_tool_call', messages)
                if messages[-1].tool_calls:
                    await self._parallel_tool_call(messages)
                else:
                    self.run_status.should_stop = True
                self._loop_callback('after_tool_call', messages)
            self._loop_callback('on_task_end', messages)
            logger.info(messages[-1].content)
            await self._cleanup_tools()
        except Exception as e:
            if self.config.help:
                logger.error(f'Runtime error, please follow the instructions:\n\n {self.config.help}')
            raise e
