# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import inspect
import os.path
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Union

import json
from modelscope_agent.callbacks import Callback, callbacks_mapping
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message, Tool
from modelscope_agent.rag.base import Rag
from modelscope_agent.rag.utils import rag_mapping
from modelscope_agent.tools import ToolManager
from modelscope_agent.utils.llm_utils import retry
from modelscope_agent.utils.logger import logger
from omegaconf import DictConfig

from .base import Agent
from .memory import Memory, memory_mapping
from .plan.base import Planer
from .plan.utils import planer_mapping
from .runtime import Runtime


class LLMAgent(Agent):
    """Agent running for llm-based tasks.

    Args:
        config_dir_or_id (`Optional[str]`): The directory or id of the config file.
        config (`Optional[DictConfig]`): The configuration object.
        env (`Optional[Dict[str, str]]`): The extra environment variables.
    """

    DEFAULT_SYSTEM = """

You are a robot assistant. You will be given many tools to help you complete tasks, and you need to use them to accomplish the tasks assigned to you.

1. Carefully review the tool descriptions to ensure you call the correct tool and pass the correct parameters.
2. You need to create a detailed task list at the beginning. If there are task management tools available, you should rely on them to store and update task progress to determine what needs to be done next.
3. For complex tasks, you can break them down into subtasks. These subtasks will be executed separately by other robot assistants, who will return the results to you. Make sure to provide appropriate system instructions and queries for each subtask to ensure they can proceed normally.
4. If you are assigned a task, ensure that the final information you return strictly follows the requirements in the system instructions and query. Do not end your task before it is completed. If there are task management tools, pay attention to the prompts they give you and strictly follow these prompts.
5. You can call tools in parallel and ensure that you call tools in each round of dialogue before the task is completed.
""" # noqa

    def __init__(self,
                 config_dir_or_id: Optional[str] = None,
                 config: Optional[DictConfig] = None,
                 env: Optional[Dict[str, str]] = None,
                 **kwargs):
        super().__init__(
            config_dir_or_id,
            config,
            env,
            tag=kwargs.get('tag'),
            trust_remote_code=kwargs.get('trust_remote_code', False))
        self.callbacks: List[Callback] = []
        self.tool_manager: Optional[ToolManager] = None
        self.memory_tools: List[Memory] = []
        self.planer: Optional[Planer] = None
        self.rag: Optional[Rag] = None
        self.llm: Optional[LLM] = None
        self.runtime: Optional[Runtime] = None
        self.round: int = 0
        self._task_begin()

    def register_callback(self, callback: Callback):
        """Register a callback."""
        self.callbacks.append(callback)

    def _register_callback_from_config(self):
        local_dir = self.config.local_dir if hasattr(self.config,
                                                     'local_dir') else None
        if hasattr(self.config, 'callbacks'):
            callbacks = self.config.callbacks or []
            for _callback in callbacks:
                subdir = os.path.dirname(_callback)
                assert local_dir is not None, 'Using external py files, but local_dir cannot be found.'
                if subdir:
                    subdir = os.path.join(local_dir, subdir)
                _callback = os.path.basename(_callback)
                if _callback not in callbacks_mapping:
                    if not self.trust_remote_code:
                        raise AssertionError(
                            '[External Code Found] Your config file contains external code, '
                            'instantiate the code may be UNSAFE, if you trust the code, '
                            'please pass `trust_remote_code=True` or `--trust_remote_code true`'
                        )
                    if local_dir not in sys.path:
                        sys.path.insert(0, local_dir)
                    if subdir and subdir not in sys.path:
                        sys.path.insert(0, subdir)
                    callback_file = importlib.import_module(_callback)
                    module_classes = {
                        name: cls
                        for name, cls in inspect.getmembers(
                            callback_file, inspect.isclass)
                    }
                    for name, cls in module_classes.items():
                        # Find cls which base class is `Callback`
                        if cls.__bases__[
                                0] is Callback and cls.__module__ == _callback:
                            self.callbacks.append(cls(self.config))
                else:
                    self.callbacks.append(callbacks_mapping[_callback](
                        self.config))

    async def _loop_callback(self, point, messages: List[Message]):
        for callback in self.callbacks:
            await getattr(callback, point)(self.runtime, messages)

    async def _parallel_tool_call(self,
                                  messages: List[Message]) -> List[Message]:
        tool_call_result = await self.tool_manager.parallel_call_tool(
            messages[-1].tool_calls)
        assert len(tool_call_result) == len(messages[-1].tool_calls)
        for tool_call_result, tool_call_query in zip(tool_call_result,
                                                     messages[-1].tool_calls):
            _new_message = Message(
                role='tool',
                content=tool_call_result,
                tool_call_id=tool_call_query['id'],
                name=tool_call_query['tool_name'])
            messages.append(_new_message)
            self._log_output(_new_message.content, self.tag)
        return messages

    async def _prepare_tools(self):
        self.tool_manager = ToolManager(self.config)
        await self.tool_manager.connect()

    async def _cleanup_tools(self):
        await self.tool_manager.cleanup()

    async def _prepare_messages(
            self, inputs: Union[List[Message], str]) -> List[Message]:
        if isinstance(inputs, list):
            return inputs
        assert isinstance(
            inputs, str
        ), f'inputs can be either a list or a string, but current is {type(inputs)}'
        system = None
        query = None
        if hasattr(self.config, 'prompt'):
            system = getattr(self.config.prompt, 'system', None)
            query = getattr(self.config.prompt, 'query', None)
        messages = [
            Message(role='system', content=system or self.DEFAULT_SYSTEM),
            Message(role='user', content=inputs or query),
        ]
        if self.rag is not None:
            messages = await self.rag.run(messages)
        return messages

    async def _prepare_memory(self):
        if hasattr(self.config, 'memory'):
            for _memory in (self.config.memory or []):
                assert _memory.name in memory_mapping, (
                    f'{_memory.name} not in memory_mapping, '
                    f'which supports: {list(memory_mapping.keys())}')
                self.memory_tools.append(memory_mapping[_memory.name](
                    self.config))

    async def _prepare_planer(self):
        if hasattr(self.config, 'planer'):
            planer = self.config.planer
            if planer is not None:
                assert planer.name in planer_mapping, (
                    f'{planer.name} not in planer_mapping, '
                    f'which supports: {list(planer_mapping.keys())}')
                self.planer = planer_mapping[planer.name](self.config)

    async def _prepare_rag(self):
        if hasattr(self.config, 'rag'):
            rag = self.config.rag
            if rag is not None:
                assert rag.name in rag_mapping, (
                    f'{rag.name} not in rag_mapping, '
                    f'which supports: {list(rag_mapping.keys())}')
                self.rag: Rag = rag_mapping(rag.name)(self.config)

    async def _refine_memory(self, messages: List[Message]) -> List[Message]:
        for memory_tool in self.memory_tools:
            messages = await memory_tool.run(messages)
        return messages

    async def _update_plan(self, messages: List[Message]) -> List[Message]:
        if self.planer is not None:
            messages = await self.planer.update_plan(self.runtime, messages)
        return messages

    def _handle_stream_message(self, messages: List[Message],
                               tools: List[Tool]) -> List[Message]:
        message = None
        for msg in self.llm.generate(messages, tools=tools):
            yield msg
            message = self.llm.merge_stream_message(message, msg)

        return message

    @staticmethod
    def _log_output(content: str, tag: str):
        for line in content.split('\n'):
            for _line in line.split('\\n'):
                logger.info(f'[{tag}] {_line}')

    @retry(max_attempts=2)
    async def _step(self, messages: List[Message], tag: str) -> List[Message]:
        messages = deepcopy(messages)
        # Refine memory
        messages = await self._refine_memory(messages)
        # Do plan
        messages = await self._update_plan(messages)
        await self._loop_callback('on_generate_response', messages)
        tools = await self.tool_manager.get_tools()
        if hasattr(self.config, 'generation_config') and getattr(
                self.config.generation_config, 'stream', False):
            _response_message = self._handle_stream_message(
                messages, tools=tools)
        else:
            _response_message = self.llm.generate(messages, tools=tools)

        if _response_message.content:
            self._log_output('[assistant]:', tag=tag)
            self._log_output(_response_message.content, tag=tag)
        if _response_message.tool_calls:
            self._log_output('[tool_calling]:', tag=tag)
            for tool_call in _response_message.tool_calls:
                self._log_output(json.dumps(tool_call), tag=tag)

        if messages[-1] is not _response_message:
            messages.append(_response_message)
        await self._loop_callback('after_generate_response', messages)
        await self._loop_callback('on_tool_call', messages)

        if _response_message.tool_calls:
            messages = await self._parallel_tool_call(messages)
        else:
            self.runtime.should_stop = True
        await self._loop_callback('after_tool_call', messages)
        return messages

    def _prepare_llm(self):
        self.llm: LLM = LLM.from_config(self.config)

    def _prepare_runtime(self):
        self.runtime: Runtime = Runtime(llm=self.llm)

    async def run(self, inputs, **kwargs) -> List[Message]:
        """Run the agent, mainly contains a llm calling and tool calling loop.

        Args:
            inputs(`Union[str, List[Message]]`): The inputs can be a prompt string,
                or a list of messages from the previous agent
        Returns:
            The final messages
        """
        try:
            self._register_callback_from_config()
            self._prepare_llm()
            self._prepare_runtime()
            await self._prepare_tools()
            await self._prepare_memory()
            await self._prepare_planer()
            await self._prepare_rag()
            self.runtime.tag = self.tag
            messages = await self._prepare_messages(inputs)
            await self._loop_callback('on_task_begin', messages)
            if self.planer:
                messages = await self.planer.make_plan(messages, self.runtime)
            for message in messages:
                self._log_output('[' + message.role + ']:', tag=self.tag)
                self._log_output(message.content, tag=self.tag)
            while not self.runtime.should_stop:
                messages = await self._step(messages, self.tag)
                self.round += 1
                if self.round >= 20:
                    if not self.runtime.should_stop:
                        messages.append(Message(role='assistant', content='Task failed, max round(20) exceeded.'))
                    self.runtime.should_stop = True
                    break
            await self._loop_callback('on_task_end', messages)
            await self._cleanup_tools()
            return messages
        except Exception as e:
            if hasattr(self.config, 'help'):
                logger.error(
                    f'[{self.tag}] Runtime error, please follow the instructions:\n\n {self.config.help}'
                )
            raise e
