# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import List

import json
from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.callbacks import Callback
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class ArtifactCallback(Callback):
    """Save the output code to local disk.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    @staticmethod
    def extract_metadata(config: DictConfig, llm: LLM,
                         messages: List[Message]):
        """Query Intent detection
        """

        assert messages[0].role == 'system' and messages[1].role == 'user'
        _system = """You are a task resolver, a user query will be given to you, identify the task type then extract the code file name from it.
Always remember your task is not generating the code, but parse the task type and the relative file name from the query.
Here shows an example:
query is:
You should write the js/index.js file, the file you need to use is main.css and js/nav.js, the interface in the code is ...

Your answer should be:
{"task_type": "generate_code", "output": "js/index.js"}
in json, do not add ``` or other explanations.

If you find the task type is not generating code, you should return task_type: `other`, for example:
query is:
You should analyze the code file: js/index.js, then find out the problems...

Your answer should be:
{"task_type": "analyze", "input": "js/index.js"}
""" # noqa
        _query = (
            f'The input query is: {messages[1].content}\n\n'
            'Now output the code file name and task type without any other information:\n'
        )
        _messages = [
            Message(role='system', content=_system),
            Message(role='user', content=_query)
        ]
        if hasattr(config, 'generation_config') and getattr(
                config.generation_config, 'stream', False):
            message = None
            for msg in llm.generate(_messages):
                message = llm.merge_stream_message(message, msg)

            _response_message = message
        else:
            _response_message = llm.generate(_messages)
        return _response_message.content

    @staticmethod
    def hot_fix_code_piece(last_message_content):
        """Damn!"""
        last_message_content = last_message_content.replace(
            '```html\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```js\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```python\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```jsx\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```java\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```javascript\n', '<code>\n')
        last_message_content = last_message_content.replace(
            '```css\n', '<code>\n')
        last_message_content = last_message_content.replace('```', '</code>')
        last_message_content = last_message_content.replace(
            '<code>\n<code>', '<code>')
        last_message_content = last_message_content.replace(
            '</code>\n</code>', '</code>')
        return last_message_content

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if messages[-1].tool_calls:
            return
        metadata = self.extract_metadata(self.config, runtime.llm, messages)
        metadata = json.loads(metadata)
        code_file = metadata.get('output')
        task_type = metadata.get('task_type')
        if task_type != 'generate_code':
            return
        last_message_content = self.hot_fix_code_piece(messages[-1].content)
        messages[-1].content = last_message_content
        if '</code>' in last_message_content:
            code = ''
            recording = False
            for message in messages:
                if message.role == 'assistant':
                    if '<code>' in message.content and '</code>' in message.content:
                        code += message.content.split('<code>')[1].split(
                            '</code>')[0]
                        break
                    elif '<code>' in code:
                        code += message.content.split('<code>')[1]
                        recording = True
                    elif '</code>' in code:
                        code += message.content.split('</code>')[0]
                        recording = False
                    elif recording:
                        code += message.content
            if code:
                try:
                    dirs = os.path.dirname(code_file)
                    await self.file_system.create_directory(dirs)
                    await self.file_system.write_file(code_file, code)
                    messages.append(
                        Message(
                            role='assistant',
                            content=
                            f'[OK] <file:{code_file}>Original query: {messages[1].content}'
                            f'Task sunning successfully, '
                            f'the code has been saved in the {code_file} file.'
                        ))
                except Exception as e:
                    result = (
                        f'[Failed] Original query: {messages[1].content} '
                        f'Task sunning failed with error {e} please consider retry generation.'
                    )
                    logger.error(result)
                    messages.append(Message(role='user', content=result))
            else:
                result = (
                    f'[Failed] Original query: {messages[1].content} Task sunning failed, '
                    f'code format error, please consider retry generation.')
                logger.error(result)
                messages.append(Message(role='user', content=result))
            runtime.should_stop = True
