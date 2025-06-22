# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import List

import json
from file_parser import extract_code_blocks
from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.callbacks import Callback
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class CodingCallback(Callback):
    """Save the output code to local disk.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        extract_code_blocks()

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if messages[-1].tool_calls:
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
                    metadata = self.extract_metadata(self.config, runtime.llm, messages)
                    metadata = json.loads(metadata)
                    code_file = metadata.get('output')
                    task_type = metadata.get('task_type')
                    if task_type != 'generate_code':
                        return
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
                        f'Task sunning failed with error {e}.'
                    )
                    logger.error(result)
                    messages.append(Message(role='user', content=result))
            else:
                result = (
                    f'[Failed] Original query: {messages[1].content} Task sunning failed, '
                    f'code format error.')
                logger.error(result)
                messages.append(Message(role='user', content=result))
            runtime.should_stop = True
