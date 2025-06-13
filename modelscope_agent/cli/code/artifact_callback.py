import os.path
import re
from typing import List

from omegaconf import DictConfig

from modelscope_agent.callbacks import Callback, RunStatus
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool


class ArtifactCallback(Callback):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, run_status: RunStatus, messages: List[Message]):
        await self.file_system.connect()

    @staticmethod
    def extract_metadata(config: DictConfig, llm: LLM, messages: List[Message]):
        assert messages[0].role == 'system' and  messages[1].role == 'user'
        _system = """You are a file name parser, I will give a user query field to you, you need to extract the code file name from it.
Always remember your task is not generating the code, but parse the file name from the query.
Here shows an example:
query is: You should write the index.js file, the file you need to use is main.css and nav.js, the interface in the code is ...

Your answer should be: index.js 
"""
        _query = (f'The input query is: {messages[1].content}\n\n'
                  'Now give me the code file name without any other information:\n')
        _messages = [
            Message(role='system', content=_system),
            Message(role='user', content=_query)
        ]
        if getattr(config.generation_config, 'stream', False):
            message = None
            for msg in llm.generate(_messages):
                message = llm.merge_stream_message(message, msg)

            _response_message = message
        else:
            _response_message = llm.generate(_messages)
        return _response_message.content

    async def after_generate_response(self, run_status: RunStatus, messages: List[Message]):
        last_message_content = messages[-1].content
        if '</code>' in last_message_content:
            code = ''
            recording = False
            for message in messages:
                if message.role == 'assistant':
                    if '<code>' in message.content and '</code>' in message.content:
                        code += message.content.split('<code>')[1].split('</code>')[0]
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
                    code_file = self.extract_metadata(self.config, run_status.llm, messages)
                    await self.file_system.create_directory('output')
                    await self.file_system.write_file(os.path.join('output', code_file), code)
                    messages.append(Message(role='assistant', content=f'Original query: {messages[1].content}'
                                                                      f'Task sunning successfully, '
                                                                      f'the code has been saved in the {code_file} file.'))
                except Exception as e:
                    messages.append(Message(role='assistant', content=f'Original query: {messages[1].content}'
                                                                      f'Task sunning failed with error {e} please consider retry generation.'))
            else:
                messages.append(Message(role='assistant', content=f'Original query: {messages[1].content}'
                                                                  f'Task sunning failed, code format error, please consider retry generation.'))
            run_status.should_stop = True
