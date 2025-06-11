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

    @staticmethod
    def extract_metadata(config: DictConfig, llm: LLM, messages: List[Message]):
        assert messages[0].role == 'system' and  messages[0].role == 'user'
        _system = """Here gives a LLM system field, and a user query field, you need to extract the code file information of it, and wraps the final result in <result></result>.
Here shows an example:
system is: You are a code engineer, you should help me to write a code file, which is a part of a complex job. The rules you need to follow are: ...
query is: You should write the index.js file, the file you need to use is main.css and nav.js, the interface in the code is ...

Your answer should be: <result>index.js</result>   
"""
        _query = (f'The input system is: {messages[0].content}\n\n'
                  f'The input query is: {messages[1].content}\n\n'
                  'Now give the code file name:\n')
        _messages = [
            Message(role='system', content=_system),
            Message(role='user', content=_query)
        ]
        if getattr(config.generation_config, 'stream', False):
            message = None
            for msg in llm.generate(messages):
                message = llm.merge_stream_message(message, msg)

            _response_message = message
        else:
            _response_message = llm.generate(messages)
        assert '<result>' in _response_message[-1].content and '</result>' in _response_message[-1].content
        return re.findall(r'<result>(.*?)</result>', _response_message[-1].content)[0]

    def after_generate_response(self, run_status: RunStatus, messages: List[Message]):
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
                code_file = self.extract_metadata(self.config, run_status.llm, messages)
                self.file_system.create_directory('./output')
                self.file_system.write_file(code_file, code)
            run_status.should_stop = True


