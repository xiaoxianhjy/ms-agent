import os.path
import re
from typing import List

from omegaconf import DictConfig
from modelscope_agent.callbacks import Callback
from modelscope_agent.engine.runtime import Runtime
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger

logger = get_logger()


class ArtifactCallback(Callback):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.code = False

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    @staticmethod
    def extract_metadata(messages: List[Message]):
        content = '\n\n'.join(m.content for m in messages)
        pattern = r'<output>(.*?)</output>'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        else:
            return None

    def hot_fix_code_piece(self, last_message_content):
        last_message_content = last_message_content.replace('```html\n', '<code>\n')
        last_message_content = last_message_content.replace('```js\n', '<code>\n')
        last_message_content = last_message_content.replace('```javascript\n', '<code>\n')
        last_message_content = last_message_content.replace('```css\n', '<code>\n')
        last_message_content = last_message_content.replace('```', '</code>')
        last_message_content = last_message_content.replace('<code>\n<code>', '<code>')
        last_message_content = last_message_content.replace('</code>\n</code>', '</code>')
        return last_message_content

    async def after_generate_response(self, runtime: Runtime, messages: List[Message]):
        if runtime.tag == 'Default workflow':
            return
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
                self.code = True
                try:
                    code_file = self.extract_metadata(messages)
                    dirs = os.path.dirname(code_file)
                    await self.file_system.create_directory(os.path.join('output', dirs))
                    await self.file_system.write_file(os.path.join('output', code_file), code)
                    messages.append(Message(role='assistant', content=f'[OK] <file:{code_file}>Original query: {messages[1].content}'
                                                                      f'Task sunning successfully, '
                                                                      f'the code has been saved in the {code_file} file.'))
                except Exception as e:
                    result = f'[Failed] Original query: {messages[1].content} Task sunning failed with error {e} please consider retry generation.'
                    logger.error(result)
                    messages.append(Message(role='user', content=result))
            else:
                result = f'[Failed] Original query: {messages[1].content} Task sunning failed, code format error, please consider retry generation.'
                logger.error(result)
                messages.append(Message(role='user', content=result))
            runtime.should_stop = True
        else:
            result = f'[Failed] Original query: {messages[1].content} Task sunning failed, code format error, please consider retry generation.'
            logger.error(result)
            messages.append(Message(role='user', content=result))

    async def on_task_end(self, runtime: Runtime, messages: List[Message]):
        if messages[-1].tool_calls:
            return
        if runtime.tag != 'Default workflow':
            if not self.code:
                logger.error(f'Code save failed in task: {runtime.tag}')
