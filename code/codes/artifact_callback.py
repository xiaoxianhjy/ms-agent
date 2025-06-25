# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

from file_parser import extract_code_blocks
from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.callbacks import Callback
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

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            return
        await self.file_system.create_directory()
        content = '\n'.join([m.content for m in messages[2:]])
        all_files, _ = extract_code_blocks(content)
        results = []
        for f in all_files:
            if not f['filename'].startswith('frontend') and not f['filename'].startswith('backend'):
                results.append(f'Error: You should generate files in frontend or backend, '
                               f'but now is: {f["filename"]}')
            else:
                results.append(await
                               self.file_system.write_file(f['filename'],
                                                           f['code']))
        if len(all_files) > 0:
            messages.append(Message(role='user', content='\n'.join(results)))
