# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

from omegaconf import DictConfig

from modelscope_agent.callbacks import Callback
from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger

logger = get_logger()


class HumanEvalCallback(Callback):
    """Eval the code by human input
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.human_eval_ended = False
        self.file_system = FileSystemTool(config)
        self.output_dir = getattr(config, 'output_dir', 'output')

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        if (not self.is_default_workflow(runtime)) or messages[-1].tool_calls or messages[-1].role == 'tool':
            # subtask or tool-calling or tool response, skip
            return

        query = input('>>> Input feedback, input <Enter> to finish:')
        if not query:
            self.human_eval_ended = True
            feedback = 'Everything is fine, task is end.'
        else:
            all_local_files = '\n'.join(await self.file_system.list_files())
            feedback = f"""Here is the feedback from user: 

{query}

Here are the local files: 

{all_local_files}

Detect then conduct a complete report to identify which code file needs to be corrected and how to correct them. 
The instructions for problem checking and fixing:
1. First call `split_to_sub_task` to start some subtasks to collect detailed problems from all the related files(Each task check only one file)

An example of your query:

```
You are a subtask to collect information for me, the user feedback is ..., you need to read the ... file and find the root cause, remember you are a evaluator, not a programmer, do not write code, just collect information.
```

2. Call `split_to_sub_task` again to correct the abnormal files. Pay attention to mention the subtask to read the existing code file first, then do a minimum change to prevent the damages to the functionalities which work normally, then output the fixed code in <code></code> block
"""
        messages.append(Message(role='user', content=feedback))

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not self.is_default_workflow(runtime):
            return
        runtime.should_stop = runtime.should_stop and self.human_eval_ended
