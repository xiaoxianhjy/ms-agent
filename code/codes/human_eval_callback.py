# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from copy import deepcopy
from typing import List
from file_parser import extract_code_blocks
from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.callbacks import Callback
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools import SplitTask
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class HumanEvalCallback(Callback):
    """Eval the code by human input
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.human_eval_ended = False
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_generate_response(self, runtime: Runtime,
                                   messages: List[Message]):
        if messages[-1].tool_calls or messages[-1].role == 'tool':  # noqa
            # subtask or tool-calling or tool response, skip
            return

        query = input('>>> Input feedback, input <Enter> to finish:')
        if not query:
            self.human_eval_ended = True
            feedback = 'Everything is fine, task is end.'
        else:
            all_local_files = await self.file_system.list_files()
            feedback = f"""Here is the feedback from user:

{query}

Here are the local files:

{all_local_files}

Detect then conduct a complete report to identify which code file needs to be corrected and how to correct them.
The instructions for problem checking and fixing:
1. Output `tasks.json` to start some subtasks to collect detailed problems from all the related files

Example:

```json:tasks.json
[
  {{
    "system": "You are a analyzer to collect information for me, remember you are not a programmer, do not write code, just collect information.",
    "query": "The user feedback is ..., you need to read the ... file and find the root cause ..."
  }},
  ... more subtasks here ...
]
```

2. In the next round, you need to output `tasks.json` again to correct the abnormal files. Pay attention to mention the subtask to read the existing code file first, then do a minimum change to prevent the damages to the functionalities which work normally

```json:tasks.json
[
  {{
    "system": "You are a senior frontend developer. You must follow instructions: ... instructions here ...",
    "query": "The user feedback is ..., root cause ..., you need to read the ... file and do a minimun change to fix this issue ... "
  }},
  ... more subtasks here ...
]
```                        
""" # noqa
        messages.append(Message(role='user', content=feedback))

    def generate_checker_prompt(self, messages: List[Message]):
        tasks, _ = extract_code_blocks(messages[-1].content, target_filename='tasks.json')
        _, arch_design = extract_code_blocks(messages[1].content)
        tasks = [t for t in tasks if t['filename'] == 'tasks.json'][0]
        tasks = tasks['code']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)

        sub_tasks = []
        for i, task in enumerate(tasks):
            system = task['system']
            query = task['query']
            checker_system = (f'{system}\n\n'
                             f'The architectural design is {arch_design}\n\n'
                             f'If you have code files to save, output your code with this format:\n\n'
                              f'```js:index.js\n'
                              f'... code ...\n'
                              f'```\n'
                              f'The `index.js` will be used to saving. '
                              f'You only need to check/fix/update the files listed in the query, '
                             f'other modules will be handled in other tasks.\n'
                             f'Now Begin:\n')
            coding_query = query
            task_arg = {
                'system': checker_system,
                'query': coding_query,
            }
            sub_tasks.append(task_arg)
        return {'tasks': sub_tasks}

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if '```json:tasks.json' in messages[-1].content:
            config = deepcopy(self.config)
            split_task = SplitTask(config)
            tasks = self.generate_checker_prompt(messages)
            tool_result = await split_task.call_tool(
                'split_task', tool_name='split_to_sub_task', tool_args=tasks)
            messages.append(Message(role='user', content=tool_result))

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        runtime.should_stop = runtime.should_stop and self.human_eval_ended


