# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
from contextlib import contextmanager
from typing import List

from modelscope_agent.agent.runtime import Runtime
from modelscope_agent.callbacks import Callback
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.utils import get_logger
from file_parser import extract_code_blocks
from omegaconf import DictConfig

logger = get_logger()


class EvalCallback(Callback):
    """Eval the code by compiling and human eval.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.feedback_ended = False
        self.file_system = FileSystemTool(config)
        self.compile_round = 20
        self.cur_round = 0

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        self.omit_intermediate_messages(messages)
        await self.file_system.connect()

    @staticmethod
    def omit_intermediate_messages(messages: List[Message]):
        messages[2].tool_calls = None
        tmp = messages[:3]
        messages.clear()
        messages.extend(tmp)

    @contextmanager
    def chdir_context(self):
        path = os.getcwd()
        work_dir = getattr(self.config, 'output_dir', 'output')
        if not path.endswith(work_dir):
            os.chdir(work_dir)
            yield
            os.chdir(path)
        else:
            yield

    def _run_compile(self):
        if self.cur_round >= self.compile_round:
            return ''
        commands = [['npm', 'install'], ['npm', 'run', 'build']]
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                output = e.stdout + '\n' + e.stderr
            else:
                output = result.stdout + '\n' + result.stderr
            if 'failed' not in output.lower() and 'error' not in output.lower(
            ):
                pass
            else:
                self.cur_round += 1
                return output
        return ''

    def get_compile_feedback(self):
        with self.chdir_context():
            return self._run_compile()

    def get_human_feedback(self):
        self.cur_round = 0
        return input('>>> Feedback:')

    async def do_arch_update(self, runtime: Runtime, messages: List[Message], updated_arch: str):
        query = (
            f'The original requirement is: \n```text\n{messages[1].content}\n```\n\n '
            f'The plan and tasks given by the architect is: \n```text\n{messages[2].content}\n```\n\n '
            f'The task arguments is : \n```json\n{messages[2].tool_calls[0] if messages[2].tool_calls else "Tool not called."}\n```\n\n'
        )

        _messages = [
            Message(role='system', content=self._arch_review_system),
            Message(role='user', content=query),
        ]
        # Model chatting
        # if hasattr(self.config, 'generation_config') and getattr(
        #         self.config.generation_config, 'stream', False):
        _response_message = runtime.llm.generate(_messages, stream=False)
        for line in _response_message.content.split('\n'):
            for _line in line.split('\\n'):
                logger.info(f'[Reviewer] {_line}')

    async def on_generate_response(self, runtime: Runtime,
                                   messages: List[Message]):
        if messages[-1].tool_calls or messages[-1].role == 'tool':  # noqa
            # subtask or tool-calling or tool response, skip
            return

        self.omit_intermediate_messages(messages)
        query = None
        if self.config.name == 'agent.yaml':
            # agent.yaml mainly for react
            query = self.get_compile_feedback().strip()
        if not query:
            query = self.get_human_feedback().strip()
        if not query:
            self.feedback_ended = True
            feedback = (
                'You have called `split_to_sub_task` to generate this project, '
                'but call and response of `split_to_sub_task` messages are omitted. '
                'The project runs Ok, you do not need to do any check of fix.')
        else:
            all_local_files = await self.file_system.list_files()
            feedback = f"""Here is a feedback:

{query}

You have called `split_to_sub_task` to generate this project, the call and response of `split_to_sub_task` messages are omitted, the generated files existing on the filesystem are:

{all_local_files}

Detect then conduct a complete report to identify which code file needs to be corrected and how to correct them.
The instructions for problem checking and fixing:
Step 1. First call `split_to_sub_task` to start some subtasks to collect detailed problems from all the related files

An example of your query:

```
You are a subtask to collect information for me, the user feedback is ..., you need to read the ... file and find the root cause, remember you are a evaluator, not a programmer, do not write code, just collect information.
```

Step 2. You may update your architectural design by output:

```text:design.txt
... your modified architectural design here ...
```

* Only update your design when the bug or new feature affect your original design, which is already in your history.
* You only need to output the **changed** parts, and mark clearly how to update.

After output design.txt, call `split_to_sub_task` again to correct the abnormal files or implement the new features.

**You need to remind the subtask do a minimum change in case that the normal code is damaged**

An example of your query:

```
The problem/feature is ..., you need to fix/implement ... file and ... file, read the existing code file first, then do a minimum change to prevent the damages to the functionalities which work normally.
```

After updating, you do not need to verify, the latest feedback will be given to you.
""" # noqa
        messages.append(Message(role='user', content=feedback))

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        design, _ = extract_code_blocks(messages[-1].content, target_filename='design.txt')
        if len(design) > 0:
            front, design = messages[-1].content.split('```text:design.txt', maxsplit=1)
            design, end = design.rsplit('```', 1)
            messages[2].content = design
            messages[-1].content = front + '\n\n' + end

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        runtime.should_stop = runtime.should_stop and self.feedback_ended
