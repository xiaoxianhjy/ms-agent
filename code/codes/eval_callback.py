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
from omegaconf import DictConfig

logger = get_logger()


class EvalCallback(Callback):
    """Eval the code by human input
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.feedback_ended = False
        self.file_system = FileSystemTool(config)
        self.compile_round = 10
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
1. First call `split_to_sub_task` to start some subtasks to collect detailed problems from all the related files(Each task check only one file)

An example of your query:

```
You are a subtask to collect information for me, the user feedback is ..., you need to read the ... file and find the root cause, remember you are a evaluator, not a programmer, do not write code, just collect information.
```

2. Call `split_to_sub_task` again to correct the abnormal files.

An example of your query:

```
The problem is ..., you need to fix ... file and ... file, read the existing code file first, then do a minimum change to prevent the damages to the functionalities which work normally.
```

After fixing, you do not need to verify, the latest feedback will be given to you.
""" # noqa
        messages.append(Message(role='user', content=feedback))

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        runtime.should_stop = runtime.should_stop and self.feedback_ended
