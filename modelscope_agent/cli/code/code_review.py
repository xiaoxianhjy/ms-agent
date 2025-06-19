# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from copy import deepcopy
from typing import List, Dict

from modelscope_agent.agent import Runtime
from modelscope_agent.agent.code.base import Code
from .artifact_callback import ArtifactCallback
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.tools.split_task import SplitTask


class CodeReview(Code):
    """Review the code automatically!"""

    _code_review_system = """You are a static code analyst. Your responsibility is to analyze code based on the architect's architectural design and the generated code, then identify and provide feedback on issues within it. The code may be normal or contain several problems. Please follow these instructions:

1. If the code works fine, do not propose modification requests, just reply "Code meets requirements"
2. Pay attention to unavailable features in the code that was not designed by the architect but added extra by subtasks
3. Pay attention to whether there are external code and resource references(imports) in the code that do not follow the actual file paths and the architectural design
4. Pay attention to whether there are modules in the code that work abnormally
5. Pay attention to parts in the code that do not meet the specific sub-requirements of code tasks and the architectural design
6. If the code file does not exist, report that the code need to be regenerated
7. Clearly declare which file you are analyzing, e.g. `The evaluate result of code file js/a.js: <your result here>\n\n`

Note: You do not need to fix these errors, just specifically point out the problems.

Now begin:
"""

    def __init__(self, config):
        super().__init__(config)
        self.llm = LLM.from_config(self.config)
        self.file_system = FileSystemTool(config)
        self.output_dir = getattr(config, 'output_dir', 'output')

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def generate_code_review_tool_args(self, messages: List[Message])-> Dict[str, Dict[str, str]]:
        """Manually generate code review tool arguments.

        Args:
            messages: The messages of the architecture.

        Returns:
            The input arguments of the split-task tool.
        """
        arch_design = messages[2].content
        tool_args = messages[2].tool_calls[0]['arguments']
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        tasks = tool_args.get('tasks')
        sub_tasks = []

        for i, task in enumerate(tasks):
            system = task['system']
            query = task['query']
            try:
                metadata = ArtifactCallback.extract_metadata(self.config,
                                                             self.llm,
                                                             [Message(role='system', content=system),
                                                                 Message(role='user', content=query)])
                metadata = json.loads(metadata)
                output_file = metadata.get('output')
            except Exception: # noqa
                # should not happen
                code = (f'Cannot fetch the code filename from the prompt of subtask: ```text\n{query}```\n '
                        'the requirement of the sub code task may be error, try to regenerate the code architecture.')
            else:
                code = self.file_system.read_file(output_file)

            review_query = (f'The architect\'s original architectural design is:\n\n```\n{arch_design}```\n\n'
                            f'The sub-requirement of the subtask is:\n\n```\n{query}```\n\n'
                            f'The generated code is:\n\n```\n{code}```\n\n, Now analyze this code:\n')
            task_arg = {
                'system': self._code_review_system,
                'query': review_query,
            }
            sub_tasks.append(task_arg)
        return {'tasks': sub_tasks}

    async def run(self, inputs, **kwargs):
        """Do a code review task.
        """
        config = deepcopy(self.config)
        sub_tasks = await self.generate_code_review_tool_args(inputs)
        config.callbacks = []
        split_task = SplitTask(config)
        tool_result = await split_task.call_tool('split_task',
                                                 tool_name='split_to_sub_task',
                                                 tool_args=sub_tasks)
        all_local_files = '\n'.join(await self.file_system.list_files())
        query = f"""A coding checking has been done. Here is the code checking result: 

{tool_result}

Here are the local files: 

{all_local_files}

Here is the instructions you need to follow:
1. Conduct/generate a complete report based on the results to identify which code files need to be corrected, and how to correct.
2. Call `split_to_sub_task` again to correct the abnormal code.

You need to mention the subtask to read the existing code file first by a tool-calling, then do a minimum change to prevent the damages to the functionalities which work normally.
For the file/function missing issue, you may assign subtasks to generate, or to remove the obsolete part of the code.
MANDATORY: Mention the subtasks to wrap the code with <code></code>.
"""
        inputs.append(Message(role='user', content=query))
        return inputs