import json
import os
import re
from copy import deepcopy
from typing import List

from modelscope_agent.cli.code.artifact_callback import ArtifactCallback
from modelscope_agent.engine.code.base import Code
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.split_task import SplitTask


class CodeReview(Code):

    _code_review_system = """You are a static code analyst. Your responsibility is to analyze code based on the architect's original architectural design, specific sub-requirement of code tasks, and the actual generated code, to identify and provide feedback on issues within it. The code may be normal or may contain several problems. Please note the following:

        1. If the code works fine, do not propose modification requests, just reply "Code meets requirements"
        2. Pay attention to unavailable features in the code that were not designed by the architect but added extra by subtasks, which may cause problems
        3. Pay attention to whether there are external code references and external resource references in the code that do not follow the actual file paths
        4. Pay attention to whether there are modules in the code that cannot run
        5. Pay attention to parts in the code that do not meet the specific sub-requirements of code tasks
        6. If the code file does not exist, mention that the code need to be regenerated.
        7. Pay attention to whether the code file import structures follow the PRD and architecture design
        8. You must clearly declare what file are you analysing, e.g. `The evaluate result of code file js/a.js: <your result here>\n\n`

        Note: You do not need to fix these errors, just specifically point out the problems.

        Now begin:
        """

    def __init__(self, config):
        super().__init__(config)
        self.llm = LLM.from_config(self.config)

    async def generate_code_review_tool_args(self, messages: List[Message]):
        # The split task tool calling
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
                metadata = ArtifactCallback.extract_metadata(self.config, self.llm,[Message(role='system', content=system),
                                                                 Message(role='user', content=query)])
                metadata = json.loads(metadata)
                output_file = metadata.get('output')
            except Exception:
                code = (f'Cannot fetch the code filename from the prompt of subtask: ```text\n{query}```\n '
                        'the requirement of the sub code task may be error, try to regenerate the architectural design')
            else:
                try:
                    with open(os.path.join('output', output_file), 'r') as f:
                        code = f.read()
                except Exception:
                    code = 'Code file not found or error, need regenerate.'

            review_query = (f'The architect\'s original architectural design is:\n\n```\n{arch_design}```\n\n'
                            f'The sub-requirement of the subtask is:\n\n```\n{query}```\n\n'
                            f'The generated code is:\n\n```\n{code}```\n\n, Now analysis this code:\n')
            task_arg = {
                'system': self._code_review_system,
                'query': review_query,
            }
            sub_tasks.append(task_arg)
        return {'tasks': sub_tasks}

    @staticmethod
    def get_all_files(folder_path):
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, folder_path)
                files.append(relative_path)
        return files

    async def run(self, inputs, **kwargs):
        config = deepcopy(self.config)
        sub_tasks = await self.generate_code_review_tool_args(inputs)
        config.callbacks = []
        split_task = SplitTask(config)
        tool_result = await split_task.call_tool('split_task', tool_name='split_to_sub_task',
                                                 tool_args=sub_tasks)
        all_local_files = '\n'.join(self.get_all_files('output'))
        inputs.append(Message(role='user', content=(f'Here is the code checking result: \n\n{tool_result}\n\n'
                                                      f'Here are the local files exist: \n\n{all_local_files}\n\n'
                                                      'You need to conduct/generate a complete analysis based on the results to '
                                                      'identify which code needs to be corrected. '
                                                      'Then call `split_to_sub_task` again to correct the abnormal code. '
                                                      'But You need to pay attention to mention the subtask to '
                                                      'read the existing code file first, then do a minimum '
                                                      'change to prevent the damages to the functionalities which work normally. '
                                                      'For the file/function missing issue, '
                                                      'you may assign subtasks to generate the missing files/functions if they are essential, or '
                                                      'tell the subtasks to remove the missing ones if they are not needed. '
                                                    'Note: Tell the subtasks to wrap the code with <code></code>, this is mandatory.')))
        return inputs