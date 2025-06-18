import os
from typing import List

from omegaconf import DictConfig

from modelscope_agent.callbacks import Callback
from modelscope_agent.engine.runtime import Runtime
from modelscope_agent.llm.utils import Message
from modelscope_agent.utils import get_logger

logger = get_logger()


class HumanEvalCallback(Callback):

    def __init__(self, config: DictConfig):
        super().__init__(config)

    @staticmethod
    def get_all_files(folder_path):
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, folder_path)
                files.append(relative_path)
        return files

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        if runtime.tag != 'Default workflow' or messages[-1].tool_calls:
            return

        query = input('>>> Input feedback, input <OK> to finish:')
        if '<OK>' in query:
            runtime.should_stop = True
            feedback = 'Everything is fine, task is end.'
        else:
            all_local_files = '\n'.join(self.get_all_files('output'))
            feedback = (f'Here is the feedback from user: \n\n{query}\n\n'
                        f'Here are the local files exist: \n\n{all_local_files}\n\n'
                        'You need to conduct/generate a complete analysis based on the feedback to '
                                                      'identify which code needs to be corrected. '
                        'You may first call `split_to_sub_task` to start some subtasks to collect detailed information from all the related files for you'
                        '(e.g., your prompt: `You are a subtask to collect information for me, the user feedback is ..., you need to read the a.js file and check what the problem is, '
                        'remember you are a evaluator, not a programmer, do not write code, just collect information for me.`), '
                                                      'Then call `split_to_sub_task` again to correct the abnormal code. '
                                                      'But You need to pay attention to mention the subtask to '
                                                      'read the existing code file first, then do a minimum '
                                                      'change to prevent the damages to the functionalities which work normally. '
                                                    'Note: Tell the subtasks to wrap the code with <code></code> and wrap the output file path with <output></output>')
        messages.append(Message(role='user', content=feedback))
