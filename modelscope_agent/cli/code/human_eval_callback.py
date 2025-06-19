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
        self.human_eval_ended = False

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
        if runtime.tag != 'Default workflow' or messages[-1].tool_calls or messages[-1].role == 'tool':
            return

        query = input('>>> Input feedback, input <OK> to finish:')
        if '<OK>' in query:
            self.human_eval_ended = True
            feedback = 'Everything is fine, task is end.'
        else:
            all_local_files = '\n'.join(self.get_all_files('output'))
            feedback = f"""Here is the feedback from user: 

{query}

Here are the local files exist: 

{all_local_files}

You need to conduct/generate a complete analysis based on the feedback to identify which code needs to be corrected. 
The instructions for your problem checking and fixing:
1. You should first call `split_to_sub_task` to start some subtasks to collect detailed problems from all the related files for you(Each task check only ONE file)

An example of your query:

```
You are a subtask to collect information for me, the user feedback is ..., you need to read the a.js file and check what the problem is, remember you are a evaluator, not a programmer, do not write code, just collect information for me.
```

2. Call `split_to_sub_task` again to correct the abnormal files. But You need to pay attention to mention the subtask to read the existing code file first, then let it do a minimum change to prevent the damages to the functionalities which work normally. Mandatory: Tell the subtasks to wrap the code with <code></code>
"""
        messages.append(Message(role='user', content=feedback))

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        if runtime.tag != 'Default workflow':
            return
        runtime.should_stop = runtime.should_stop and self.human_eval_ended
