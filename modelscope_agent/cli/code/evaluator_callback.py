from typing import List

from omegaconf import DictConfig

from modelscope_agent.callbacks import Callback, Runtime
from modelscope_agent.llm.utils import Message
from modelscope_agent.utils import get_logger

logger = get_logger()


class EvaluatorCallback(Callback):

    _system = """You are a software architecture evaluator whose job is to assess whether software architectures created by other architects are reasonable. The actual workflow is:

1. An original requirement is given
2. A software architect provides the modules that need to be designed and breaks these modules down into different subtasks for completion, with each subtask responsible for writing one specific file
3. After the subtasks are completed, they are automatically saved to disk, and these modules will work together collaboratively

However, software architects have a high probability of making mistakes, including but not limited to:

1. Modules that don't meet user requirements, such as insufficient content richness. In this case, you can try prompting the software architect about whether there are other features that can be added, and you can also provide examples
2. Dependencies between subtasks MUST BE clear. For example, if file1 in subtask1 needs to import and use file2 from subtask2 and file3 from subtask3, you need to CAREFULLY REVIEW whether the DEPENDENCY PROMPTS exist and reasonable. PAY SPECIAL ATTENTION TO THIS REQUEST!!
3. Since files between subtasks work collaboratively, the interfaces between them must be reliable and clear. You need to check whether the interface design provided by the architect is sufficient to support collaborative work requirements
4. Subtasks may use different programming languages or different technology(we don't want to use es6 modules or node.js) or encounter other scenarios where they cannot work together collaboratively. You need to carefully point these out
5. The architect will call `split_task`to start all subtasks at one time, which needs a list of systems and queries. You need to check each subtask's arguments(system and query), whether the information is sufficient for collaborative work requirements.
6. Check whether the architect has mentioned all subtasks the generated files are in one folder, so when importing other files, no dir prefix should be given, and the resources(links, images) should be valid or from the unsplash-like websites, do not use local invalid images.
7. Your reply should be like `You should ...`, `Does you consider...`, or `Here is a problem which...`, at last you should say: `Now correct these problems and keep the good part and generate a new plan and call `split_task` again`
8. Some designs from the architect may be good, point out the good parts to encourage the architect to keep them!
9. Whether the architect use more html than javascript(because we want a beautiful page!)
10. Whether the architect mention what page language to use in subtasks

Your specific job is:
Carefully analyze the errors within, prompt the software architect to make corrections, and when you feel the plan already meets the requirements, output the <OK> character, at which point the conversation will terminate.
Remember: You are not a software architect, you are an evaluator. You don't need to design architecture, you only need to point out or inspire awareness of the errors. 
Now Begin:

"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.argue_ended = False
        self.argue_round = 0

    async def after_generate_response(self, runtime: Runtime, messages: List[Message]):
        if runtime.tag != 'Default workflow':
            self.argue_ended = True
            return

        if len(messages) > 3:
            temp = messages[:2] + messages[-1:]
            messages.clear()
            messages.extend(temp)

        if self.argue_round >= 1:
            self.argue_ended = True
            return

        query = (f'The original requirement is: \n```text\n{messages[1].content}\n```\n\n '
                 f'The plan given by the architect is: \n```text\n{messages[2].content}\n```\n\n '
                 f'The task arguments is : \n```json\n{messages[2].tool_calls[0]}\n```\n\n')

        _messages = [
            Message(role='system', content=self._system),
            Message(role='user', content=query),
        ]
        if getattr(self.config.generation_config, 'stream', False):
            message = None
            for msg in runtime.llm.generate(_messages):
                message = runtime.llm.merge_stream_message(message, msg)

            _response_message = message
        else:
            _response_message = runtime.llm.generate(_messages)
        self.argue_round += 1
        for line in _response_message.content.split('\n'):
            for _line in line.split('\\n'):
                logger.info(f'[Evaluator] {_line}')

        if '<OK>' in _response_message.content or self.argue_ended:
            self.argue_ended = True
        else:
            messages[-1].tool_calls = None
            messages.append(Message(role='user', content=_response_message.content))

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        runtime.should_stop = runtime.should_stop and self.argue_ended