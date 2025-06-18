from typing import List

from omegaconf import DictConfig

from modelscope_agent.callbacks import Callback
from modelscope_agent.engine.runtime import Runtime
from modelscope_agent.llm.utils import Message
from modelscope_agent.utils import get_logger

logger = get_logger()


class ArchReviewCallback(Callback):

    _arch_review_system = """You are a software architecture evaluator whose job is to assess whether the software architecture is reasonable. The actual workflow is:

1. An original requirement is given
2. A software architect provides the modules that need to be designed and breaks these modules down into different subtasks for completion, with each subtask responsible for writing one specific file
3. After the subtasks are completed, they are automatically saved to disk, and these modules will work together collaboratively

However, software architects have a high probability of making mistakes, including but not limited to:

1. Modules that don't meet user requirements, such as insufficient content richness. In this case, you can try prompting the software architect about whether there are other features that can be added, and you can also provide examples.Meanwhile, user may give detail requirements, like `a carousel at the bottom`, check carefully if these requirements are fulfilled
2. Dependencies and interface designs between subtasks MUST BE clear and reliable and sufficient for collaborative work. For example, if file1 in subtask1 needs to import and use file2 from subtask2 and file3 from subtask3, you need to CAREFULLY REVIEW whether the DEPENDENCY PROMPTS exist and reasonable
3. Subtasks may use different programming languages or different technology, check whether they are reasonable
4. The architect will call `split_to_sub_task`to start all subtasks at one time, which needs a list of systems and queries. You need to check each subtask's arguments, whether the information is sufficient
6. Check whether the architect has give the correct relative output file path, like `js/a.js`, and the resources(links, images) should be valid or from the unsplash-like websites, not local invalid links should be given
7. Your reply should be like `You should ...`, `Does you consider...`, or `Here is a problem which...`, at last you should say: `Now correct these problems and keep the good part and generate a new plan and call `split_to_sub_task` again`
8. Some designs from the architect may be good, point out the good parts to encourage the architect to keep them!
9. Whether the architect mention what language to show in pages in subtasks

Carefully analyze the errors within, prompt the software architect to make corrections, when you feel the plan already meets the requirements, output the <OK> character, at which point the conversation will terminate.
Remember: You are not a software architect, you are an evaluator. You don't need to design architecture, you only need to point out or inspire awareness of the errors. 
Now Begin:

"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.arch_review_ended = False
        self.argue_round = 0

    async def do_arch_review(self, runtime: Runtime, messages: List[Message]):
        if not self.arch_review_ended and len(messages) > 3:
            temp = messages[:2] + messages[-1:]
            messages.clear()
            messages.extend(temp)

        if self.argue_round >= 1:
            self.arch_review_ended = True
            return

        query = (f'The original requirement is: \n```text\n{messages[1].content}\n```\n\n '
                 f'The plan given by the architect is: \n```text\n{messages[2].content}\n```\n\n '
                 f'The task arguments is : \n```json\n{messages[2].tool_calls[0]}\n```\n\n')

        _messages = [
            Message(role='system', content=self._arch_review_system),
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

        if '<OK>' in _response_message.content or self.arch_review_ended:
            self.arch_review_ended = True
        else:
            messages[-1].tool_calls = None
            messages.append(Message(role='user', content=_response_message.content))


    async def after_generate_response(self, runtime: Runtime, messages: List[Message]):
        if runtime.tag != 'Default workflow':
            self.arch_review_ended = True
            return

        await self.do_arch_review(runtime, messages)

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        runtime.should_stop = runtime.should_stop and self.arch_review_ended
