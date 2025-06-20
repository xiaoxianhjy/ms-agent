# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

from modelscope_agent.agent import Runtime
from modelscope_agent.callbacks import Callback
from modelscope_agent.llm import Message
from modelscope_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class ArchReviewCallback(Callback):
    """Do architecture review. This is done with another role.
    """

    _arch_review_system = """You are a software architecture evaluator whose job is to assess whether the software architecture is reasonable. The actual workflow is:

1. An original requirement is given
2. A software architect provides the code architecture and breaks down into different subtasks for completion, each subtask is responsible for writing one code file
3. After the subtasks are completed, they are automatically saved to disk, these modules will work together collaboratively

However, software architects have a high probability of making mistakes, including but not limited to:

1. The architecture does not meet user requirements, especially the detailed requirements. Such as insufficient content richness or misunderstanding
2. Dependencies and interface designs between subtasks MUST BE clear and reliable and sufficient for collaborative work
3. Check the input arguments of `split_to_sub_task`:
    * A system field and a query field must exist
    * The system and query contains sufficient information for subtasks to begin coding
    * The output file path in the query matches with the architecture design, especially the folder
    * The system field contains information of mentioning the subtask do not use invalid media links
    * The system or the query field contains information of the page language
4. Some designs from the architect may be good, point out the good parts to encourage the architect to keep them!
5. Your reply should be like `You should ...`, `Did you consider...`, or `Here is a problem which...`, at last you should say: `Now correct these problems and keep the good parts and generate a new plan and call `split_to_sub_task` again`

Carefully analyze the errors within, prompt the software architect to make corrections, if the plan already meets the requirements, output the <OK> character.
Remember: You are not a software architect, you are an evaluator. You don't need to design architecture, you only need to point out or inspire awareness of the errors.
Now Begin:

""" # noqa

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.arch_review_ended = False
        self.argue_round = 0

    async def do_arch_review(self, runtime: Runtime, messages: List[Message]):
        if not self.arch_review_ended and len(messages) > 3:
            # Dump the previous review rounds, leave only the system, query and the last architect design
            temp = messages[:2] + messages[-1:]
            messages.clear()
            messages.extend(temp)

        if self.argue_round >= 1:
            # Only one round
            self.arch_review_ended = True
            return

        query = (
            f'The original requirement is: \n```text\n{messages[1].content}\n```\n\n '
            f'The plan given by the architect is: \n```text\n{messages[2].content}\n```\n\n '
            f'The task arguments is : \n```json\n{messages[2].tool_calls[0]}\n```\n\n'
        )

        _messages = [
            Message(role='system', content=self._arch_review_system),
            Message(role='user', content=query),
        ]
        # Model chatting
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
            # If something wrong, do no tool-calling, refine the design
            messages[-1].tool_calls = None
            messages.append(
                Message(role='user', content=_response_message.content))

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if not self.is_default_workflow(runtime):
            # Not work in subtasks
            self.arch_review_ended = True
            return

        await self.do_arch_review(runtime, messages)

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not self.is_default_workflow(runtime):
            return
        # When reviewing architecture, tool_calls is None, prevent the loop from ending.
        runtime.should_stop = runtime.should_stop and self.arch_review_ended
