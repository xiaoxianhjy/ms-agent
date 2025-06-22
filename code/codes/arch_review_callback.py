# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
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
2. A software architect provides the code architectural design and breaks down into different subtasks for completion
    ```json:tasks.json
    [
      {
        "system": "You are a senior frontend developer. You must follow instructions: ... instructions here ...",
        "query": "Create package.json in root directory with project name 'christmas-ecommerce', create vite.config.ts in ..."
      },
      ... more subtasks here ...
    ]
    ```
    Here is the format of starting tasks to implement the code,architect must output this to distribute tasks.
3. After the subtasks are completed, they are automatically saved to disk, these modules will work together collaboratively

However, software architects have a high probability of making mistakes, here are instructions:

1. The architectural design should meet user requirements, especially the detailed requirements. Normal problems such as insufficient content richness, misunderstanding or insufficient details
2. Dependencies and interface designs between modules MUST BE clear, reliable and sufficient for collaborative work
3. Check the tasks.json:
    * A system field and a query field must exist
    * The system and query contains sufficient information for subtasks to begin coding
    * The output files in the query must matches with the architecture design, especially the folder
    * The system or the query field contains information of the page language
    * Whether all the essential project files are considered
        ** e.g. If it's a react project, files like package.json or vite.config.ts should be considered
4. Some designs from the architect may be good, point out the good parts to encourage the architect to keep them!
5. Your reply should be like `You should ...`, `Did you consider...`, or `Here is a problem which...`, at last you should say: `Now correct these problems and keep the good parts and generate a new PRD & architectual design and regenerate the tasks.json:`

Carefully analyze the errors within, prompt the software architect to make corrections, if the plan already meets the requirements, output the <OK> character.
Remember: You are not a software architect, you are an evaluator. You don't need to design architecture, you only need to point out or inspire awareness of the errors.
Now Begin:

""" # noqa

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.arch_review_ended = False
        self.argue_round = 0

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        await self.do_arch_review(runtime, messages)

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        # When reviewing architecture, tool_calls is None, prevent the loop from ending.
        runtime.should_stop = runtime.should_stop and self.arch_review_ended

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
            f'The plan and tasks given by the architect is: \n```text\n{messages[2].content}\n```\n\n '
        )

        _messages = [
            Message(role='system', content=self._arch_review_system),
            Message(role='user', content=query),
        ]
        # Model chatting
        # if hasattr(self.config, 'generation_config') and getattr(
        #         self.config.generation_config, 'stream', False):
        if False:
            logger.info(f'[ArchReviewer] ')
            _content = ''
            for _response_message in runtime.llm.generate(messages):
                new_content = _response_message.content[len(_content):]
                sys.stdout.write(new_content)
                sys.stdout.flush()
                _content = _response_message.content
        else:
            _response_message = runtime.llm.generate(_messages, stream=False)
            for line in _response_message.content.split('\n'):
                for _line in line.split('\\n'):
                    logger.info(f'[ArchReviewer] {_line}')

        self.argue_round += 1

        if '<OK>' in _response_message.content or self.arch_review_ended:
            self.arch_review_ended = True
        else:
            messages.append(
                Message(role='user', content=_response_message.content))
