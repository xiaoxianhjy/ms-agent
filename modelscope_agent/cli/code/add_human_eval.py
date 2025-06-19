# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union, List

from modelscope_agent.agent import Code
from modelscope_agent.llm import Message


class RemoveArchReview(Code):
    """Remove arch review because the architecture design phase has ended.
    """

    def __init__(self, config):
        super().__init__(config)

    async def run(self, inputs: Union[str, List[Message]], **kwargs):
        self.config.callbacks = ['artifact_callback', 'prompt_callback', 'human_eval_callback']
        return inputs