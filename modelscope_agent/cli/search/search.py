# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os

from modelscope_agent.config import Config
from modelscope_agent.agent import SimpleLLMAgent

if __name__ == '__main__':
    cur_file = __file__
    cur_dir = os.path.dirname(cur_file)
    config = Config.from_task(os.path.join(cur_dir, 'search.yaml'))
    engine = SimpleLLMAgent(config=config)
    query = input('>>>Please input the query')
    asyncio.run(engine.run(query))
