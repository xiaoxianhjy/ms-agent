# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from modelscope_agent.config import Config
from modelscope_agent.workflow.chain_workflow import ChainWorkflow

if __name__ == '__main__':
    query = input('>>>Please input the query:')
    cur_file = __file__
    cur_dir = os.path.dirname(cur_file)
    config = Config.from_task(os.path.join(cur_dir, 'workflow.yaml'))
    engine = ChainWorkflow(config=config)
    asyncio.run(engine.run(query))

