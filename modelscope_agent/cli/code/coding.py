# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os

from modelscope_agent.config import Config
from modelscope_agent.engine.simple_engine import SimpleEngine

if __name__ == '__main__':
    cur_file = __file__
    cur_dir = os.path.dirname(cur_file)
    config = Config.from_task(os.path.join(cur_dir, 'coding.yaml'))
    engine = SimpleEngine(config=config, trust_remote_code=True)
    query = '写一个唐朝的建立与发展到灭亡的中文网站，要求信息全面，图文并茂' # input('>>>Please input the query')
    asyncio.run(engine.run(query))
