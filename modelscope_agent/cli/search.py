import asyncio
import os

from omegaconf import OmegaConf

from modelscope_agent.engine.simple_engine import SimpleEngine

if __name__ == '__main__':
    cur_file = __file__
    cur_dir = os.path.dirname(cur_file)
    config = OmegaConf.load(os.path.join(cur_dir, 'search.yaml'))
    engine = SimpleEngine(config=config)
    loop = asyncio.get_running_loop()
    query = input('>>>Please input the query')
    loop.run_until_complete(engine.run(query))
