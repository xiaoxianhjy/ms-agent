# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import asyncio
import os

from modelscope import snapshot_download

from modelscope_agent.config import Config
from modelscope_agent.utils import strtobool
from modelscope_agent.workflow.chain_workflow import ChainWorkflow
from modelscope_agent.agent.llm_agent import LLMAgent

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_dir_or_id', required=True, type=str,
                            help='The directory or the repo id of the config file')
    arg_parser.add_argument('--trust_remote_code', required=False, type=str, default='false',
                            help='Trust the code belongs to the config file, set this if you trust the code')
    arg_parser.add_argument('--load_cache', required=False, type=str, default='true',
                            help='Load previous step histories from cache, this is useful when a query fails '
                                 'and retry')
    args, _ = arg_parser.parse_known_args()
    if not os.path.exists(args.config_dir_or_id):
        args.config_dir_or_id = snapshot_download(args.config_dir_or_id)
    args.trust_remote_code = strtobool(args.trust_remote_code)
    args.load_cache = strtobool(args.load_cache)

    config = Config.from_task(args.config_dir_or_id)
    if Config.is_workflow(config):
        engine = ChainWorkflow(config=config, trust_remote_code=args.trust_remote_code, load_cache=args.load_cache)
    else:
        engine = LLMAgent(config=config, trust_remote_code=args.trust_remote_code)

    query = 'Design a demo website' # input('>>>Please input the query:')
    asyncio.run(engine.run(query))
