# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import asyncio
import os

from modelscope_agent.agent.llm_agent import LLMAgent
from modelscope_agent.config import Config
from modelscope_agent.utils import strtobool
from modelscope_agent.workflow.chain_workflow import ChainWorkflow

from modelscope import snapshot_download

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config_dir_or_id',
        required=False,
        type=str,
        default=None,
        help='The directory or the repo id of the config file')
    arg_parser.add_argument(
        '--trust_remote_code',
        required=False,
        type=str,
        default='false',
        help=
        'Trust the code belongs to the config file, set this if you trust the code'
    )
    arg_parser.add_argument(
        '--load_cache',
        required=False,
        type=str,
        default='true',
        help=
        'Load previous step histories from cache, this is useful when a query fails '
        'and retry')
    arg_parser.add_argument(
        '--mcp_server_file',
        required=False,
        type=str,
        default=None,
        help='An extra mcp server file.')
    args, _ = arg_parser.parse_known_args()
    if not args.config_dir_or_id:
        dir_name = os.path.dirname(__file__)
        args.config_dir_or_id = os.path.join(dir_name, 'agent.yaml')
        args.trust_remote_code = 'true'
    if not os.path.exists(args.config_dir_or_id):
        args.config_dir_or_id = snapshot_download(args.config_dir_or_id)
    args.trust_remote_code: bool = strtobool(args.trust_remote_code)  # noqa
    args.load_cache = strtobool(args.load_cache)

    config = Config.from_task(args.config_dir_or_id)
    if Config.is_workflow(config):
        engine = ChainWorkflow(
            config=config,
            trust_remote_code=args.trust_remote_code,
            load_cache=args.load_cache,
            mcp_server_file=args.mcp_server_file)
    else:
        engine = LLMAgent(
            config=config,
            trust_remote_code=args.trust_remote_code,
            load_cache=args.load_cache,
            mcp_server_file=args.mcp_server_file)

    query = ('编写一个名叫魔搭，英文名（ModelScope）的大模型社区网站，要求：'
             '1. 有模型浏览下载能力，并可以增加新模型'
             '2. 有数据集浏览下载能力，并可以增加新数据集'
             '3. 有应用浏览使用能力，并可以增加新的应用'
             '4. 有组织管理能力，公司或个人可以注册组织，并将自己的所有模型数据集放入组织中'
             '5. 有社区能力，用户可以按照板块对大模型进行讨论'
             '网站要求有比较好的科技范'
             '前端所有的增删改查用户注册等都从后端拉取数据，'
             '后端使用node.js，且不需要连接中间件，使用json文件作为数据库即可')  # input('>>>Please input the query:')
    asyncio.run(engine.run(query))
