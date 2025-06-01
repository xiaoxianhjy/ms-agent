# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from omegaconf import DictConfig

def parse_args(config: DictConfig):
    arg_parser = argparse.ArgumentParser()
    args, unknown = arg_parser.parse_known_args()
    config.merge_with(DictConfig(unknown))
