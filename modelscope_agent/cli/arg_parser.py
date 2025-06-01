# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from omegaconf import DictConfig, OmegaConf


def parse_args(config: DictConfig):
    arg_parser = argparse.ArgumentParser()
    args, unknown = arg_parser.parse_known_args()
    if unknown:
        for idx in range(0, len(unknown), 2):
            key = unknown[idx]
            value = unknown[idx + 1]
            assert key.startswith('--'), f'Parameter not correct: {unknown}'
            OmegaConf.update(config, key[2:], value)
