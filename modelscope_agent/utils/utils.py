# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import importlib
from typing import Optional


def assert_package_exist(package, message: Optional[str] = None):
    message = message or f'Cannot find the pypi package: {package}, please install it by `pip install -U {package}`'
    assert importlib.util.find_spec(package), message


def strtobool(val):
    val = val.lower()
    if val in {'y', 'yes', 't', 'true', 'on', '1'}:
        return 1
    if val in {'n', 'no', 'f', 'false', 'off', '0'}:
        return 0
    raise ValueError(f'invalid truth value {val!r}')


def str_to_md5(text: str) -> str:
    text_bytes = text.encode('utf-8')
    md5_hash = hashlib.md5(text_bytes)
    return md5_hash.hexdigest()
