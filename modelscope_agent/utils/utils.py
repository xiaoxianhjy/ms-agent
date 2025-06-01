import importlib
from typing import Optional


def assert_package_exist(package, message: Optional[str] = None):
    message = message or f'Cannot find the pypi package: {package}, please install it by `pip install -U {package}`'
    assert importlib.util.find_spec(package), message
