# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'search': 'modelscopeagent.cli.search',
}


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    method_name = argv[0].replace('_', '-')
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    python_cmd = sys.executable
    args = [python_cmd, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
