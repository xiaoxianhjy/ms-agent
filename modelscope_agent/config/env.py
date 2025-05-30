import os.path
from copy import copy
from typing import Dict

from dotenv import load_dotenv


class Env:

    @staticmethod
    def load_env(envs: Dict[str, str]=None):
        load_dotenv()
        envs = copy(os.environ)
        envs.update(envs or {})
        return envs
