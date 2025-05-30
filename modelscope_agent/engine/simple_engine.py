from modelscope_agent.config.config import Config
from modelscope_agent.llm.llm import LLM


class SimpleEngine:

    def __init__(self, task_dir_or_id=None, env=None, **kwargs):
        self.config = Config.from_task(task_dir_or_id, env)
        self.llm = LLM.from_config(self.config)

    def run(self, prompt, **kwargs):

        while True:

