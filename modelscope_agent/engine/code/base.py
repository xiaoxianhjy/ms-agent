from abc import abstractmethod


class Code:

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    async def run(self, inputs, **kwargs):
        pass