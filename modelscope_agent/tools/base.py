from abc import abstractmethod


class Tool:

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def connect(self):
        pass

    async def cleanup(self):
        pass

    @abstractmethod
    async def get_tools(self):
        pass

    @abstractmethod
    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        pass