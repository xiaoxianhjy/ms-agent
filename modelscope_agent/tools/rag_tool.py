from modelscope_agent.tools.base import Tool


class RagTool(Tool):

    def __init__(self, config):
        super(RagTool, self).__init__(config)


    async def connect(self):
        pass

    async def get_tools(self):
        pass

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        pass