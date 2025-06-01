from modelscope_agent.rag.base import Rag
from modelscope_agent.tools.rag_tool import RagTool


class MCPRag(Rag):

    def __init__(self, config):
        super().__init__(config)
        self.rag_tool = RagTool(config)

    def add_document(self, url: str, content: str, **metadata):
        tool_args = {
            'url': url,
            'content': content,
            'metadata': metadata,
        }
        return self.rag_tool.call_tool('ragdocs', tool_name='add_document', tool_args=tool_args)

    def search_documents(self, query: str, limit: int = 5, score_threshold: float = 0.7, **filters):
        tool_args = {
            'query': query,
            'limit': limit,
            'score_threshold': score_threshold,
            'filters': filters,
        }
        return self.rag_tool.call_tool('ragdocs', tool_name='search_documents', tool_args=tool_args)

    def delete_document(self, url: str):
        tool_args = {
            'url': url,
        }
        return self.rag_tool.call_tool('ragdocs', tool_name='delete_document', tool_args=tool_args)