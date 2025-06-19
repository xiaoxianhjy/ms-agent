from abc import abstractmethod
from typing import List

from modelscope_agent.llm import Message


class Rag:

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def add_document(self, url: str, content: str, **metadata):
        pass

    @abstractmethod
    async def search_documents(self,
                         query: str,
                         limit: int = 5,
                         score_threshold: float=0.7,
                         **filters):
        pass

    @abstractmethod
    async def delete_document(self, url: str):
        pass

    @abstractmethod
    async def run(self, inputs: List[Message]) -> List[Message]:
        pass