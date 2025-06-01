from abc import abstractmethod


class Rag:

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def add_document(self, url: str, content: str, **metadata):
        pass

    @abstractmethod
    def search_documents(self,
                         query: str,
                         limit: int = 5,
                         score_threshold: float=0.7,
                         **filters):
        pass

    @abstractmethod
    def delete_document(self, url: str):
        pass