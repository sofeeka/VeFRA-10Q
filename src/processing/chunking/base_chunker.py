from abc import ABC, abstractmethod

from docling_core.types.doc import DoclingDocument


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: DoclingDocument, user_id: str): ...
