from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass
class DocChunks:
    """
    An chunks

    Attributes
    -----
    chunks
        An iterable of chunks where each of chunk
        has tokens

    """
    chunks: Iterable[Iterable[str]]


class Chunkcer(ABCMeta):
    """

    Chunker that splits a document into mulitiple blocks (chunks).

    """

    @abstractmethod
    def chunk(self, doc: str) -> DocChunks:
        pass
