from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from bowi.chunk import DocChunks


@dataclass
class WeightedChunks:
    """
    Attributes
    -----
    doc_chunks
        Original document chunked
    weights:
        An iterable of numpy array.
        weights[i][j] corresponds to the weight of the j-th token
        in the i-th chunk.
    """
    doc_chunks: DocChunks
    weights: Iterable[np.npdarray[int]]


class Weighter(ABCMeta):
    """
    Weight tokens in a document
    """

    @abstractmethod
    def weight(self,
               doc: DocChunks) -> WeightedChunks:
        pass
