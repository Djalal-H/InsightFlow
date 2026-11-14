"""Lexical sparse vectors for hybrid retrieval, computed without a model.

A document vector holds the BM25 term-frequency weight of each term; Qdrant applies the inverse
document frequency at query time (the collection's sparse vector uses the IDF modifier). Terms
are hashed to stable 32-bit indices, so there is no vocabulary to store or download.
"""

import re
from collections import Counter
from hashlib import blake2b
from unicodedata import normalize

from insightflow.rag.models import SparseVector

SPARSE_ENCODER_NAME = "bm25_hashed"
SPARSE_ENCODER_VERSION = "1.0.0"

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split text into case-folded word tokens."""
    return _TOKEN_RE.findall(normalize("NFKC", text).casefold())


def term_index(term: str) -> int:
    """Map a term to a stable unsigned 32-bit index."""
    return int.from_bytes(blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


class Bm25SparseEncoder:
    """Encode documents with BM25 term weights and queries with unit weights."""

    encoder_name = SPARSE_ENCODER_NAME
    encoder_version = SPARSE_ENCODER_VERSION

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, average_length: float = 256.0) -> None:
        if k1 < 0 or not 0 <= b <= 1 or average_length <= 0:
            raise ValueError("invalid BM25 parameters")
        self._k1 = k1
        self._b = b
        self._average_length = average_length

    @property
    def fingerprint(self) -> str:
        """Identify everything that changes the stored document vectors."""
        return (
            f"{self.encoder_name}:{self.encoder_version}:"
            f"{self._k1}:{self._b}:{self._average_length}"
        )

    def encode_document(self, text: str) -> SparseVector:
        """Weight each term by its saturated, length-normalized frequency."""
        terms = tokenize(text)
        counts = Counter(term_index(term) for term in terms)
        length_norm = self._k1 * (1 - self._b + self._b * len(terms) / self._average_length)
        indices = sorted(counts)
        return SparseVector(
            indices=indices,
            values=[
                counts[index] * (self._k1 + 1) / (counts[index] + length_norm)
                for index in indices
            ],
        )

    def encode_query(self, text: str) -> SparseVector:
        """Give each distinct query term the weight 1; a query without terms is empty."""
        indices = sorted({term_index(term) for term in tokenize(text)})
        return SparseVector(indices=indices, values=[1.0] * len(indices))
