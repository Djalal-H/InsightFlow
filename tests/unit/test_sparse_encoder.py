"""Tests for the model-free BM25 sparse encoder."""

from __future__ import annotations

import pytest

from insightflow.rag.models import SparseVector
from insightflow.rag.sparse import Bm25SparseEncoder, term_index, tokenize


def test_tokenize_folds_case_and_splits_on_punctuation() -> None:
    assert tokenize("The Río-Grande, CS101!") == ["the", "río", "grande", "cs101"]


def test_term_index_is_stable_and_fits_32_bits() -> None:
    assert term_index("river") == term_index("river")
    assert term_index("river") != term_index("rivers")
    assert 0 <= term_index("river") < 2**32


def test_document_vector_weights_repeated_and_short_text_higher() -> None:
    encoder = Bm25SparseEncoder(average_length=4)

    once = encoder.encode_document("river bank water flows")
    twice = encoder.encode_document("river river water flows")
    long = encoder.encode_document("river " + "filler " * 40)
    river = term_index("river")

    def weight(vector: SparseVector) -> float:
        return vector.values[vector.indices.index(river)]

    assert weight(twice) > weight(once) > weight(long)
    assert once.indices == sorted(once.indices)
    assert len(once.indices) == 4


def test_query_vector_has_unit_weights_and_can_be_empty() -> None:
    encoder = Bm25SparseEncoder()

    query = encoder.encode_query("River, river BANK")

    assert sorted(query.indices) == sorted({term_index("river"), term_index("bank")})
    assert query.values == [1.0, 1.0]
    assert encoder.encode_query("?!").indices == []


def test_fingerprint_changes_with_parameters() -> None:
    assert Bm25SparseEncoder().fingerprint == Bm25SparseEncoder().fingerprint
    assert Bm25SparseEncoder().fingerprint != Bm25SparseEncoder(k1=2.0).fingerprint
    with pytest.raises(ValueError, match="BM25"):
        Bm25SparseEncoder(b=2.0)


def test_sparse_vector_rejects_inconsistent_pairs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        SparseVector(indices=[1], values=[])
    with pytest.raises(ValueError, match="unique"):
        SparseVector(indices=[1, 1], values=[1.0, 1.0])
