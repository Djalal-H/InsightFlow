"""Typed, strategy-neutral RAG configuration values."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ChunkingStrategy = Literal["fixed_window", "structure_recursive", "semantic_max_min"]
DistanceMetric = Literal["cosine", "dot", "euclid", "manhattan"]
RetrievalMode = Literal["dense", "hybrid"]
QueryExpansionMode = Literal["off", "multi_query", "decompose"]


class RAGConfigModel(BaseModel):
    """Strict immutable base for RAG configuration passed to implementations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(RAGConfigModel):
    """Parameters shared by the Stage 3 chunking strategies."""

    strategy: ChunkingStrategy = "structure_recursive"
    target_tokens: int = Field(default=450, ge=1)
    max_tokens: int = Field(default=600, ge=1)
    min_tokens: int = Field(default=80, ge=1)
    overlap_tokens: int = Field(default=0, ge=0)
    hard_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    similarity_coefficient: float = Field(default=0.9, gt=0.0, le=2.0)
    initialization_constant: float = Field(default=1.5, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_token_limits(self) -> Self:
        """Require a coherent token-size envelope."""
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("token limits must satisfy min_tokens <= target_tokens <= max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.strategy == "semantic_max_min" and self.overlap_tokens != 0:
            raise ValueError("semantic_max_min does not support token overlap")
        return self


class IngestionConfig(RAGConfigModel):
    """Batch controls that do not depend on a parser or storage SDK."""

    embedding_batch_size: int = Field(default=32, ge=1)
    upload_batch_size: int = Field(default=64, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    force: bool = False


class RetrievalConfig(RAGConfigModel):
    """Retrieval, refinement, and context-selection controls.

    The defaults give plain dense retrieval. Each Stage 6 technique is an explicit option, so
    that it can be compared on the evaluation set before it becomes a default.
    """

    mode: RetrievalMode = "dense"
    candidate_top_k: int = Field(default=12, ge=1, le=100)
    generation_max_chunks: int = Field(default=6, ge=1)
    context_token_budget: int = Field(default=6000, ge=1)
    score_threshold: float | None = None
    query_expansion: QueryExpansionMode = "off"
    query_expansion_max_queries: int = Field(default=3, ge=2, le=6)
    rerank: bool = False
    rerank_top_n: int = Field(default=6, ge=1, le=100)
    corrective: bool = False
    neighbors_per_hit: int = Field(default=0, ge=0, le=2)
    compress_context: bool = False


class EmbeddingConfig(RAGConfigModel):
    """Embedding and collection compatibility contract."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    max_input_tokens: int = Field(ge=1)
    distance: DistanceMetric = "cosine"
    normalization: str = "provider_default"
    document_input_type: str | None = None

    @property
    def fingerprint(self) -> str:
        """Return a stable description of vector compatibility.

        The document input type is part of it only when set, so indexes made without one keep
        their fingerprint.
        """
        base = (
            f"{self.provider}:{self.model}:{self.dimensions}:"
            f"{self.distance}:{self.normalization}"
        )
        if self.document_input_type:
            return f"{base}:input_type={self.document_input_type}"
        return base
