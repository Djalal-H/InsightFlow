"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    litellm_api_key: str | None = None
    litellm_api_base: str | None = None
    litellm_chat_model: str = ""
    litellm_embedding_model: str = ""
    embedding_provider: str = ""
    embedding_dimensions: int | None = Field(default=None, ge=1)
    embedding_max_input_tokens: int | None = Field(default=None, ge=1)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "insightflow_documents"
    qdrant_vector_name: str = "dense"
    qdrant_distance: Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"
    rag_corpus_dir: Path | None = None
    rag_chunking_strategy: Literal["fixed_window", "structure_recursive"] = (
        "structure_recursive"
    )
    rag_chunk_target_tokens: int = Field(default=450, ge=1)
    rag_chunk_max_tokens: int = Field(default=600, ge=1)
    rag_chunk_min_tokens: int = Field(default=80, ge=1)
    rag_chunk_overlap_tokens: int = Field(default=0, ge=0)
    rag_embedding_batch_size: int = Field(default=32, ge=1)
    rag_upload_batch_size: int = Field(default=64, ge=1)
    rag_candidate_top_k: int = Field(default=12, ge=1, le=100)
    rag_generation_max_chunks: int = Field(default=6, ge=1)
    rag_context_token_budget: int = Field(default=6000, ge=1)

    @model_validator(mode="after")
    def validate_rag_token_limits(self) -> Self:
        """Reject incoherent chunk settings at application startup."""
        if not (
            self.rag_chunk_min_tokens
            <= self.rag_chunk_target_tokens
            <= self.rag_chunk_max_tokens
        ):
            raise ValueError(
                "RAG chunk limits must satisfy minimum <= target <= maximum"
            )
        if self.rag_chunk_overlap_tokens >= self.rag_chunk_max_tokens:
            raise ValueError("RAG chunk overlap must be smaller than the maximum")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one settings instance per process."""
    return Settings()
