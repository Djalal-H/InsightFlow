# Max–Min Semantic Chunking

## Goal

Add a provider-neutral semantic chunker that groups consecutive, meaningfully related document
units while preserving source traceability and respecting the configured token-size envelope. A
manual notebook makes the real sentence-embedding and chunking behavior inspectable before the
separate Qdrant ingestion stage is implemented.

## Semantic embeddings in plain language

A semantic embedding converts text into a numeric vector that represents its meaning. Texts with
similar meanings produce vectors that point in similar directions, even when they use different
words.

For example, imagine a simplified embedding with only two dimensions:

```text
"Neural networks learn patterns from data."  -> [0.92, 0.10]
"Training adjusts a model's weights."        -> [0.88, 0.14]
"The Eiffel Tower is located in Paris."      -> [0.05, 0.95]
```

The first two vectors are close, so Max–Min will probably place those sentences in the same
cluster. The third points in a different direction and will probably start a new cluster. Real
embedding vectors have many more dimensions; the two-number vectors above only illustrate the
idea.

InsightFlow uses these embeddings in two distinct stages:

1. **Sentence embeddings now:** temporary vectors used by Max–Min to choose semantic boundaries.
   They are not persisted.
2. **Chunk embeddings later:** one vector for each completed chunk, intended for storage and
   retrieval in Qdrant as part of the ingestion pipeline.

## Implementation overview

```text
NormalizedDocument
  -> sentence and structured-element units
  -> hosted embedding batches
  -> contiguous Max–Min clustering
  -> maximum split / minimum merge pass
  -> deterministic, traceable Chunk objects
```

The algorithm processes units in document order. A new unit joins the current cluster when its
strongest similarity to that cluster clears the dynamic Max–Min threshold; otherwise it begins a
new cluster. Oversized clusters are split at a weak semantic boundary near the target size, while
undersized clusters are merged with the most similar legal neighbor. The maximum token limit is
strict and the minimum is best effort.

## Completed checklist

- [x] Added asynchronous Max–Min clustering backed by the hosted embedding-provider contract.
- [x] Added deterministic English-first sentence segmentation and atomic handling for headings,
  tables, captions, titles, and code.
- [x] Added strict maximum-token enforcement, best-effort minimum merging, stable IDs, neighbor
  links, checksums, headings, pages, and contributing-element metadata.
- [x] Added validated strategy settings and documented the published default thresholds.
- [x] Added a notebook for PDF, TXT, Markdown, and DOCX inspection using real configured sentence
  embeddings without Qdrant writes.
- [x] Verified 78 tests, Ruff checks, strict mypy checks across 37 source files, notebook syntax,
  and `git diff --check`.

## Main file changes

| File | Purpose |
|---|---|
| `src/insightflow/rag/chunkers/max_min.py` | Owns sentence units, embedding validation, Max–Min clustering, bounded split/merge behavior, and traceable chunk construction. |
| `src/insightflow/rag/config.py` | Adds the `semantic_max_min` strategy and validates token and similarity parameters. |
| `src/insightflow/core/config.py` | Exposes the strategy and Max–Min controls through environment-backed application settings. |
| `src/insightflow/providers/embeddings.py` | Adds model-aware token counting and passes configured API key/base options to LiteLLM embedding calls. |
| `src/insightflow/rag/protocols.py` | Makes the chunker contract asynchronous because semantic chunking requires hosted API calls. |
| `notebooks/inspect_max_min_chunking.ipynb` | Parses one supported document, runs real semantic chunking, and displays chunk content and provenance without persistence. |
| `tests/unit/test_max_min_chunker.py` | Covers segmentation, clustering thresholds, batching, token bounds, structured elements, traceability, determinism, and malformed vectors. |
| `README.md` and `.env.example` | Document notebook usage, provider configuration, strategy selection, and tunable defaults. |

## Public contract

| Interface | Behavior |
|---|---|
| `ChunkingStrategy` | Accepts `semantic_max_min`. |
| `Chunker.chunk(...)` | Is asynchronous and returns ordered `Chunk` objects. |
| `EmbeddingProvider.count_tokens(...)` | Counts text with the configured embedding model's tokenizer. |
| `RAG_MAX_MIN_HARD_THRESHOLD` | Defaults to `0.6`; sets the minimum semantic similarity gate. |
| `RAG_MAX_MIN_SIMILARITY_COEFFICIENT` | Defaults to `0.9`; scales the cluster-dependent threshold. |
| `RAG_MAX_MIN_INITIALIZATION_CONSTANT` | Defaults to `1.5`; controls whether the second unit joins a new cluster. |
| Semantic overlap | Must remain `0`; fixed-window overlap is rejected for this strategy. |

## Current boundary

- Sentence embeddings are temporary; final chunk embedding and Qdrant persistence are intentionally
  deferred to the parser-to-Qdrant ingestion task.
- The first implementation uses an English-first rule-based sentence splitter. Manual inspection
  found that some PDF chunks can still begin or end at awkward layout fragments. The underlying
  element boundary and `forced_token_split` metadata should be checked before attributing a case to
  sentence detection.
- A future refinement can introduce an injectable sentence-segmenter protocol, stronger
  offset-preserving segmentation, and conservative PDF fragment repair. These changes are not part
  of the completed implementation.
- Default tests mock hosted providers and do not require credentials or Qdrant. Real chunk quality
  remains corpus- and embedding-model-dependent and is evaluated through the notebook.
