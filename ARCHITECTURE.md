# Architecture

## Overview

nim-opensansad is a RAG (Retrieval-Augmented Generation) pipeline for Indian
parliamentary documents (Lok Sabha Q&A), built on the NVIDIA NIM microservices stack
with local embeddings and Milvus vector search.

## Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL (your machine)                         │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ HuggingFace      │  │ Sentence-    │  │ pymilvus direct      │  │
│  │ opensansad/      │─▶│ Splitter     │─▶│ batch insert         │  │
│  │ lok-sabha-qa     │  │ (512t, 64ov) │  │ (HNSW index)         │  │
│  └──────────────────┘  └──────┬───────┘  └──────────┬───────────┘  │
│                               │                      │              │
│                        chunks.parquet          embeddings.npy       │
│                               │                      │              │
│                               ▼                      │              │
│                  ┌──────────────────┐                 │              │
│                  │ e5-large-v2      │─────────────────┘              │
│                  │ (local, 1024-dim)│                                │
│                  │ MPS / CUDA / CPU │                                │
│                  └──────────────────┘                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Docker                                                       │   │
│  │  ┌────────────────────────────┐                              │   │
│  │  │ Milvus 2.5 standalone     │                               │   │
│  │  │  • etcd (metadata)        │                               │   │
│  │  │  • minio (blob storage)   │                               │   │
│  │  │  • milvus (vector index)  │──── retrieve ──▶ top-K chunks │   │
│  │  │  port: 19530 (gRPC)       │                               │   │
│  │  └────────────────────────────┘                              │   │
│  │  ┌────────────────────────────┐                              │   │
│  │  │ Attu UI (optional)        │                               │   │
│  │  │  port: 8080               │                               │   │
│  │  └────────────────────────────┘                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │ ▲
                     API calls│ │responses (search-time only)
                              ▼ │
┌─────────────────────────────────────────────────────────────────────┐
│                  NVIDIA API (api.nvidia.com)                        │
│                  Free tier: 40 RPM, no GPU needed                   │
│                                                                     │
│  ┌──────────────────────┐  Called during search only                │
│  │ Rerank NIM           │  Model: nvidia/llama-3.2-nv-rerankqa-1b  │
│  │ cross-encoder        │  Input: (query, chunk) pairs             │
│  └──────────────────────┘  Output: relevance scores (logits)       │
│                                                                     │
│  ┌──────────────────────┐  Called during search only                │
│  │ LLM NIM              │  Model: meta/llama-3.1-70b-instruct     │
│  │ answer synthesis     │  Input: query + top reranked chunks      │
│  └──────────────────────┘  Output: natural language answer         │
└─────────────────────────────────────────────────────────────────────┘
```

**Embeddings are fully local** — the NVIDIA API is only used at search time for
reranking and LLM synthesis (2 calls per query).

## Ingest pipeline

The ingest pipeline is split into three independent, resumable phases. Each phase
produces a local artifact that can be inspected, re-run, or replaced independently.

### Phase 1: Chunk (`opensansad chunk`)

```
HuggingFace dataset (opensansad/lok-sabha-qa)
  → Load full_text (pre-extracted markdown from Docling + EasyOCR upstream)
  → SentenceSplitter (512 tokens, 64 overlap)
  → data/chunks.parquet
      Columns: node_id, doc_id, text, qa_id, lok_no, session_no,
               ques_no, type, date, subject, ministry, members
```

120k+ documents produce ~770k chunks. The parquet file is ~1.5 GB.

### Phase 2: Embed (`opensansad embed`)

```
data/chunks.parquet
  → sentence-transformers (intfloat/e5-large-v2, 1024-dim)
  → data/embeddings.npy (float32, shape: [n_chunks, 1024])
```

**Resumable**: Checks the existing `.npy` row count on startup and skips
already-embedded chunks. Checkpoints to disk every N batches (default 10),
so Ctrl-C is always safe.

**Device options**:
- `--fp16`: Half-precision — ~2x speedup on MPS/CUDA with no quality loss
- `--device cpu|mps|cuda`: Force a specific device (auto-detects by default)
- `--backend onnx`: Use ONNX Runtime (forces CPU; CoreML EP deadlocks on Apple Silicon)

**Benchmarks** (MacBook Air M5, 24 GB RAM, e5-large-v2):

| Device | Precision | Batch size | Throughput |
|--------|-----------|------------|------------|
| CPU    | fp32      | 64         | ~5 chunks/s |
| MPS    | fp32      | 32         | ~10 chunks/s |
| MPS    | fp16      | 32         | ~19 chunks/s |
| ONNX (CPU) | fp32  | 64         | ~4 chunks/s |

MPS + fp16 is the sweet spot for Apple Silicon. Smaller batch sizes (32) actually
outperform larger ones on MPS due to memory transfer overhead — this is the opposite
of what you'd expect on a discrete GPU.

At ~19 chunks/s, embedding the full 770k dataset takes approximately 11 hours.

### Phase 3: Index (`opensansad index`)

```
data/chunks.parquet + data/embeddings.npy
  → pymilvus MilvusClient batch insert (2000 rows/batch)
  → Milvus collection with HNSW index (M=16, efConstruction=256, IP metric)
```

**Resumable**: Queries all existing `qa_id` values in the collection via
`query_iterator` and skips rows that are already indexed. Re-running after an
interruption only inserts the missing chunks.

**Schema** (per chunk):
- `embedding`: FLOAT_VECTOR (1024-dim), HNSW index
- `text`: VARCHAR (chunk content)
- `doc_id`, `qa_id`, `lok_no`, `session_no`, `ques_no`, `type`, `date`,
  `subject`, `ministry`, `members`: VARCHAR metadata fields

Why pymilvus direct insert instead of LlamaIndex's MilvusVectorStore for indexing:
the LlamaIndex path embeds + inserts in a single pass, which is ~8x slower and
not resumable. Decoupling them allows re-embedding with a different model or
re-indexing into a different collection without repeating the expensive embedding step.

## Query pipeline

```
User query + optional --mp/--min flags
  → Resolve MP canonical name to all historical variants (data/mp_aliases.json)
  → Build Milvus filter expression:
      members: LIKE "%name%" with OR across all alias variants
      ministry: exact == match
  → e5-large-v2 embed query locally (same model as ingest)
  → Milvus HNSW ANN search (TOP_K=10 candidates, with metadata filter if set)
  → NVIDIARerank cross-encoder (keeps RERANK_TOP_N=4)
  → NVIDIA LLM synthesis with prompt template containing:
      - retrieved chunk text (context_str)
      - stats evidence packet if --mp/--min given (baked into template)
      - user query (query_str)
  → Answer with cited sources
```

## MP name canonicalisation

MP names vary across Lok Sabhas (spelling changes, transliteration differences).
The HuggingFace dataset includes supplementary `members.json` files with `mpNo`
identifiers that link the same person across sessions.

- **SQLite metadata DB** (`data/metadata.db`): Stores canonicalised MP names.
  Picks the most recent Lok Sabha's spelling as canonical.
- **Milvus chunks**: Store raw (non-canonical) names as they appear in the parquet.
- **Query-time inverse expansion**: When a user queries with a canonical name
  (e.g. "Shri Rahul Gandhi"), the alias map expands it to all historical variants,
  and the Milvus filter ORs across all of them with LIKE.

## Storage

| What              | Where                          | Built by                    |
|-------------------|--------------------------------|-----------------------------|
| Chunk parquet     | `data/chunks.parquet`          | `opensansad chunk`          |
| Embeddings        | `data/embeddings.npy`          | `opensansad embed`          |
| Vector index      | `volumes/milvus/`              | `opensansad index`          |
| Metadata DB       | `data/metadata.db`             | `opensansad build-db`       |
| MP alias map      | `data/mp_aliases.json`         | `opensansad build-aliases`  |
| Milvus metadata   | `volumes/etcd/`                | Docker                      |
| Milvus blobs      | `volumes/minio/`               | Docker                      |
| App config        | `.env`                         | Manual                      |

## Configuration

All config is via environment variables (`.env` file):

| Variable          | Purpose                        | Default                              |
|-------------------|--------------------------------|--------------------------------------|
| `NVIDIA_API_KEY`  | Auth for rerank + LLM API      | (required)                           |
| `EMBED_MODEL`     | Local embedding model          | `intfloat/e5-large-v2`             |
| `EMBED_DIM`       | Embedding dimensions           | `1024`                              |
| `LLM_MODEL`       | LLM model ID                   | `meta/llama-3.1-70b-instruct`      |
| `RERANK_MODEL`    | Reranker model ID              | `nvidia/llama-3.2-nv-rerankqa-1b-v2`|
| `MILVUS_URI`      | Milvus connection              | `http://localhost:19530`            |
| `COLLECTION_NAME` | Milvus collection name         | `opensansad`                        |
| `TOP_K`           | Candidates from vector search  | `10`                                |
| `RERANK_TOP_N`    | Chunks kept after reranking    | `4`                                 |
| `ENABLE_HYBRID`   | Enable BM25 + RRF hybrid search| `false`                             |
| `RRF_K`           | RRF fusion parameter           | `60`                                |

## Hybrid search (experimental, not adopted)

The codebase supports hybrid search (BM25 sparse + dense vectors fused with RRF)
via `ENABLE_HYBRID=true` and the `--hybrid` flag on `opensansad index`. This was
built and evaluated but did not improve retrieval quality for this dataset.

See the [Hybrid Search Experiment](README.md#hybrid-search-experiment) section in
README.md for the full analysis.

## NIM portability

The same code works against:
- **Hosted NIMs** (current): `api.nvidia.com`, free tier, no GPU
- **On-prem NIMs** (future): change `NVIDIA_BASE_URL` to your NIM container endpoint

This is the core NIM value prop — write once, deploy anywhere.
