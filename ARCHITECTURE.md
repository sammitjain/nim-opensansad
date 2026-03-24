# Architecture

## Overview

nim-opensansad is a RAG (Retrieval-Augmented Generation) pipeline for Indian
parliamentary documents, built on the NVIDIA NIM microservices stack.

## Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL (your machine)                         │
│                                                                     │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────────────────┐ │
│  │ PDF docs │────▶│ unstructured │────▶│ LlamaIndex SentenceSplit│ │
│  │ data/docs│     │ (hi_res OCR) │     │ 512 tok chunks, 64 ovlp│ │
│  └──────────┘     └──────────────┘     └───────────┬─────────────┘ │
│                                                     │               │
│  ┌──────────────────────────────────────────────────┼─────────────┐ │
│  │ Docker                                           │             │ │
│  │  ┌────────────────────────────┐                  │             │ │
│  │  │ Milvus standalone         │◀── store ─────────┘             │ │
│  │  │  • etcd (metadata)        │                                 │ │
│  │  │  • minio (blob storage)   │                                 │ │
│  │  │  • milvus (vector index)  │──── retrieve ──▶ top-K chunks  │ │
│  │  │  port: 19530 (gRPC)       │                                 │ │
│  │  └────────────────────────────┘                                │ │
│  │  ┌────────────────────────────┐                                │ │
│  │  │ Attu UI (optional)        │                                 │ │
│  │  │  port: 8080               │                                 │ │
│  │  └────────────────────────────┘                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │ ▲
                     API calls│ │responses
                              ▼ │
┌─────────────────────────────────────────────────────────────────────┐
│                  NVIDIA API (api.nvidia.com)                        │
│                  Free tier: 40 RPM, no GPU needed                   │
│                                                                     │
│  ┌──────────────────────┐  Called during ingest + search            │
│  │ Embed NIM            │  Model: nvidia/nv-embedqa-e5-v5          │
│  │ 1024-dim vectors     │  Input: text chunk or query              │
│  └──────────────────────┘  Output: 1024-float vector               │
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

## Data flow

### Ingest (`opensansad ingest data/docs/`)

```
PDF file
  → unstructured (local, hi_res OCR via detectron2 + tesseract)
  → raw text with page metadata
  → SentenceSplitter (512 tokens, 64 overlap)
  → chunks (LlamaIndex "nodes")
  → NVIDIAEmbedding NIM (remote API call, returns 1024-dim vector per chunk)
  → MilvusVectorStore (local Docker, stores text + metadata + vector)
```

### Search (`opensansad search "..."`)

```
User query (text)
  → NVIDIAEmbedding NIM (same model as ingest, returns query vector)
  → Milvus ANN search (approximate nearest neighbor, returns TOP_K=10 chunks)
  → NVIDIARerank NIM (cross-encoder rescoring of 10 candidates, keeps top 4)
  → NVIDIA LLM NIM (synthesizes answer from query + 4 best chunks)
  → CLI output (answer + source citations)
```

## Storage

| What              | Where                          | Persisted?                |
|-------------------|--------------------------------|---------------------------|
| Raw PDFs          | `data/docs/`                   | Yes, your files           |
| Vector index      | `volumes/milvus/`              | Yes, Docker volume        |
| Milvus metadata   | `volumes/etcd/`                | Yes, Docker volume        |
| Milvus blobs      | `volumes/minio/`               | Yes, Docker volume        |
| App config        | `.env`                         | Yes, gitignored           |

## Configuration

All config is via environment variables (`.env` file):

| Variable          | Purpose                        | Default                              |
|-------------------|--------------------------------|--------------------------------------|
| `NVIDIA_API_KEY`  | Auth for all NIM API calls     | (required)                           |
| `EMBED_MODEL`     | Embedding model ID             | `nvidia/nv-embedqa-e5-v5`           |
| `LLM_MODEL`       | LLM model ID                   | `meta/llama-3.1-70b-instruct`      |
| `RERANK_MODEL`    | Reranker model ID              | `nvidia/llama-3.2-nv-rerankqa-1b-v2`|
| `MILVUS_URI`      | Milvus connection              | `http://localhost:19530`            |
| `COLLECTION_NAME` | Milvus collection name         | `opensansad`                        |
| `TOP_K`           | Candidates from vector search  | `10`                                |
| `RERANK_TOP_N`    | Chunks kept after reranking    | `4`                                 |

## NIM portability

The same code works against:
- **Hosted NIMs** (current): `api.nvidia.com`, free tier, no GPU
- **On-prem NIMs** (future): change `NVIDIA_BASE_URL` to your NIM container endpoint

This is the core NIM value prop — write once, deploy anywhere.
