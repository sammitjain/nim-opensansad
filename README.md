# nim-opensansad

RAG pipeline for Indian parliamentary data (Lok Sabha Q&A), built on the NVIDIA NIM stack.

## Architecture

```
opensansad/lok-sabha-qa (HuggingFace or local parquet)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  INGEST (local, no API calls)                       │
│                                                     │
│  full_text (pre-extracted markdown)                 │
│    → SentenceSplitter (512 tokens, 64 overlap)      │
│    → e5-large-v2 embedding (local HuggingFace, CPU) │
│    → Milvus standalone (Docker)                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  QUERY                                              │
│                                                     │
│  user query                                         │
│    → e5-large-v2 embedding (local, same model)      │
│    → Milvus ANN search (top_k=10)                   │
│    → NVIDIA NIM reranker (llama-3.2-nv-rerankqa)    │
│    → NVIDIA NIM LLM (llama-3.1-70b-instruct)       │
│    → synthesized answer with sources                │
└─────────────────────────────────────────────────────┘
```

**Embeddings are fully local** — no NVIDIA API calls during ingestion. The NVIDIA API key is only used at query time for reranking and LLM synthesis (2 calls per query, well within the free tier's 40 RPM limit).

## Stack

| Component | Technology | Local / API |
|---|---|---|
| Data source | [opensansad/lok-sabha-qa](https://huggingface.co/datasets/opensansad/lok-sabha-qa) | HuggingFace download |
| Document parsing | Docling + EasyOCR (done upstream in [lok-sabha-dataset](https://github.com/opensansad/lok-sabha-dataset)) | Pre-computed |
| Embeddings | `intfloat/e5-large-v2` via LlamaIndex | Local (CPU) |
| Vector DB | Milvus standalone | Local (Docker) |
| Reranker | NVIDIA NIM `nvidia/llama-3.2-nv-rerankqa-1b-v2` | API |
| LLM | NVIDIA NIM `meta/llama-3.1-70b-instruct` | API |
| Orchestration | LlamaIndex | Local |

## Quick start

```bash
# 1. Clone and install
git clone <repo-url>
cd nim-opensansad
uv sync

# 2. Start Milvus
docker compose up -d

# 3. Configure
cp .env.example .env
# Edit .env — add your NVIDIA_API_KEY from https://build.nvidia.com

# 4. Ingest (downloads from HuggingFace by default)
uv run opensansad ingest --limit 20 --overwrite    # test with 20 docs
uv run opensansad ingest                            # full dataset

# Or from a local parquet:
uv run opensansad ingest --parquet /path/to/lok_sabha_qa.parquet --limit 100

# 5. Build metadata DB (for aggregate stats)
uv run opensansad build-db

# 6. Search
uv run opensansad search "What did the Home Minister say about border security?"

# Search with MP/ministry stats injection
uv run opensansad search "what issues have been raised" --mp "KANGNA RANAUT"
uv run opensansad search "recent questions" --min "TRIBAL AFFAIRS"
uv run opensansad search "education questions" --mp "SHASHI THAROOR" --min "EDUCATION"

# Discover canonical names
uv run opensansad list-mps --search "kangna"
uv run opensansad list-ministries --search "education"

# Collection stats
uv run opensansad stats
```

## System requirements

- Python 3.11+
- Docker (for Milvus)
- ~1.3 GB disk for the embedding model (auto-downloaded on first run)
- NVIDIA API key (free tier at build.nvidia.com)
- No GPU required

## Design decisions

- **Local embeddings over NIM API**: `nv-embedqa-e5-v5` is not downloadable — NVIDIA only serves it via API/NGC containers requiring NVIDIA GPUs. Using `intfloat/e5-large-v2` (same E5 family, 1024-dim) locally removes the rate limit bottleneck for bulk ingestion and keeps the embedding step completely free. MPS (Apple Silicon GPU) is used automatically when available.

- **Pre-parsed data**: Document extraction (OCR, layout detection) is handled upstream by [lok-sabha-dataset](https://github.com/opensansad/lok-sabha-dataset) using a Docling + EasyOCR two-pass pipeline. This project consumes the already-extracted markdown, keeping the codebase focused on retrieval.

- **Milvus standalone (not Lite)**: Full Docker deployment with etcd + minio. Same architecture scales from development to production — switching to Milvus cluster requires no code changes.

- **Aggregate stats via SQLite**: Pure chunk retrieval can't answer analytical queries like "What kind of questions has MP X raised?" A separate SQLite metadata DB provides pre-computed aggregates (by ministry, session, type) that are injected into the LLM context as an evidence packet alongside retrieved chunks. MP names are canonicalised across Lok Sabhas using `mpNo` from supplementary data.

- **Ministry name caveats**: Ministry names are not canonicalised across Lok Sabhas because some ministries were genuinely renamed (e.g. "Human Resource Development" → "Education") while others had their codes reassigned to different ministries. Whitespace is normalised, but renames are kept as-is. The LLM prompt should note that ministries may have historical alternate names.

## Roadmap

- [ ] **Hybrid search + RRF**: Add BM25 sparse retrieval alongside dense vectors, fused with Reciprocal Rank Fusion. Milvus supports this natively. Important for exact matches on MP names, ministry names, bill numbers.

- [ ] **Metadata-filtered vector search**: When `--mp` or `--min` are specified, scope the Milvus vector search to matching metadata (e.g. `members LIKE %THAROOR%`), not just inject stats.

- [ ] **Agentic query decomposition**: Auto-detect when a query needs aggregate stats vs chunk retrieval, replacing the explicit `--mp`/`--min` flags. Small LLM call to extract structured intent from natural language.

- [ ] **RAG evaluation**: Integrate RAGAS or similar framework. The dataset has natural (question, ground_truth_answer) pairs. Metrics: faithfulness, context recall, answer correctness. Stratified test set across ministries and question types.

- [ ] **Scale to 500k+ docs**: Switch Milvus index from HNSW to IVF_PQ for memory efficiency. Export pre-built Milvus snapshots for distribution.

- [ ] **Ministry name canonicalisation**: Curate a manual mapping of known ministry renames across Lok Sabhas (e.g. HRD → Education, Shipping → Ports Shipping and Waterways). Complex because some `minCode` values were reassigned to entirely different ministries.

- [ ] **System prompt**: Add parliamentary domain context to the LLM via `system_prompt`.

- [ ] **Temperature tuning**: Set LLM temperature to 0.1-0.2 for deterministic factual answers.

- [ ] **Frontend**: Web UI with typeahead search for MP/ministry names (populates `--mp`/`--min` filters automatically).
