# nim-opensansad

RAG pipeline for Indian parliamentary data (Lok Sabha Q&A), built on the NVIDIA NIM stack.

## Architecture

```
opensansad/lok-sabha-qa (HuggingFace or local parquet)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  INGEST — three-phase pipeline (local, no API calls)     │
│                                                          │
│  Phase 1: CHUNK                                          │
│    full_text → SentenceSplitter (512 tok, 64 overlap)    │
│    → data/chunks.parquet  (770k+ chunks)                 │
│                                                          │
│  Phase 2: EMBED                                          │
│    chunks → e5-large-v2 (local, MPS/CUDA/CPU)            │
│    → data/embeddings.npy  (1024-dim, fp32)               │
│    Supports: --fp16, --device, --backend onnx            │
│    Resumable: checkpoints every N batches                │
│                                                          │
│  Phase 3: INDEX                                          │
│    parquet + .npy → Milvus batch insert (pymilvus)       │
│    Resumable: skips already-indexed qa_ids               │
│    Supports: --hybrid for BM25 sparse vectors            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  QUERY                                                   │
│                                                          │
│  user query + optional --mp/--min filters                │
│    → resolve MP aliases (canonical → all name variants)  │
│    → e5-large-v2 embedding (local, same model)           │
│    → Milvus HNSW ANN search (top_k=10, with filters)     │
│    → NVIDIA NIM reranker (llama-3.2-nv-rerankqa)         │
│    → NVIDIA NIM LLM (llama-3.1-70b-instruct)             │
│    → synthesized answer with cited sources               │
└──────────────────────────────────────────────────────────┘
```

**Embeddings are fully local** — no NVIDIA API calls during ingestion. The NVIDIA API key is only used at query time for reranking and LLM synthesis (2 calls per query, well within the free tier's 40 RPM limit).

The three-phase design separates chunking, embedding, and indexing into independent steps with local artifact checkpoints (`chunks.parquet`, `embeddings.npy`). Each phase is resumable — embedding picks up where it left off, and indexing skips chunks already in Milvus. This matters: embedding 770k chunks takes ~11 hours on a MacBook Air M5 with MPS + fp16.

## Stack

| Component | Technology | Local / API |
|---|---|---|
| Data source | [opensansad/lok-sabha-qa](https://huggingface.co/datasets/opensansad/lok-sabha-qa) | HuggingFace download |
| Document parsing | Docling + EasyOCR (done upstream in [lok-sabha-dataset](https://github.com/opensansad/lok-sabha-dataset)) | Pre-computed |
| Embeddings | `intfloat/e5-large-v2` via sentence-transformers | Local (MPS/CUDA/CPU) |
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

# 4. Ingest — three-phase pipeline
# Phase 1: Chunk documents into 512-token segments
uv run opensansad chunk                             # full dataset from HuggingFace
uv run opensansad chunk --limit 100                 # test with 100 docs

# Phase 2: Embed chunks (resumable — safe to Ctrl-C and restart)
uv run opensansad embed --fp16                      # MPS/CUDA auto-detected
uv run opensansad embed --device cpu --batch-size 64  # force CPU

# Phase 3: Index into Milvus (resumable — skips already-indexed chunks)
uv run opensansad index --collection opensansad

# NOTE: There is also a legacy single-step ingest (opensansad ingest) but it
# is not compatible with the current query pipeline without extra changes.

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

## Evaluation

A metadata-driven retrieval eval lives in `eval/`. It runs queries through the real pipeline (embed → Milvus → filter), skipping only the NIM reranker and LLM, and scores retrieved chunks against expected metadata (MP name, ministry, topic keywords).

Run it:
```bash
uv run opensansad eval           # 18 curated queries, top_k=10
uv run opensansad eval --debug   # also prints per-chunk pass/fail
```

**Results — 2026-04-02 (18 queries, top_k=10, dense search):**

| Mode       | Queries | Avg P@k | Avg Hit@1 | Avg Hit@k | Avg Latency |
|------------|---------|---------|-----------|-----------|-------------|
| unfiltered |      18 |    0.66 |      0.72 |      0.94 |       154ms |
| filtered   |      13 |    1.00 |      1.00 |      1.00 |       150ms |

Key takeaways:
- **Filtered retrieval is perfect (P@k=1.00)** — Milvus `LIKE` filters with MP alias expansion reliably scope results to the right MP/ministry
- **Unfiltered Hit@k=0.94** — semantic search finds at least one relevant chunk in 17/18 queries
- **Unfiltered P@k=0.66** — without filters, ~3 of every 10 chunks are off-target; this is the gap metadata filtering closes
- **Hybrid search (BM25 + RRF) was evaluated but did not improve results** — see [Hybrid Search Experiment](#hybrid-search-experiment) below
- Next step: extend with LLM-as-judge scoring (faithfulness, answer relevance) via RAGAS

## Embedding benchmarks

Embedding 770k+ chunks with `intfloat/e5-large-v2` (1024-dim) on a MacBook Air M5 (24 GB RAM):

| Device | Precision | Batch size | Throughput | Notes |
|--------|-----------|------------|------------|-------|
| CPU    | fp32      | 64         | ~5 chunks/s | Baseline |
| MPS    | fp32      | 32         | ~10 chunks/s | Apple Silicon GPU, diminishing returns above batch 32 |
| MPS    | fp16      | 32         | ~19 chunks/s | `--fp16` flag, best local option |
| ONNX (CPU) | fp32  | 64         | ~4 chunks/s | CoreML EP deadlocks on Apple Silicon; forced CPUExecutionProvider |

MPS + fp16 is the sweet spot for Apple Silicon. The fp16 flag halves the model's memory footprint and nearly doubles throughput with no measurable impact on retrieval quality (eval scores are identical to fp32).

The embed phase checkpoints to disk every 10 batches (~2560 chunks), so it's safe to Ctrl-C and resume later. The index phase queries Milvus for existing `qa_id` values and skips them, so re-running after an interruption only inserts the missing chunks.

## Hybrid search experiment

We built and evaluated BM25 + dense hybrid search with Reciprocal Rank Fusion (RRF) on Milvus 2.5. The hypothesis was that BM25 lexical matching would improve unfiltered retrieval, especially for queries mentioning MP names or specific terms.

**Setup**: Milvus 2.5.10 with `BM25BuiltInFunction` (server-side tokenisation), `SPARSE_INVERTED_INDEX`, RRF k=60, same 770k+ chunks.

**Results (18 queries, top_k=10):**

| Mode | Dense P@k | Hybrid P@k | Delta |
|------|-----------|------------|-------|
| unfiltered | 0.66 | 0.64 | -0.02 |
| filtered | 1.00 | 0.99 | -0.01 |

**Why it didn't help:**

1. **MP-name queries**: MP names appear in the `members` metadata field but only in ~14% of chunk text bodies. BM25 searches the `text` field, so it has almost no signal for name-based queries. Worse, common parliamentary terms ("Lok Sabha", "minister", "issues") have near-zero IDF across 770k chunks, so BM25 returns noisy results that dilute the good dense rankings after RRF fusion.

2. **Topic queries**: BM25 works well here (e.g. 9/10 Aadhaar matches from BM25 alone), but dense search with e5-large-v2 already handles topic queries equally well. The strong embedding model leaves no gap for BM25 to fill.

3. **Hub documents**: One document (`LS16-S11-STARRED-328`, "Fake Appointment in Banks") appeared as a false positive across 5+ queries due to high BM25 scores on common terms.

**Conclusion**: For this dataset — uniform parliamentary language, strong 1024-dim embeddings, and entity-specific queries best served by metadata filters — RRF adds noise rather than signal. Dense search with metadata filtering (P@k=1.00) is the right approach. The hybrid infrastructure (`--hybrid` flag on `opensansad index`, `ENABLE_HYBRID` config) remains available for future experimentation.

## Roadmap

### Completed
- [x] **Metadata-filtered vector search**: `--mp` and `--min` flags scope Milvus ANN search using native filter expressions (`members LIKE %name%`, `ministry == "X"`), with alias expansion across Lok Sabhas via `data/mp_aliases.json`
- [x] **Retrieval evaluation**: 18-query metadata-driven eval (`uv run opensansad eval`) with per-query and summary metrics (P@k, Hit@1, Hit@k, latency)
- [x] **Fast three-phase ingest pipeline**: Separated chunking, embedding, and indexing into independent resumable phases with local artifact checkpoints. Embedding supports MPS/CUDA with fp16 (~2x speedup over fp32 on Apple Silicon)
- [x] **Hybrid search + RRF (evaluated, not adopted)**: Built and tested BM25 sparse retrieval + RRF fusion on Milvus 2.5. Did not improve results — see [Hybrid Search Experiment](#hybrid-search-experiment)
- [x] **Scale to 770k+ chunks**: Full Lok Sabha Q&A dataset indexed with HNSW (M=16, efConstruction=256) on Milvus standalone

### Ahead
- [ ] **LLM-as-judge eval**: Extend current retrieval eval with RAGAS-style faithfulness and answer relevance scoring. Requires curated reference answers for a subset of test queries.

- [ ] **Agentic query decomposition**: Auto-detect when a query needs aggregate stats vs chunk retrieval, replacing the explicit `--mp`/`--min` flags. Small LLM call to extract structured intent (MP name, ministry, topic) from natural language, then apply metadata filters automatically.

- [ ] **Ministry name canonicalisation**: Curate a manual mapping of known ministry renames across Lok Sabhas (e.g. HRD → Education, Shipping → Ports Shipping and Waterways). Complex because some `minCode` values were reassigned to entirely different ministries.

- [ ] **System prompt**: Add parliamentary domain context to the LLM via `system_prompt`.

- [ ] **Temperature tuning**: Set LLM temperature to 0.1-0.2 for deterministic factual answers.

- [ ] **Frontend**: Web UI with typeahead search for MP/ministry names (populates `--mp`/`--min` filters automatically).
