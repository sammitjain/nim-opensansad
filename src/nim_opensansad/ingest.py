"""
Ingest pre-parsed Lok Sabha Q&A documents into Milvus.

Data source (one of):
  - HuggingFace: opensansad/lok-sabha-qa (default, downloads automatically)
  - Local parquet file (--parquet flag)

Pipeline:
  full_text (markdown from parquet)
    → SentenceSplitter (chunk into ~512-token nodes)
    → HuggingFaceEmbedding (intfloat/e5-large-v2, local MPS/CPU)
    → MilvusVectorStore (full standalone, Docker)

Ingestion is batched: documents are processed in groups of INGEST_BATCH_SIZE
to keep memory bounded. Each batch is chunked, embedded, and inserted into
Milvus before the next batch is loaded.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from pymilvus import MilvusClient
from rich.console import Console
from rich.table import Table
from datasets import load_dataset
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from . import config

console = Console()

HF_DATASET = "opensansad/lok-sabha-qa"

# Batching: process this many documents at a time to keep memory bounded.
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "200"))

def _get_device() -> str:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Metadata fields to carry through to Milvus (for future filtered search).
# Note: the dataset's "id" field is stored as "qa_id" in Milvus to avoid
# clashing with Milvus's auto-generated primary key field named "id".
META_FIELDS = [
    "lok_no", "session_no", "ques_no", "type",
    "date", "subject", "ministry", "members",
]
DOC_ID_FIELD = "qa_id"  # mapped from dataset's "id" column


def _build_vector_store(overwrite: bool = False) -> MilvusVectorStore:
    return MilvusVectorStore(
        uri=config.MILVUS_URI,
        collection_name=config.COLLECTION_NAME,
        dim=config.EMBED_DIM,
        overwrite=overwrite,
    )


def _get_indexed_doc_ids() -> set[str]:
    """Query Milvus for all distinct document IDs already indexed using the query iterator."""
    from pymilvus import connections, Collection

    connections.connect(uri=config.MILVUS_URI)
    if not client_has_collection():
        return set()

    collection = Collection(config.COLLECTION_NAME)
    collection.load()

    all_ids: set[str] = set()
    iterator = collection.query_iterator(
        expr="",
        output_fields=[DOC_ID_FIELD],
        batch_size=5000,
    )
    while True:
        batch = iterator.next()
        if not batch:
            break
        for r in batch:
            if DOC_ID_FIELD in r:
                all_ids.add(r[DOC_ID_FIELD])

    iterator.close()
    return all_ids


def client_has_collection() -> bool:
    client = MilvusClient(uri=config.MILVUS_URI)
    return client.has_collection(config.COLLECTION_NAME)


def _load_documents(
    parquet_path: Path | None = None,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
) -> list[Document]:
    """
    Load documents from HuggingFace dataset or a local parquet file.

    Each row becomes a LlamaIndex Document with:
      - text = full_text (pre-extracted markdown)
      - metadata = structured fields (ministry, date, subject, etc.)

    Metadata is excluded from embed text and LLM context —
    embeddings focus purely on content, metadata is stored as
    scalar fields in Milvus for filtered search.

    Documents whose id is in skip_ids are skipped (already indexed).
    """
    if parquet_path:
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    else:
        ds = load_dataset(HF_DATASET, split="train")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    skip_ids = skip_ids or set()
    documents = []
    skipped = 0
    PREVIEW_COUNT = 5

    for row in ds:
        doc_id = row.get("id", "")  # e.g. "LS16-S14-STARRED-573"

        if doc_id in skip_ids:
            skipped += 1
            continue

        text = row.get("full_text", "")
        if not text or len(text.strip()) < 50:
            continue

        metadata = {DOC_ID_FIELD: doc_id}
        for field in META_FIELDS:
            val = row.get(field)
            if val is not None:
                # Milvus scalar fields need simple types
                if isinstance(val, list):
                    metadata[field] = ", ".join(str(v) for v in val)
                else:
                    metadata[field] = val

        all_meta_keys = [DOC_ID_FIELD] + list(META_FIELDS)
        doc = Document(
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=all_meta_keys,
            excluded_llm_metadata_keys=all_meta_keys,
        )
        documents.append(doc)

        # Print first 5 docs, then summarize the rest
        idx = len(documents)
        if idx <= PREVIEW_COUNT:
            subject = metadata.get("subject", "?")
            ministry = metadata.get("ministry", "?")
            console.print(
                f"  [dim]loaded[/dim] {doc_id}  "
                f"[cyan]{ministry}[/cyan] — {subject}"
            )

    if len(documents) > PREVIEW_COUNT:
        console.print(f"  [dim]...and {len(documents) - PREVIEW_COUNT} more documents[/dim]")

    if skipped:
        console.print(f"  [yellow]skipped {skipped} already-indexed documents[/yellow]")

    return documents


def ingest(
    parquet_path: Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """
    Load documents, chunk, embed, and store in Milvus.

    Documents are processed in batches of INGEST_BATCH_SIZE to keep memory
    bounded. Each batch goes through: load → chunk → embed (MPS/CPU) → Milvus insert.

    Args:
        parquet_path: Path to a local parquet file. If None, downloads from HF.
        limit:        Only ingest the first N documents (useful for testing).
        overwrite:    Drop and recreate the Milvus collection before ingesting.

    Returns:
        Number of nodes (chunks) indexed.
    """
    # --- Dedup: find what's already indexed ---
    if not overwrite:
        indexed_ids = _get_indexed_doc_ids()
        if indexed_ids:
            console.print(f"[dim]Found {len(indexed_ids)} documents already indexed in Milvus.[/dim]")
    else:
        indexed_ids = set()

    documents = _load_documents(parquet_path, limit, skip_ids=indexed_ids)
    if not documents:
        console.print("[green]Nothing new to index — all documents already in Milvus.[/green]")
        return 0

    # --- Set up shared resources (created once, reused across batches) ---
    device = _get_device()
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL,
        device=device,
        query_instruction="query: ",
        text_instruction="passage: ",
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    # Only overwrite on the very first batch
    vector_store = _build_vector_store(overwrite=overwrite)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    console.print(f"[dim]Device: {device} | Batch size: {INGEST_BATCH_SIZE} docs[/dim]")

    total_docs = len(documents)
    total_chunks = 0
    num_batches = (total_docs + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * INGEST_BATCH_SIZE
        end = min(start + INGEST_BATCH_SIZE, total_docs)
        batch_docs = documents[start:end]

        # Chunk this batch
        nodes = splitter.get_nodes_from_documents(batch_docs)

        console.print(
            f"\n[bold]Batch {batch_idx + 1}/{num_batches}[/bold]: "
            f"{len(batch_docs)} docs → {len(nodes)} chunks"
        )

        # Embed + store
        VectorStoreIndex(
            nodes,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=True,
        )

        total_chunks += len(nodes)

        # After first batch, don't overwrite on subsequent inserts.
        # VectorStoreIndex reuses the storage_ctx which already has
        # overwrite=False after initial creation, so this is safe.

        # Free batch memory
        del batch_docs, nodes
        gc.collect()

    # --- Summary ---
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Documents ingested", str(total_docs))
    table.add_row("Chunks indexed", str(total_chunks))
    table.add_row("Batches", str(num_batches))
    table.add_row("Device", device)
    table.add_row("Embed model", config.EMBED_MODEL)
    table.add_row("Collection", config.COLLECTION_NAME)
    table.add_row("Milvus", config.MILVUS_URI)
    console.print(table)

    return total_chunks
