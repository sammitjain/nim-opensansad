"""
Fast three-phase ingest: chunk → embed → index.

Separates embedding from Milvus insertion for maximum throughput.
Intermediate artifacts are saved locally so phases can be re-run
independently (e.g. re-embed with a different model, or re-index
into a different collection without re-embedding).

  opensansad chunk  [--limit N]                          → data/chunks.parquet
  opensansad embed  [--batch-size 256]                   → data/embeddings.npy
  opensansad index  --collection NAME [--hybrid]         → Milvus
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import config

console = Console()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_CHUNKS_PATH = DATA_DIR / "chunks.parquet"
DEFAULT_EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

META_FIELDS = [
    "lok_no", "session_no", "ques_no", "type",
    "date", "subject", "ministry", "members",
]
DOC_ID_FIELD = "qa_id"


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Phase 1: Chunk ──────────────────────────────────────────────────────────


def chunk(
    parquet_path: Path | None = None,
    limit: int | None = None,
    output: Path = DEFAULT_CHUNKS_PATH,
) -> int:
    """Load docs from HF/parquet, chunk with SentenceSplitter, save parquet."""
    from datasets import load_dataset
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    console.print("[bold]Phase 1: Chunk[/bold]")

    if parquet_path:
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
        console.print(f"  Source: {parquet_path}")
    else:
        ds = load_dataset("opensansad/lok-sabha-qa", split="train")
        console.print("  Source: opensansad/lok-sabha-qa (HuggingFace)")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        console.print(f"  Limit: {limit} documents")

    documents = []
    for row in ds:
        text = row.get("full_text", "")
        if not text or len(text.strip()) < 50:
            continue
        meta = {DOC_ID_FIELD: str(row.get("id", ""))}
        for f in META_FIELDS:
            val = row.get(f)
            if val is None:
                meta[f] = ""
            elif isinstance(val, list):
                meta[f] = ", ".join(str(v) for v in val)
            else:
                meta[f] = str(val)

        all_keys = [DOC_ID_FIELD] + list(META_FIELDS)
        documents.append(Document(
            text=text,
            metadata=meta,
            excluded_embed_metadata_keys=all_keys,
            excluded_llm_metadata_keys=all_keys,
        ))

    console.print(f"  Loaded {len(documents)} documents")

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    console.print(f"  → {len(nodes)} chunks")

    # Serialize to parquet
    columns: dict[str, list] = {
        "node_id": [],
        "doc_id": [],
        "text": [],
    }
    for f in [DOC_ID_FIELD] + list(META_FIELDS):
        columns[f] = []

    for node in nodes:
        columns["node_id"].append(node.node_id)
        columns["doc_id"].append(node.ref_doc_id or "")
        columns["text"].append(node.get_content())
        for f in [DOC_ID_FIELD] + list(META_FIELDS):
            columns[f].append(str(node.metadata.get(f, "")))

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), output)

    size_mb = output.stat().st_size / 1024 / 1024
    console.print(f"  Saved: {output} ({size_mb:.1f} MB)")
    return len(nodes)


# ── Phase 2: Embed ──────────────────────────────────────────────────────────


def embed(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    output: Path = DEFAULT_EMBEDDINGS_PATH,
    batch_size: int = 256,
    checkpoint_every: int = 10,
    device: str | None = None,
    fp16: bool = False,
    backend: str = "torch",
) -> int:
    """Embed all chunks with sentence-transformers directly. Saves .npy.

    Automatically resumes if output already exists — skips already-embedded
    chunks (by row position) and appends new embeddings to the file.

    Checkpoints every `checkpoint_every` batches (default 10 = every 2560
    chunks at batch_size=256), so you can Ctrl-C and resume later.

    Args:
        device: Force device ("cpu", "mps", "cuda"). Auto-detects if None.
        fp16: Use float16 precision (faster on MPS/CUDA, halves memory).
        backend: "torch" (default) or "onnx" (uses ONNX Runtime, often faster).
        checkpoint_every: Save .npy file every N batches. Lower = more frequent
            saves but slightly more I/O overhead.
    """
    from sentence_transformers import SentenceTransformer

    console.print("[bold]Phase 2: Embed[/bold]")

    table = pq.read_table(chunks_path, columns=["text"])
    texts = table.column("text").to_pylist()
    n = len(texts)

    # ── Resume: skip already-embedded chunks ──
    already_done = 0
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        already_done = existing.shape[0]
        del existing  # release mmap immediately
        if already_done >= n:
            console.print(f"  All {n:,} chunks already embedded — nothing to do.")
            return n
        console.print(f"  Resuming: {already_done:,} already done, {n - already_done:,} remaining")
    else:
        console.print(f"  Chunks: {n:,}")

    remaining_texts = texts[already_done:]

    device = device or _get_device()
    console.print(
        f"  Model: {config.EMBED_MODEL} | Device: {device} | "
        f"Batch: {batch_size} | FP16: {fp16} | Backend: {backend} | "
        f"Checkpoint every: {checkpoint_every} batches ({checkpoint_every * batch_size:,} chunks)"
    )

    model_kwargs: dict = {}
    if backend == "onnx":
        model_kwargs["backend"] = "onnx"
        # Force CPUExecutionProvider only — CoreML EP hangs on Apple Silicon
        # due to partial graph support causing split execution deadlocks.
        model_kwargs["model_kwargs"] = {"provider": "CPUExecutionProvider"}
        device = "cpu"

    model = SentenceTransformer(config.EMBED_MODEL, device=device, **model_kwargs)
    if fp16 and device != "cpu" and backend != "onnx":
        model = model.half()

    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    return _embed_with_checkpoints(
        model, remaining_texts, output, already_done, n,
        batch_size, checkpoint_every, start, console,
    )


def _embed_with_checkpoints(
    model,
    remaining_texts: list[str],
    output: Path,
    already_done: int,
    n_total: int,
    batch_size: int,
    checkpoint_every: int,
    start_time: float,
    console: Console,
) -> int:
    """Inner loop: encode in batches, checkpoint to .npy periodically."""
    n_remaining = len(remaining_texts)
    n_batches = (n_remaining + batch_size - 1) // batch_size
    buffer: list[np.ndarray] = []
    batches_since_save = 0
    embedded_this_run = 0

    with Progress(
        SpinnerColumn(), BarColumn(),
        MofNCompleteColumn(), TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Embedding", total=n_remaining)

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, n_remaining)
            batch_texts = [f"passage: {t}" for t in remaining_texts[b_start:b_end]]

            batch_emb = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype(np.float32)

            buffer.append(batch_emb)
            embedded_this_run += len(batch_texts)
            batches_since_save += 1
            prog.update(task, advance=len(batch_texts))

            is_last_batch = (b_idx == n_batches - 1)
            if batches_since_save >= checkpoint_every or is_last_batch:
                _flush_checkpoint(output, buffer)
                total_done = already_done + embedded_this_run
                elapsed = time.perf_counter() - start_time
                rate = embedded_this_run / elapsed if elapsed > 0 else 0
                console.print(
                    f"  💾 Checkpoint: {total_done:,}/{n_total:,} "
                    f"({rate:.0f} chunks/sec)"
                )
                buffer.clear()
                batches_since_save = 0

    elapsed = time.perf_counter() - start_time
    size_mb = output.stat().st_size / 1024 / 1024
    console.print(
        f"  {embedded_this_run:,} new chunks in {elapsed:.1f}s "
        f"({embedded_this_run / elapsed:.0f}/sec)\n"
        f"  Saved: {output} ({size_mb:.0f} MB, "
        f"{already_done + embedded_this_run:,} total)"
    )
    return already_done + embedded_this_run


def _flush_checkpoint(output: Path, buffer: list[np.ndarray]) -> None:
    """Append buffered embeddings to the .npy file on disk.

    Uses atomic write (save to temp file, then rename) so that a Ctrl-C
    during the save can never corrupt the existing .npy.
    """
    new_emb = np.concatenate(buffer)
    if output.exists():
        existing = np.load(output)
        combined = np.concatenate([existing, new_emb])
        del existing
    else:
        combined = new_emb

    tmp = output.with_suffix(".npy.tmp")
    np.save(tmp, combined)
    tmp.rename(output)  # atomic on same filesystem
    del combined, new_emb


# ── Phase 3: Index ──────────────────────────────────────────────────────────


def _get_indexed_qa_ids(client, collection_name: str) -> set[str]:
    """Return all qa_ids already present in a Milvus collection."""
    from pymilvus import Collection, connections

    connections.connect(uri=config.MILVUS_URI)
    collection = Collection(collection_name)
    collection.load()

    all_ids: set[str] = set()
    iterator = collection.query_iterator(
        expr="",
        output_fields=[DOC_ID_FIELD],
        batch_size=10000,
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


def _create_collection(client, collection_name: str, hybrid: bool, embed_dim: int):
    """Create Milvus collection with schema and indexes."""
    from pymilvus import DataType, Function, FunctionType

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=512)

    text_kwargs: dict = {"enable_analyzer": True} if hybrid else {}
    schema.add_field("text", DataType.VARCHAR, max_length=65535, **text_kwargs)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embed_dim)

    meta_fields = [DOC_ID_FIELD] + list(META_FIELDS)
    for f in meta_fields:
        schema.add_field(f, DataType.VARCHAR, max_length=65535)

    if hybrid:
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_embedding"],
        ))

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 256},
    )
    if hybrid:
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    collection_name: str | None = None,
    hybrid: bool = False,
    overwrite: bool = False,
    batch_size: int = 2000,
) -> int:
    """Insert pre-embedded chunks into Milvus. Optionally enable BM25 hybrid.

    Automatically skips chunks whose qa_id is already in the collection, so
    re-running after an interruption only inserts the missing chunks.
    Use --overwrite to drop and rebuild from scratch.
    """
    from pymilvus import MilvusClient

    collection_name = collection_name or config.COLLECTION_NAME
    mode_label = "  [cyan](hybrid: dense + BM25)[/cyan]" if hybrid else ""
    console.print(f"[bold]Phase 3: Index → {collection_name}[/bold]{mode_label}")

    # Load artifacts — only the rows covered by embeddings.npy
    embeddings = np.load(embeddings_path)
    n_embedded = embeddings.shape[0]
    tbl = pq.read_table(chunks_path)
    n_chunks = len(tbl)

    if n_embedded < n_chunks:
        console.print(
            f"  [yellow]Warning: {n_embedded:,} embeddings < {n_chunks:,} chunks. "
            f"Indexing only the embedded portion.[/yellow]"
        )
        tbl = tbl.slice(0, n_embedded)

    n = len(tbl)
    console.print(f"  Loaded {n:,} chunks + embeddings ({embeddings.shape[1]}-dim)")

    client = MilvusClient(uri=config.MILVUS_URI)

    if overwrite and client.has_collection(collection_name):
        client.drop_collection(collection_name)
        console.print(f"  Dropped: {collection_name}")

    # Create collection if it doesn't exist yet
    if not client.has_collection(collection_name):
        _create_collection(client, collection_name, hybrid, embeddings.shape[1])
        console.print("  Collection + indexes created")
        skip_qa_ids: set[str] = set()
    else:
        console.print("  Collection exists — checking for already-indexed chunks...")
        skip_qa_ids = _get_indexed_qa_ids(client, collection_name)
        console.print(f"  Already indexed: {len(skip_qa_ids):,} unique qa_ids — will skip these")

    # ── Batch insert (skipping already-indexed qa_ids) ──
    meta_fields = [DOC_ID_FIELD] + list(META_FIELDS)
    col_doc_id = tbl.column("doc_id").to_pylist()
    col_text = tbl.column("text").to_pylist()
    col_qa_id = tbl.column(DOC_ID_FIELD).to_pylist()
    meta_cols = {f: tbl.column(f).to_pylist() for f in meta_fields}

    # Pre-filter to rows not yet indexed
    pending_indices = [
        i for i in range(n)
        if col_qa_id[i] not in skip_qa_ids
    ]
    n_pending = len(pending_indices)

    if n_pending == 0:
        console.print("  [green]All chunks already indexed — nothing to do.[/green]")
        return 0

    console.print(f"  Inserting {n_pending:,} new chunks...")

    inserted = 0
    with Progress(
        SpinnerColumn(), BarColumn(),
        MofNCompleteColumn(), TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Inserting", total=n_pending)
        for b_start in range(0, n_pending, batch_size):
            b_end = min(b_start + batch_size, n_pending)
            idx_slice = pending_indices[b_start:b_end]

            batch_embeddings = embeddings[idx_slice].tolist()
            rows = [
                {
                    "doc_id": col_doc_id[i],
                    "text": col_text[i],
                    "embedding": batch_embeddings[k],
                    **{f: meta_cols[f][i] for f in meta_fields},
                }
                for k, i in enumerate(idx_slice)
            ]
            client.insert(collection_name, rows)
            inserted += len(rows)
            prog.update(task, advance=len(rows))

    # ── Summary ──
    summary = Table(title="Indexing Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Collection", collection_name)
    summary.add_row("Chunks", f"{inserted:,}")
    summary.add_row("Hybrid (BM25)", "Yes" if hybrid else "No")
    summary.add_row("Dense index", "HNSW (M=16, efConstruction=256)")
    if hybrid:
        summary.add_row("Sparse index", "SPARSE_INVERTED_INDEX (BM25)")
    console.print(summary)

    return inserted
