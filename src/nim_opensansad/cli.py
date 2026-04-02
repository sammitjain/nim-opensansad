"""
CLI entry point.

Commands:
  opensansad ingest                         — download from HF + index into Milvus
  opensansad ingest --parquet path.parquet  — use local parquet instead
  opensansad search "<query>"               — retrieve, rerank, and synthesize an answer
  opensansad search "<query>" --mp "NAME"   — include MP stats in LLM context
  opensansad search "<query>" --min "NAME"  — include ministry stats in LLM context
  opensansad build-db                       — build SQLite metadata DB from HF dataset
  opensansad list-mps --search "term"       — search canonical MP names
  opensansad list-ministries --search "term" — search ministry names
  opensansad stats                          — show collection size and chunk count
  opensansad eval                           — run retrieval eval against test set
"""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="opensansad", add_completion=False)
console = Console()


@app.command()
def ingest(
    parquet: Optional[Path] = typer.Option(None, "--parquet", help="Local parquet file. If omitted, downloads from HuggingFace."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only ingest first N documents (useful for testing)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Drop existing collection and re-index from scratch."),
):
    """Load documents, chunk, embed, and store into Milvus via NVIDIA NIM."""
    from .ingest import ingest as _ingest

    if parquet:
        if not parquet.exists():
            console.print(f"[red]Parquet file not found: {parquet}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Ingesting from local parquet:[/bold] {parquet}")
    else:
        console.print("[bold]Ingesting from HuggingFace:[/bold] opensansad/lok-sabha-qa")

    if limit:
        console.print(f"[dim]Limiting to first {limit} documents.[/dim]")
    if overwrite:
        console.print("[yellow]--overwrite set: existing collection will be dropped.[/yellow]")

    n = _ingest(parquet_path=parquet, limit=limit, overwrite=overwrite)
    console.print(f"[green]Done.[/green] Indexed [bold]{n}[/bold] chunks into Milvus.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query"),
    mp: Optional[str] = typer.Option(None, "--mp", help="MP name — injects aggregate stats into LLM context"),
    min_: Optional[str] = typer.Option(None, "--min", help="Ministry name — injects aggregate stats into LLM context"),
    top_q: int = typer.Option(10, "--top-q", help="Number of recent questions to include in stats"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Print source chunks after the answer"),
):
    """Retrieve, rerank, and synthesize an answer from indexed documents."""
    from .query import build_query_engine
    from .stats import build_evidence_packet
    from .metadata import DEFAULT_DB_PATH, DEFAULT_ALIAS_PATH, load_alias_map

    console.print(f"\n[bold]Query:[/bold] {query}")
    if mp:
        console.print(f"[bold]MP filter:[/bold] {mp}")
    if min_:
        console.print(f"[bold]Ministry filter:[/bold] {min_}")
    console.print()

    # Resolve MP aliases for Milvus filtering
    mp_aliases: list[str] | None = None
    if mp:
        if DEFAULT_ALIAS_PATH.exists():
            alias_map = load_alias_map(DEFAULT_ALIAS_PATH)
            mp_aliases = alias_map.get(mp)
            if mp_aliases and len(mp_aliases) > 1:
                console.print(f"[dim]Alias expansion: {', '.join(mp_aliases)}[/dim]")
        else:
            console.print("[dim]No alias map found — filtering with canonical name only. "
                          "Run 'opensansad build-aliases' for better recall.[/dim]")

    # Build stats evidence packet if filters are specified
    evidence = None
    if mp or min_:
        if not DEFAULT_DB_PATH.exists():
            console.print("[red]Metadata DB not found. Run 'opensansad build-db' first.[/red]")
            raise typer.Exit(1)

        evidence = build_evidence_packet(
            mp_name=mp, ministry=min_, top_q=top_q
        )
        if evidence:
            console.print(Panel(evidence, title="Stats Evidence Packet", border_style="cyan"))
        else:
            console.print("[yellow]No stats found for the specified filters.[/yellow]")

    with console.status("Querying NVIDIA NIMs..."):
        engine = build_query_engine(mp=mp, ministry=min_, evidence=evidence, mp_aliases=mp_aliases)
        response = engine.query(query)

    console.print(Panel(Markdown(str(response)), title="Answer", border_style="green"))

    if sources and response.source_nodes:
        console.print("\n[bold]Sources:[/bold]")
        for i, node in enumerate(response.source_nodes, 1):
            subject = node.metadata.get("subject", "unknown")
            ministry = node.metadata.get("ministry", "")
            doc_id = node.metadata.get("qa_id", "")
            score = f"{node.score:.3f}" if node.score is not None else "n/a"
            console.print(f"  [{i}] {doc_id}  (score: {score})")
            console.print(f"      [dim]{ministry} — {subject}[/dim]")
            console.print(f"      {node.text[:200].strip()}...\n")


@app.command(name="build-db")
def build_db(
    parquet: Optional[Path] = typer.Option(None, "--parquet", help="Local parquet file. If omitted, downloads from HuggingFace."),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Output SQLite path (default: data/metadata.db)."),
):
    """Build the SQLite metadata database from the HuggingFace dataset."""
    from .metadata import build_metadata_db, DEFAULT_DB_PATH

    target = db_path or DEFAULT_DB_PATH
    build_metadata_db(parquet_path=parquet, db_path=target)


@app.command(name="build-aliases")
def build_aliases(
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output JSON path (default: data/mp_aliases.json)."),
):
    """Build the MP alias map (canonical name → all name variants) from HuggingFace supplementary data."""
    from .metadata import save_alias_map, DEFAULT_ALIAS_PATH

    target = out_path or DEFAULT_ALIAS_PATH
    save_alias_map(out_path=target)


@app.command(name="list-mps")
def list_mps(
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter MP names (case-insensitive substring match)"),
):
    """List canonical MP names. Use --search to filter."""
    from .metadata import get_all_canonical_mp_names

    with console.status("Loading MP names from HuggingFace..."):
        names = get_all_canonical_mp_names()

    if search:
        term = search.upper()
        names = [n for n in names if term in n.upper()]

    if not names:
        console.print("[yellow]No matching MPs found.[/yellow]")
        return

    console.print(f"[bold]{len(names)} MPs found:[/bold]\n")
    for name in names:
        console.print(f"  {name}")


@app.command(name="list-ministries")
def list_ministries(
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter ministry names (case-insensitive substring match)"),
):
    """List all ministry names. Use --search to filter."""
    from .metadata import get_all_ministry_names

    with console.status("Loading ministry names from HuggingFace..."):
        names = get_all_ministry_names()

    if search:
        term = search.upper()
        names = [n for n in names if term in n.upper()]

    if not names:
        console.print("[yellow]No matching ministries found.[/yellow]")
        return

    console.print(f"[bold]{len(names)} ministries found:[/bold]\n")
    for name in names:
        console.print(f"  {name}")


@app.command()
def stats(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name (default: COLLECTION_NAME env var)."),
):
    """Show collection stats: chunk count, data size, and all collections."""
    from pymilvus import MilvusClient, connections, Collection
    from . import config

    connections.connect(uri=config.MILVUS_URI)
    client = MilvusClient(uri=config.MILVUS_URI)

    all_collections = client.list_collections()
    if not all_collections:
        console.print("[red]No collections found in Milvus.[/red]")
        raise typer.Exit(1)

    target = collection or config.COLLECTION_NAME
    if target not in all_collections:
        console.print(f"[red]Collection '{target}' not found.[/red]")
        console.print(f"Available: {', '.join(all_collections)}")
        raise typer.Exit(1)

    def fmt_size(kb: int) -> str:
        if kb >= 1_048_576:
            return f"{kb / 1_048_576:.1f} GB"
        elif kb >= 1024:
            return f"{kb / 1024:.1f} MB"
        return f"{kb} KB"

    def estimate_storage(num_rows: int, dim: int, has_sparse: bool) -> str:
        """Estimate on-disk storage from entity count + vector dimensions.

        Dense vectors (raw + HNSW index ≈ 2.5×):  num_rows × dim × 4B × 2.5
        Sparse BM25 vectors (≈ 300B avg per row):  num_rows × 300B
        Metadata + overhead:                        num_rows × 500B
        """
        dense_bytes = num_rows * dim * 4 * 2.5
        sparse_bytes = num_rows * 300 if has_sparse else 0
        meta_bytes = num_rows * 500
        total = int(dense_bytes + sparse_bytes + meta_bytes)
        return fmt_size(total // 1024) + " [dim](est.)[/dim]"

    # ── Per-collection stats ──
    col = Collection(target)
    col.flush()
    num_chunks = col.num_entities

    # Detect sparse field to know if hybrid
    schema = col.schema
    has_sparse = any(
        int(f.dtype) == 104
        for f in schema.fields
    )

    table = Table(title=f"Collection: {target}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Chunks", f"{num_chunks:,}")
    table.add_row("Hybrid (BM25)", "Yes" if has_sparse else "No")
    table.add_row("Est. storage", estimate_storage(num_chunks, config.EMBED_DIM, has_sparse))

    # ── All collections summary ──
    table.add_row("", "")
    table.add_row("[dim]All collections[/dim]", "")
    for cname in sorted(all_collections):
        try:
            c = Collection(cname)
            c.flush()
            c_sparse = any(
                int(f.dtype) == 104
                for f in c.schema.fields
            )
            marker = " ◀" if cname == target else ""
            est = estimate_storage(c.num_entities, config.EMBED_DIM, c_sparse)
            table.add_row(
                f"  {cname}{marker}",
                f"{c.num_entities:,} chunks  {est}",
            )
        except Exception:
            table.add_row(f"  {cname}", "[dim]unavailable[/dim]")

    # ── Shared volume disk usage ──
    volumes_dir = Path(__file__).parent.parent.parent / "volumes"
    table.add_row("", "")
    table.add_row("[dim]Shared volume disk (all collections)[/dim]", "")
    total_kb = 0
    for name in ("milvus", "minio", "etcd"):
        p = volumes_dir / name
        if p.exists():
            result = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True)
            kb = int(result.stdout.split()[0])
            total_kb += kb
            table.add_row(f"  volumes/{name}/", fmt_size(kb))
    table.add_row("  [bold]Total[/bold]", f"[bold]{fmt_size(total_kb)}[/bold]")

    console.print(table)


# ── Fast ingest pipeline (chunk → embed → index) ─────────────────────────


@app.command()
def chunk(
    parquet: Optional[Path] = typer.Option(None, "--parquet", help="Local parquet file. If omitted, downloads from HuggingFace."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only process first N documents."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output parquet path (default: data/chunks.parquet)."),
):
    """Phase 1: Load documents, chunk, and save as parquet."""
    from .ingest_v2 import chunk as _chunk, DEFAULT_CHUNKS_PATH

    target = output or DEFAULT_CHUNKS_PATH
    n = _chunk(parquet_path=parquet, limit=limit, output=target)
    console.print(f"[green]Done.[/green] {n:,} chunks saved to {target}")


@app.command()
def embed(
    chunks: Optional[Path] = typer.Option(None, "--chunks", help="Input chunks parquet (default: data/chunks.parquet)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output .npy path (default: data/embeddings.npy)."),
    batch_size: int = typer.Option(256, "--batch-size", "-b", help="Encoding batch size (higher = faster, more VRAM)."),
    checkpoint_every: int = typer.Option(10, "--checkpoint-every", "-cp", help="Save .npy every N batches (lower = more frequent saves, safer to Ctrl-C)."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Force device: cpu, mps, or cuda (auto-detects if omitted)."),
    fp16: bool = typer.Option(False, "--fp16", help="Use float16 precision (faster on MPS/CUDA, halves memory)."),
    backend: str = typer.Option("torch", "--backend", help="torch (default) or onnx (uses ONNX Runtime)."),
):
    """Phase 2: Embed chunks with sentence-transformers and save as .npy."""
    from .ingest_v2 import embed as _embed, DEFAULT_CHUNKS_PATH, DEFAULT_EMBEDDINGS_PATH

    src = chunks or DEFAULT_CHUNKS_PATH
    dst = output or DEFAULT_EMBEDDINGS_PATH
    if not src.exists():
        console.print(f"[red]Chunks file not found: {src}[/red]\nRun 'opensansad chunk' first.")
        raise typer.Exit(1)

    n = _embed(chunks_path=src, output=dst, batch_size=batch_size, checkpoint_every=checkpoint_every, device=device, fp16=fp16, backend=backend)
    console.print(f"[green]Done.[/green] {n:,} embeddings saved to {dst}")


@app.command()
def index(
    chunks: Optional[Path] = typer.Option(None, "--chunks", help="Input chunks parquet (default: data/chunks.parquet)."),
    embeddings: Optional[Path] = typer.Option(None, "--embeddings", help="Input .npy embeddings (default: data/embeddings.npy)."),
    collection: str = typer.Option(..., "--collection", "-c", help="Milvus collection name (e.g. opensansad_hybrid)."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable BM25 sparse vectors for hybrid search."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Drop existing collection first."),
    batch_size: int = typer.Option(2000, "--batch-size", "-b", help="Milvus insert batch size."),
):
    """Phase 3: Insert pre-embedded chunks into Milvus."""
    from .ingest_v2 import index as _index, DEFAULT_CHUNKS_PATH, DEFAULT_EMBEDDINGS_PATH

    src_chunks = chunks or DEFAULT_CHUNKS_PATH
    src_embeds = embeddings or DEFAULT_EMBEDDINGS_PATH

    if not src_chunks.exists():
        console.print(f"[red]Chunks file not found: {src_chunks}[/red]\nRun 'opensansad chunk' first.")
        raise typer.Exit(1)
    if not src_embeds.exists():
        console.print(f"[red]Embeddings file not found: {src_embeds}[/red]\nRun 'opensansad embed' first.")
        raise typer.Exit(1)

    n = _index(
        chunks_path=src_chunks,
        embeddings_path=src_embeds,
        collection_name=collection,
        hybrid=hybrid,
        overwrite=overwrite,
        batch_size=batch_size,
    )
    console.print(f"[green]Done.[/green] {n:,} chunks indexed into [bold]{collection}[/bold]")


@app.command(name="eval")
def eval_cmd(
    test_set: Optional[Path] = typer.Option(None, "--test-set", help="Path to test set JSON (default: eval/test_set.json)."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override TOP_K for retrieval."),
    debug: bool = typer.Option(False, "--debug", help="Dump chunk metadata for queries with P@k < 1.0."),
):
    """Run retrieval evaluation against a curated test set (no LLM calls)."""
    from .eval import run_eval, print_report, DEFAULT_TEST_SET

    target = test_set or DEFAULT_TEST_SET
    if not target.exists():
        console.print(f"[red]Test set not found: {target}[/red]")
        raise typer.Exit(1)

    report = run_eval(test_set_path=target, top_k=top_k, debug=debug)
    print_report(report)
