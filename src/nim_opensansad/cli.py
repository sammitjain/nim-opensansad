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
    from .metadata import DEFAULT_DB_PATH

    console.print(f"\n[bold]Query:[/bold] {query}")
    if mp:
        console.print(f"[bold]MP filter:[/bold] {mp}")
    if min_:
        console.print(f"[bold]Ministry filter:[/bold] {min_}")
    console.print()

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
        engine = build_query_engine()
        # If we have stats evidence, prepend it to the query for the LLM
        if evidence:
            augmented_query = (
                f"Use the following statistical context to help answer the query.\n\n"
                f"{evidence}\n\n"
                f"Query: {query}"
            )
            response = engine.query(augmented_query)
        else:
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
def stats():
    """Show collection stats: chunk count, disk usage, and extrapolations."""
    from pymilvus import connections, Collection
    from . import config
    from .ingest import client_has_collection

    if not client_has_collection():
        console.print("[red]No collection found. Run 'opensansad ingest' first.[/red]")
        raise typer.Exit(1)

    connections.connect(uri=config.MILVUS_URI)
    col = Collection(config.COLLECTION_NAME)
    col.flush()
    num_chunks = col.num_entities

    # Disk usage per component
    volumes_dir = Path(__file__).parent.parent.parent / "volumes"
    components = {"milvus": None, "minio": None, "etcd": None}
    total_kb = 0
    for name in components:
        p = volumes_dir / name
        if p.exists():
            result = subprocess.run(
                ["du", "-sk", str(p)], capture_output=True, text=True
            )
            kb = int(result.stdout.split()[0])
            components[name] = kb
            total_kb += kb

    def fmt_size(kb: int) -> str:
        if kb >= 1_048_576:
            return f"{kb / 1_048_576:.1f} GB"
        elif kb >= 1024:
            return f"{kb / 1024:.1f} MB"
        return f"{kb} KB"

    table = Table(title=f"Collection: {config.COLLECTION_NAME}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Chunks", f"{num_chunks:,}")
    table.add_row("", "")
    table.add_row("[dim]Disk breakdown[/dim]", "")
    for name, kb in components.items():
        if kb is not None:
            table.add_row(f"  volumes/{name}/", fmt_size(kb))
    table.add_row("  [bold]Total[/bold]", f"[bold]{fmt_size(total_kb)}[/bold]")

    # Extrapolation
    if num_chunks > 0:
        kb_per_chunk = total_kb / num_chunks
        table.add_row("", "")
        table.add_row("[dim]Extrapolations[/dim]", "")
        for target in [500_000, 1_000_000, 2_500_000]:
            est = kb_per_chunk * target
            table.add_row(f"  at {target:,} chunks", fmt_size(int(est)))

    console.print(table)
