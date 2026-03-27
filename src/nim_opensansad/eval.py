"""Evaluation pipeline for retrieval quality.

Runs queries from a test set through the real retriever (build_retriever
from query.py) and scores retrieved chunks against expected metadata / text.

No LLM or reranker calls — this evaluates the retrieval layer only.

Modes per query:
  - "unfiltered":  pure semantic search, no --mp/--min
  - "filtered":    with metadata filters (only when mp/ministry is set)

Metrics (per query):
  - precision@k:   fraction of retrieved chunks matching expected metadata
  - hit@1 / hit@k: does at least one chunk match?
  - unique_docs:   how many distinct qa_ids were retrieved

Usage:
  uv run opensansad eval
  uv run opensansad eval --test-set eval/test_set.json --top-k 20
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llama_index.core.schema import NodeWithScore
from rich.console import Console
from rich.table import Table

from . import config
from .metadata import DEFAULT_ALIAS_PATH, load_alias_map
from .query import build_retriever

console = Console()

DEFAULT_TEST_SET = Path(__file__).parent.parent.parent / "eval" / "test_set.json"


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TestCase:
    id: str
    query: str
    category: str
    mp: Optional[str]
    ministry: Optional[str]
    expected: dict
    notes: str = ""


@dataclass
class RetrievalResult:
    test_id: str
    mode: str  # "unfiltered" or "filtered"
    query: str
    category: str
    nodes: list[NodeWithScore]
    latency_ms: float
    precision_at_k: float = 0.0
    hit_at_1: bool = False
    hit_at_k: bool = False
    unique_docs: int = 0
    k: int = 0


@dataclass
class EvalReport:
    results: list[RetrievalResult] = field(default_factory=list)

    def add(self, result: RetrievalResult):
        self.results.append(result)

    def summary_by_mode(self, mode: str) -> dict:
        subset = [r for r in self.results if r.mode == mode]
        if not subset:
            return {}
        return {
            "count": len(subset),
            "avg_precision_at_k": sum(r.precision_at_k for r in subset) / len(subset),
            "avg_hit_at_1": sum(r.hit_at_1 for r in subset) / len(subset),
            "avg_hit_at_k": sum(r.hit_at_k for r in subset) / len(subset),
            "avg_unique_docs": sum(r.unique_docs for r in subset) / len(subset),
            "avg_latency_ms": sum(r.latency_ms for r in subset) / len(subset),
        }


# ── Scoring ──────────────────────────────────────────────────────────────────


def _normalize_ws(s: str) -> str:
    """Collapse runs of whitespace into a single space."""
    return " ".join(s.split())


def _chunk_matches(node: NodeWithScore, expected: dict) -> bool:
    """Check if a single retrieved chunk matches the expected criteria."""
    meta = node.metadata
    text = node.text or ""

    if "members_contains_any" in expected:
        members = _normalize_ws(meta.get("members", "")).lower()
        if not any(_normalize_ws(name).lower() in members for name in expected["members_contains_any"]):
            return False

    if "ministry_eq" in expected:
        if _normalize_ws(meta.get("ministry", "")) != _normalize_ws(expected["ministry_eq"]):
            return False

    if "text_contains_any" in expected:
        combined = _normalize_ws(text + " " + meta.get("subject", "")).lower()
        if not any(term.lower() in combined for term in expected["text_contains_any"]):
            return False

    return True


def _effective_expected(test: TestCase, mp_aliases: list[str] | None) -> dict:
    """Build the expected dict used for scoring, expanding members_contains_any
    to the full alias list when available — partial name matching in the test
    set is unreliable given title/case/transliteration variations."""
    expected = dict(test.expected)
    if mp_aliases and "members_contains_any" in expected:
        expected["members_contains_any"] = mp_aliases
    return expected


def score_retrieval(
    test: TestCase,
    nodes: list[NodeWithScore],
    mode: str,
    latency_ms: float,
    mp_aliases: list[str] | None = None,
) -> RetrievalResult:
    """Score retrieved nodes against expected metadata."""
    expected = _effective_expected(test, mp_aliases)
    k = len(nodes)
    matches = [_chunk_matches(n, expected) for n in nodes]
    unique_docs = len({n.metadata.get("qa_id", i) for i, n in enumerate(nodes)})

    return RetrievalResult(
        test_id=test.id,
        mode=mode,
        query=test.query,
        category=test.category,
        nodes=nodes,
        latency_ms=latency_ms,
        precision_at_k=sum(matches) / k if k > 0 else 0.0,
        hit_at_1=matches[0] if matches else False,
        hit_at_k=any(matches),
        unique_docs=unique_docs,
        k=k,
    )


# ── Runner ───────────────────────────────────────────────────────────────────


def _debug_dump(test: TestCase, result: RetrievalResult, mp_aliases: list[str] | None = None):
    """Print each chunk's metadata with pass/fail and the reason for failure."""
    expected = _effective_expected(test, mp_aliases)
    console.print(f"\n    [bold]DEBUG {test.id} ({result.mode})[/bold]  "
                  f"expected: {expected}")
    for i, node in enumerate(result.nodes):
        meta = node.metadata
        match = _chunk_matches(node, expected)
        marker = "[green]PASS[/green]" if match else "[red]FAIL[/red]"

        reasons = []
        if not match:
            if "members_contains_any" in expected:
                members = _normalize_ws(meta.get("members", "")).lower()
                if not any(_normalize_ws(n).lower() in members for n in expected["members_contains_any"]):
                    reasons.append(f"members='{meta.get('members', '')}'")
            if "ministry_eq" in expected:
                if _normalize_ws(meta.get("ministry", "")) != _normalize_ws(expected["ministry_eq"]):
                    reasons.append(f"ministry='{meta.get('ministry')}'")
            if "text_contains_any" in expected:
                combined = _normalize_ws((node.text or "") + " " + meta.get("subject", "")).lower()
                if not any(t.lower() in combined for t in expected["text_contains_any"]):
                    reasons.append(f"subject='{meta.get('subject', '')}'")

        reason_str = f"  ← {'; '.join(reasons)}" if reasons else ""
        console.print(
            f"    [{i+1:2d}] {marker}  qa_id={meta.get('qa_id', '?'):30s}  "
            f"ministry={meta.get('ministry', '?'):40s}  "
            f"members={meta.get('members', '?')[:60]}"
            f"{reason_str}"
        )
    console.print()


def load_test_set(path: Path = DEFAULT_TEST_SET) -> list[TestCase]:
    with open(path, "r") as f:
        raw = json.load(f)
    return [
        TestCase(
            id=t["id"], query=t["query"], category=t["category"],
            mp=t.get("mp"), ministry=t.get("ministry"),
            expected=t["expected"], notes=t.get("notes", ""),
        )
        for t in raw
    ]


def run_eval(
    test_set_path: Path = DEFAULT_TEST_SET,
    top_k: int | None = None,
    debug: bool = False,
) -> EvalReport:
    """Run the full evaluation using the real retriever from query.py."""
    top_k = top_k or config.TOP_K
    tests = load_test_set(test_set_path)
    report = EvalReport()

    alias_map: dict[str, list[str]] = {}
    if DEFAULT_ALIAS_PATH.exists():
        alias_map = load_alias_map(DEFAULT_ALIAS_PATH)

    console.print(f"[bold]Running {len(tests)} queries (top_k={top_k})...[/bold]\n")

    for test in tests:
        # Resolve aliases once per test case — used in both scoring and Milvus filter
        mp_aliases = alias_map.get(test.mp) if test.mp else None

        # --- Unfiltered mode: pure semantic search ---
        retriever = build_retriever(top_k=top_k)

        start = time.perf_counter()
        nodes = retriever.retrieve(test.query)
        latency = (time.perf_counter() - start) * 1000

        result = score_retrieval(test, nodes, "unfiltered", latency, mp_aliases=mp_aliases)
        report.add(result)

        console.print(
            f"  {test.id} [dim]unfiltered[/dim]  "
            f"P@{result.k}={result.precision_at_k:.2f}  "
            f"hit@1={'Y' if result.hit_at_1 else 'N'}  "
            f"docs={result.unique_docs}  "
            f"{result.latency_ms:.0f}ms"
        )
        if debug and result.precision_at_k < 1.0:
            _debug_dump(test, result, mp_aliases=mp_aliases)

        # --- Filtered mode: with --mp/--min (only if applicable) ---
        if test.mp or test.ministry:
            retriever = build_retriever(
                mp=test.mp, ministry=test.ministry,
                mp_aliases=mp_aliases, top_k=top_k,
            )

            start = time.perf_counter()
            nodes = retriever.retrieve(test.query)
            latency = (time.perf_counter() - start) * 1000

            result = score_retrieval(test, nodes, "filtered", latency, mp_aliases=mp_aliases)
            report.add(result)

            console.print(
                f"  {test.id} [cyan]filtered  [/cyan]  "
                f"P@{result.k}={result.precision_at_k:.2f}  "
                f"hit@1={'Y' if result.hit_at_1 else 'N'}  "
                f"docs={result.unique_docs}  "
                f"{result.latency_ms:.0f}ms"
            )
            if debug and result.precision_at_k < 1.0:
                _debug_dump(test, result, mp_aliases=mp_aliases)

    return report


# ── Display ──────────────────────────────────────────────────────────────────


def print_report(report: EvalReport):
    """Print the eval report as rich tables."""
    # Per-query table
    table = Table(title="Retrieval Eval — Per Query")
    table.add_column("ID", style="bold")
    table.add_column("Category")
    table.add_column("Mode")
    table.add_column("P@k", justify="right")
    table.add_column("Hit@1", justify="center")
    table.add_column("Hit@k", justify="center")
    table.add_column("Docs", justify="right")
    table.add_column("Latency", justify="right")

    for r in report.results:
        table.add_row(
            r.test_id,
            r.category,
            r.mode,
            f"{r.precision_at_k:.2f}",
            "Y" if r.hit_at_1 else "N",
            "Y" if r.hit_at_k else "N",
            str(r.unique_docs),
            f"{r.latency_ms:.0f}ms",
        )

    console.print(table)

    # Summary by mode
    summary = Table(title="Retrieval Eval — Summary by Mode")
    summary.add_column("Mode", style="bold")
    summary.add_column("Queries", justify="right")
    summary.add_column("Avg P@k", justify="right")
    summary.add_column("Avg Hit@1", justify="right")
    summary.add_column("Avg Hit@k", justify="right")
    summary.add_column("Avg Docs", justify="right")
    summary.add_column("Avg Latency", justify="right")

    for mode in ["unfiltered", "filtered"]:
        s = report.summary_by_mode(mode)
        if s:
            summary.add_row(
                mode,
                str(s["count"]),
                f"{s['avg_precision_at_k']:.2f}",
                f"{s['avg_hit_at_1']:.2f}",
                f"{s['avg_hit_at_k']:.2f}",
                f"{s['avg_unique_docs']:.1f}",
                f"{s['avg_latency_ms']:.0f}ms",
            )

    console.print(summary)
