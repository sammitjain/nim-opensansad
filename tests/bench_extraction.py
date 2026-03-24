"""
Benchmark: compare PDF extraction quality across 3 engines.

Engines:
  1. docling         — your existing extract_single_pdf(engine="docling")
  2. docling+easyocr — your existing extract_single_pdf(engine="easyocr")
  3. unstructured    — unstructured hi_res (detectron2 + tesseract)

Test set: 10 smallest + 10 largest parsed JSONs from lok-sabha-dataset session 7.

Output:
  - Per-file timing + memory to console
  - Full extraction text to tests/bench_outputs/{engine}/{stem}.txt
  - Summary JSON to tests/bench_results.json
  - Comparison against your existing extractions in the dataset

Usage:
  cd ~/Downloads/nim-opensansad
  uv run python tests/bench_extraction.py
  uv run python tests/bench_extraction.py --engines unstructured
  uv run python tests/bench_extraction.py --limit 3 --group small
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

# --- Make lok-sabha-dataset importable ---
_DATASET_SRC = Path("/Users/sammitjain/Downloads/lok-sabha-dataset/src")
if str(_DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(_DATASET_SRC))

DATASET_BASE = Path("/Users/sammitjain/Downloads/lok-sabha-dataset")
OUTPUT_DIR = Path(__file__).parent
BENCH_OUTPUTS = OUTPUT_DIR / "bench_outputs"

TEST_FILES = {
    "small": [
        "AU2835_9TgYcF", "AU3936_GrbGhh", "AU2777_Q48hXO", "AU749_edDhL6",
        "AU1091_fsq7rk", "AU3284_Qfs8Ot", "AU1508_2GXUM0", "AU4140_yTmvBE",
        "AU2012_AaCnFi", "AU161_yEFRkM",
    ],
    "large": [
        "AU3660_UGV3NH", "AS39_NyOnQR", "AS284_m3ib4S", "AU3340_jjKrE7",
        "AU3427_u7OhkJ", "AU2103_FrvTJQ", "AU2356_lSA4ks", "AU2186_bCwfDa",
        "AU1193_kCESGe", "AU926_PgMd3z",
    ],
}


def resolve_pdf(stem: str) -> Path:
    return DATASET_BASE / "data" / "18" / "pdfs" / "session_7" / f"{stem}.pdf"


def load_existing_extraction(stem: str) -> dict | None:
    p = DATASET_BASE / "data" / "18" / "parsed" / "session_7" / f"{stem}.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


def save_full_text(engine_name: str, stem: str, text: str):
    """Write full extracted text to bench_outputs/{engine}/{stem}.txt"""
    engine_dir = BENCH_OUTPUTS / engine_name.replace("+", "_")
    engine_dir.mkdir(parents=True, exist_ok=True)
    out = engine_dir / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Engine wrappers — each returns (text: str, extra_meta: dict)
# ---------------------------------------------------------------------------

def extract_docling(pdf_path: Path) -> tuple[str, dict]:
    """Uses your existing pipeline: extract_single_pdf(engine='docling')"""
    from lok_sabha_dataset.pipeline.extract import extract_single_pdf
    result = extract_single_pdf(pdf_path, engine="docling")
    return result["full_markdown"], {
        "num_pages": result["metadata"]["num_pages"],
        "engine_label": result["engine"],
    }


def extract_easyocr(pdf_path: Path) -> tuple[str, dict]:
    """Uses your existing pipeline: extract_single_pdf(engine='easyocr')"""
    from lok_sabha_dataset.pipeline.extract import extract_single_pdf
    result = extract_single_pdf(pdf_path, engine="easyocr")
    return result["full_markdown"], {
        "num_pages": result["metadata"]["num_pages"],
        "engine_label": result["engine"],
    }


def extract_unstructured(pdf_path: Path) -> tuple[str, dict]:
    """Unstructured with hi_res strategy (detectron2 + tesseract)."""
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(str(pdf_path), strategy="hi_res")
    text = "\n\n".join(el.text for el in elements if el.text)
    return text, {
        "num_elements": len(elements),
        "element_types": sorted({type(el).__name__ for el in elements}),
    }


def extract_unstructured_md(pdf_path: Path) -> tuple[str, dict]:
    """Unstructured hi_res → markdown via elements_to_md (built-in converter)."""
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import elements_to_md
    elements = partition_pdf(str(pdf_path), strategy="hi_res", infer_table_structure=True)
    text = elements_to_md(elements)
    return text, {
        "num_elements": len(elements),
        "element_types": sorted({type(el).__name__ for el in elements}),
    }


ENGINES = {
    "docling": extract_docling,
    "docling+easyocr": extract_easyocr,
    "unstructured": extract_unstructured,
    "unstructured-md": extract_unstructured_md,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_one(engine_name: str, engine_fn, pdf_path: Path, stem: str) -> dict:
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()

    try:
        text, meta = engine_fn(pdf_path)
        error = None
    except Exception as e:
        text = ""
        meta = {}
        error = f"{type(e).__name__}: {e}"

    elapsed = round(time.perf_counter() - t0, 2)
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_delta_mb = round((mem_peak - mem_before) / (1024 * 1024), 1)
    word_count = len(text.split()) if text else 0

    # Save full extraction text
    save_full_text(engine_name, stem, text)

    return {
        "engine": engine_name,
        "pdf": pdf_path.name,
        "stem": stem,
        "time_sec": elapsed,
        "mem_peak_delta_mb": mem_delta_mb,
        "char_count": len(text),
        "word_count": word_count,
        "preview": (text[:300] if text else "(empty)"),
        "metadata": meta,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PDF extraction engines")
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINES.keys()),
        choices=list(ENGINES.keys()),
        help="Which engines to test (default: all three)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to N files per group (small/large)",
    )
    parser.add_argument(
        "--group", choices=["small", "large", "all"], default="all",
        help="Which file group to test",
    )
    args = parser.parse_args()

    groups = ["small", "large"] if args.group == "all" else [args.group]
    stems = []
    for g in groups:
        g_stems = TEST_FILES[g]
        if args.limit:
            g_stems = g_stems[:args.limit]
        stems.extend([(g, s) for s in g_stems])

    total_runs = len(stems) * len(args.engines)
    print(f"Benchmark: {len(stems)} files x {len(args.engines)} engines = {total_runs} extractions")
    print(f"Engines: {', '.join(args.engines)}")
    print(f"Full text output: {BENCH_OUTPUTS}/")
    print()

    results = []

    for engine_name in args.engines:
        engine_fn = ENGINES[engine_name]
        print(f"{'='*70}")
        print(f"ENGINE: {engine_name}")
        print(f"{'='*70}")

        for group, stem in stems:
            pdf_path = resolve_pdf(stem)
            if not pdf_path.exists():
                print(f"  [SKIP] {stem} -- PDF not found")
                continue

            print(f"  [{group:>5}] {stem} ... ", end="", flush=True)
            r = bench_one(engine_name, engine_fn, pdf_path, stem)
            r["group"] = group
            results.append(r)

            if r["error"]:
                print(f"ERROR ({r['time_sec']}s): {r['error']}")
            else:
                print(
                    f"{r['time_sec']:>6.1f}s  {r['mem_peak_delta_mb']:>6.1f}MB  "
                    f"{r['char_count']:>7} chars  {r['word_count']:>6} words"
                )

        print()

    # --- Save full results JSON ---
    results_path = OUTPUT_DIR / "bench_results.json"
    with results_path.open("w") as f:
        # Strip preview from the saved JSON to keep it clean;
        # full text is already in bench_outputs/
        saved = [{k: v for k, v in r.items() if k != "preview"} for r in results]
        json.dump(saved, f, indent=2, ensure_ascii=False)
    print(f"Results JSON: {results_path}")

    # --- Summary table ---
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Engine':<20} {'Group':<6} {'File':<20} {'Time':>7} {'Memory':>8} {'Chars':>8} {'Words':>7}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['engine']:<20} {r['group']:<6} {r['stem']:<20} "
            f"{r['time_sec']:>6.1f}s {r['mem_peak_delta_mb']:>7.1f}MB "
            f"{r['char_count']:>8} {r['word_count']:>7}"
        )

    # --- Per-engine aggregates ---
    print(f"\n{'='*70}")
    print("AGGREGATES")
    print(f"{'='*70}")
    for eng in args.engines:
        eng_results = [r for r in results if r["engine"] == eng and not r["error"]]
        if not eng_results:
            print(f"{eng}: no successful extractions")
            continue
        total_time = sum(r["time_sec"] for r in eng_results)
        avg_time = total_time / len(eng_results)
        total_chars = sum(r["char_count"] for r in eng_results)
        avg_mem = sum(r["mem_peak_delta_mb"] for r in eng_results) / len(eng_results)
        empty = sum(1 for r in eng_results if r["char_count"] < 50)
        print(
            f"  {eng:<20}  total={total_time:>7.1f}s  avg={avg_time:>6.1f}s  "
            f"chars={total_chars:>9}  avg_mem={avg_mem:>6.1f}MB  empty(<50chars)={empty}"
        )

    # --- Comparison vs existing extractions ---
    print(f"\n{'='*70}")
    print("vs EXISTING EXTRACTIONS (from lok-sabha-dataset)")
    print(f"{'='*70}")
    print(f"  {'Engine':<20} {'File':<20} {'Existing':>9} {'New':>9} {'Diff':>9}")
    print("  " + "-" * 68)
    for r in results:
        existing = load_existing_extraction(r["stem"])
        if existing:
            existing_len = len(existing.get("full_markdown", ""))
            diff = r["char_count"] - existing_len
            sign = "+" if diff > 0 else ""
            print(
                f"  {r['engine']:<20} {r['stem']:<20} "
                f"{existing_len:>9} {r['char_count']:>9} {sign}{diff:>8}"
            )

    print(f"\nFull extracted text is in: {BENCH_OUTPUTS}/")
    print("  e.g. diff tests/bench_outputs/docling/AU926_PgMd3z.txt tests/bench_outputs/unstructured/AU926_PgMd3z.txt")


if __name__ == "__main__":
    main()
