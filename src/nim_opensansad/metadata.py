"""Build a SQLite metadata database from the HuggingFace dataset.

Schema:
  questions      — one row per parliamentary question (primary key: lok/session/type/ques_no)
  question_mps   — many-to-many: which MPs asked which question (canonicalised names)

The DB is used for aggregate stats (MP profiles, ministry summaries) that can't be
answered by chunk retrieval alone.

Usage:
  uv run opensansad build-db
  uv run opensansad build-db --parquet /path/to/local.parquet
  uv run opensansad build-db --db-path data/custom.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_tree
from rich.console import Console

from . import config

console = Console()

HF_DATASET = "opensansad/lok-sabha-qa"
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "metadata.db"

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS questions (
    lok_no       INTEGER,
    session_no   INTEGER,
    ques_no      INTEGER,
    type         TEXT,
    date         TEXT,
    ministry     TEXT,
    subject      TEXT,
    PRIMARY KEY (lok_no, session_no, type, ques_no)
);

CREATE TABLE IF NOT EXISTS question_mps (
    lok_no     INTEGER,
    session_no INTEGER,
    ques_no    INTEGER,
    type       TEXT,
    mp_name    TEXT
);

CREATE INDEX IF NOT EXISTS idx_qmp_question
    ON question_mps (lok_no, session_no, type, ques_no);

CREATE INDEX IF NOT EXISTS idx_qmp_mp
    ON question_mps (mp_name);
"""


# ── MP name canonicalisation ─────────────────────────────────────────────────


def _discover_loks(dataset_repo: str) -> list[int]:
    """Discover available Lok Sabha numbers from supplementary/ on HuggingFace."""
    loks = []
    for entry in list_repo_tree(dataset_repo, path_in_repo="supplementary", repo_type="dataset"):
        name = Path(entry.path).name
        if name.isdigit():
            loks.append(int(name))
    return sorted(loks)


def _build_mp_name_map(dataset_repo: str) -> dict[str, str]:
    """Build old_name → canonical_name map using mpNo from members.json.

    When the same mpNo appears across multiple Lok Sabhas with different
    display names, the most recent (highest) Lok Sabha's name is canonical.
    """
    by_mpno: dict[int, list[tuple[int, str]]] = {}
    loks = _discover_loks(dataset_repo)
    console.print(f"  [dim]Discovered Lok Sabhas: {loks}[/dim]")

    for lok in loks:
        try:
            path = hf_hub_download(
                repo_id=dataset_repo,
                filename=f"supplementary/{lok}/members.json",
                repo_type="dataset",
            )
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            continue

        for entry in entries:
            mp_no = entry.get("mpNo")
            mp_name = entry.get("mpName")
            if mp_no and mp_name:
                by_mpno.setdefault(mp_no, []).append((lok, mp_name))

    # For each mpNo, pick the name from the latest lok as canonical
    name_map: dict[str, str] = {}
    for mp_no, entries in by_mpno.items():
        canonical = max(entries, key=lambda x: x[0])[1]
        for _, name in entries:
            name_map[name] = canonical

    return name_map


def get_all_canonical_mp_names(dataset_repo: str = HF_DATASET) -> list[str]:
    """Return sorted list of all canonical MP names (for list/search commands)."""
    name_map = _build_mp_name_map(dataset_repo)
    # Canonical names are the unique values
    return sorted(set(name_map.values()))


def get_all_ministry_names(dataset_repo: str = HF_DATASET) -> list[str]:
    """Return sorted list of all ministry names from supplementary data.

    Whitespace-normalised but not renamed (ministry renames across Lok Sabhas
    are kept as separate entries — see ARCHITECTURE.md for rationale).
    """
    all_names: set[str] = set()
    loks = _discover_loks(dataset_repo)

    for lok in loks:
        try:
            path = hf_hub_download(
                repo_id=dataset_repo,
                filename=f"supplementary/{lok}/ministries.json",
                repo_type="dataset",
            )
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            continue

        for entry in entries:
            name = entry.get("minName", "").strip()
            if name:
                all_names.add(name)

    return sorted(all_names)


# ── Build DB ─────────────────────────────────────────────────────────────────


def build_metadata_db(
    parquet_path: Optional[Path] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> Path:
    """Build the metadata SQLite database from the dataset.

    Args:
        parquet_path: Local parquet file. If None, downloads from HuggingFace.
        db_path: Where to write the SQLite database.

    Returns:
        Path to the created database.
    """
    # Load dataset
    if parquet_path:
        console.print(f"[bold]Loading from local parquet:[/bold] {parquet_path}")
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    else:
        console.print(f"[bold]Loading from HuggingFace:[/bold] {HF_DATASET}")
        ds = load_dataset(HF_DATASET, split="train")

    console.print(f"  {len(ds)} rows loaded")

    # Build MP canonicalisation map
    console.print("[bold]Building MP name canonicalisation map...[/bold]")
    mp_name_map = _build_mp_name_map(HF_DATASET)
    conflicts = sum(1 for k, v in mp_name_map.items() if k != v)
    console.print(f"  {len(mp_name_map)} names, {conflicts} will be remapped")

    # Create DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    q_count = 0
    mp_count = 0
    mp_renamed = 0

    for row in ds:
        lok = row.get("lok_no")
        sess = row.get("session_no")
        qno = row.get("ques_no")
        if lok is None or sess is None or qno is None:
            continue

        qtype = row.get("type")
        # Whitespace-normalise ministry name
        ministry = (row.get("ministry") or "").strip() or None

        conn.execute(
            "INSERT OR REPLACE INTO questions "
            "(lok_no, session_no, ques_no, type, date, ministry, subject) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (lok, sess, qno, qtype, row.get("date"), ministry,
             row.get("subject") or None),
        )
        q_count += 1

        # Delete existing MP rows for this question (idempotent on re-run)
        conn.execute(
            "DELETE FROM question_mps WHERE lok_no=? AND session_no=? AND type=? AND ques_no=?",
            (lok, sess, qtype, qno),
        )
        members = row.get("members") or []
        if isinstance(members, str):
            members = [m.strip() for m in members.split(",")]
        for mp in members:
            mp = mp.strip()
            if mp:
                canonical = mp_name_map.get(mp, mp)
                if canonical != mp:
                    mp_renamed += 1
                conn.execute(
                    "INSERT INTO question_mps (lok_no, session_no, ques_no, type, mp_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (lok, sess, qno, qtype, canonical),
                )
                mp_count += 1

    conn.commit()
    conn.close()

    console.print(
        f"[green]Done.[/green] {q_count} questions, {mp_count} MP links "
        f"({mp_renamed} canonicalisations) → {db_path}"
    )
    return db_path
