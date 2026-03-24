"""Query MP and ministry statistics from the SQLite metadata database.

Provides:
  - get_mp_stats()              → aggregate profile of an MP
  - get_ministry_stats()        → aggregate profile of a ministry
  - get_overlap_stats()         → intersection: specific MP × specific ministry
  - format_*_for_llm()          → text blocks for LLM context injection
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .metadata import DEFAULT_DB_PATH


@dataclass
class QuestionRecord:
    lok_no: int
    session_no: int
    ques_no: int
    type: str
    date: Optional[str]
    ministry: str
    subject: str


@dataclass
class MpStats:
    mp_name: str
    total_questions: int
    by_lok: dict[int, int]
    by_session: dict[str, int]
    by_type: dict[str, int]
    by_ministry: List[Tuple[str, int]]  # sorted desc, top 15
    recent_questions: List[QuestionRecord]


@dataclass
class MinistryStats:
    ministry: str
    total_questions: int
    by_lok: dict[int, int]
    by_session: dict[str, int]
    by_type: dict[str, int]
    top_mps: List[Tuple[str, int]]  # sorted desc, top 15
    recent_questions: List[QuestionRecord]


@dataclass
class OverlapStats:
    mp_name: str
    ministry: str
    total_questions: int
    by_type: dict[str, int]
    recent_questions: List[QuestionRecord]


# ── Queries ──────────────────────────────────────────────────────────────────


def get_mp_stats(
    mp_name: str,
    top_q: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Optional[MpStats]:
    """Aggregate stats for an MP. Returns None if no questions found."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT q.lok_no, q.session_no, q.ques_no, q.type, q.date,
                   q.ministry, q.subject
            FROM questions q
            JOIN question_mps m USING (lok_no, session_no, type, ques_no)
            WHERE m.mp_name = ?
            ORDER BY q.date DESC, q.ques_no DESC
            """,
            (mp_name,),
        ).fetchall()

        if not rows:
            return None

        by_lok: dict[int, int] = {}
        by_session: dict[str, int] = {}
        by_ministry: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for r in rows:
            by_lok[r["lok_no"]] = by_lok.get(r["lok_no"], 0) + 1
            key = f"Lok {r['lok_no']} Session {r['session_no']}"
            by_session[key] = by_session.get(key, 0) + 1
            by_ministry[r["ministry"]] = by_ministry.get(r["ministry"], 0) + 1
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1

        ministry_sorted = sorted(by_ministry.items(), key=lambda x: -x[1])[:15]

        recent = [
            QuestionRecord(
                lok_no=r["lok_no"], session_no=r["session_no"],
                ques_no=r["ques_no"], type=r["type"], date=r["date"],
                ministry=r["ministry"], subject=(r["subject"] or "").strip(),
            )
            for r in rows[:top_q]
        ]

        return MpStats(
            mp_name=mp_name,
            total_questions=len(rows),
            by_lok=dict(sorted(by_lok.items())),
            by_session=dict(sorted(by_session.items())),
            by_type=dict(sorted(by_type.items())),
            by_ministry=ministry_sorted,
            recent_questions=recent,
        )
    finally:
        conn.close()


def get_ministry_stats(
    ministry: str,
    top_q: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Optional[MinistryStats]:
    """Aggregate stats for a ministry. Returns None if no questions found."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT lok_no, session_no, ques_no, type, date,
                   ministry, subject
            FROM questions
            WHERE ministry = ?
            ORDER BY date DESC, ques_no DESC
            """,
            (ministry,),
        ).fetchall()

        if not rows:
            return None

        by_lok: dict[int, int] = {}
        by_session: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for r in rows:
            by_lok[r["lok_no"]] = by_lok.get(r["lok_no"], 0) + 1
            key = f"Lok {r['lok_no']} Session {r['session_no']}"
            by_session[key] = by_session.get(key, 0) + 1
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1

        mp_rows = conn.execute(
            """
            SELECT m.mp_name, COUNT(*) as cnt
            FROM question_mps m
            JOIN questions q USING (lok_no, session_no, type, ques_no)
            WHERE q.ministry = ?
            GROUP BY m.mp_name
            ORDER BY cnt DESC
            LIMIT 15
            """,
            (ministry,),
        ).fetchall()
        top_mps = [(r["mp_name"], r["cnt"]) for r in mp_rows]

        recent = [
            QuestionRecord(
                lok_no=r["lok_no"], session_no=r["session_no"],
                ques_no=r["ques_no"], type=r["type"], date=r["date"],
                ministry=r["ministry"], subject=(r["subject"] or "").strip(),
            )
            for r in rows[:top_q]
        ]

        return MinistryStats(
            ministry=ministry,
            total_questions=len(rows),
            by_lok=dict(sorted(by_lok.items())),
            by_session=dict(sorted(by_session.items())),
            by_type=dict(sorted(by_type.items())),
            top_mps=top_mps,
            recent_questions=recent,
        )
    finally:
        conn.close()


def get_overlap_stats(
    mp_name: str,
    ministry: str,
    top_q: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Optional[OverlapStats]:
    """Stats for the intersection: questions by this MP to this ministry."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT q.lok_no, q.session_no, q.ques_no, q.type, q.date,
                   q.ministry, q.subject
            FROM questions q
            JOIN question_mps m USING (lok_no, session_no, type, ques_no)
            WHERE m.mp_name = ? AND q.ministry = ?
            ORDER BY q.date DESC, q.ques_no DESC
            """,
            (mp_name, ministry),
        ).fetchall()

        if not rows:
            return None

        by_type: dict[str, int] = {}
        for r in rows:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1

        recent = [
            QuestionRecord(
                lok_no=r["lok_no"], session_no=r["session_no"],
                ques_no=r["ques_no"], type=r["type"], date=r["date"],
                ministry=r["ministry"], subject=(r["subject"] or "").strip(),
            )
            for r in rows[:top_q]
        ]

        return OverlapStats(
            mp_name=mp_name,
            ministry=ministry,
            total_questions=len(rows),
            by_type=dict(sorted(by_type.items())),
            recent_questions=recent,
        )
    finally:
        conn.close()


# ── LLM formatters ──────────────────────────────────────────────────────────


def _format_questions(questions: List[QuestionRecord], label: str = "Recent") -> list[str]:
    """Shared helper for formatting question lists."""
    lines = []
    if questions:
        lines.append(f"{label} Questions ({len(questions)}):")
        for i, q in enumerate(questions, 1):
            lines.append(
                f"  {i}. [Lok {q.lok_no}, Session {q.session_no}, "
                f"Q{q.ques_no}, {q.type}] "
                f"{q.ministry} — {q.subject} ({q.date or 'N/A'})"
            )
        lines.append("")
    return lines


def format_mp_stats_for_llm(stats: MpStats) -> str:
    """Format MP stats into a text block for LLM context."""
    lines: list[str] = []

    lines.append("=== MP STATISTICS ===")
    lines.append(f"MP Name: {stats.mp_name}")
    lines.append(f"Total questions asked: {stats.total_questions}")
    lines.append("")

    lines.append("By Lok Sabha:")
    for lok, count in stats.by_lok.items():
        lines.append(f"  Lok {lok}: {count}")
    lines.append("")

    lines.append("By Session:")
    for key, count in stats.by_session.items():
        lines.append(f"  {key}: {count}")
    lines.append("")

    lines.append("By Type:")
    for qtype, count in stats.by_type.items():
        lines.append(f"  {qtype}: {count}")
    lines.append("")

    lines.append("Top Ministries:")
    for ministry, count in stats.by_ministry:
        lines.append(f"  {count:4d}  {ministry}")
    lines.append("")

    lines.extend(_format_questions(stats.recent_questions))

    lines.append(
        "NOTE: Statistics cover ALL questions by this MP, including those "
        "whose full text may not appear in the retrieved chunks."
    )
    lines.append("=== END MP STATISTICS ===")

    return "\n".join(lines)


def format_ministry_stats_for_llm(stats: MinistryStats) -> str:
    """Format ministry stats into a text block for LLM context."""
    lines: list[str] = []

    lines.append("=== MINISTRY STATISTICS ===")
    lines.append(f"Ministry: {stats.ministry}")
    lines.append(f"Total questions received: {stats.total_questions}")
    lines.append("")

    lines.append("By Lok Sabha:")
    for lok, count in stats.by_lok.items():
        lines.append(f"  Lok {lok}: {count}")
    lines.append("")

    lines.append("By Type:")
    for qtype, count in stats.by_type.items():
        lines.append(f"  {qtype}: {count}")
    lines.append("")

    lines.append("Top MPs asking this ministry:")
    for mp, count in stats.top_mps:
        lines.append(f"  {count:4d}  {mp}")
    lines.append("")

    lines.extend(_format_questions(stats.recent_questions))

    lines.append(
        "NOTE: Statistics cover ALL questions to this ministry, including those "
        "whose full text may not appear in the retrieved chunks."
    )
    lines.append("=== END MINISTRY STATISTICS ===")

    return "\n".join(lines)


def format_overlap_stats_for_llm(stats: OverlapStats) -> str:
    """Format overlap (MP × ministry) stats into a text block for LLM context."""
    lines: list[str] = []

    lines.append(f"=== OVERLAP: {stats.mp_name} × {stats.ministry} ===")
    lines.append(
        f"Questions from this MP to this Ministry: {stats.total_questions}"
    )

    if stats.by_type:
        lines.append("By Type: " + ", ".join(
            f"{t}: {c}" for t, c in stats.by_type.items()
        ))
    lines.append("")

    lines.extend(_format_questions(stats.recent_questions))

    lines.append(f"=== END OVERLAP ===")

    return "\n".join(lines)


def build_evidence_packet(
    mp_name: Optional[str] = None,
    ministry: Optional[str] = None,
    top_q: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Optional[str]:
    """Build the combined evidence packet for LLM context injection.

    When both mp_name and ministry are given:
      - Overlap section gets full top_q
      - MP and ministry sections each get top_q // 2

    When only one is given, it gets full top_q.

    Returns None if no stats found for any filter.
    """
    sections: list[str] = []

    if mp_name and ministry:
        # Both filters — overlap first, then reduced individual stats
        half_q = max(top_q // 2, 3)

        overlap = get_overlap_stats(mp_name, ministry, top_q=top_q, db_path=db_path)
        if overlap:
            sections.append(format_overlap_stats_for_llm(overlap))

        mp_stats = get_mp_stats(mp_name, top_q=half_q, db_path=db_path)
        if mp_stats:
            sections.append(format_mp_stats_for_llm(mp_stats))

        min_stats = get_ministry_stats(ministry, top_q=half_q, db_path=db_path)
        if min_stats:
            sections.append(format_ministry_stats_for_llm(min_stats))

    elif mp_name:
        mp_stats = get_mp_stats(mp_name, top_q=top_q, db_path=db_path)
        if mp_stats:
            sections.append(format_mp_stats_for_llm(mp_stats))

    elif ministry:
        min_stats = get_ministry_stats(ministry, top_q=top_q, db_path=db_path)
        if min_stats:
            sections.append(format_ministry_stats_for_llm(min_stats))

    if not sections:
        return None

    return "\n\n".join(sections)
