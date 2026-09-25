"""Archive search over speaker turns: pgvector similarity plus a Postgres full-text bonus.

Measured in bench/retrieval.py: on natural questions word matching alone finds the right turn first
in 36 % of cases, e5 vectors in 79 %. Rank fusion (RRF) let weak double matches outrank the best
vector hit, so scores are fused instead: similarity + a small bonus for word matches, which breaks
ties and helps with names and numbers.
"""

from sqlalchemy import Float, Text, cast, func, literal, or_, select

from . import ml
from .db import Meeting, Segment, Speaker

MIN_SIMILARITY = 0.8  # e5 cosine similarity below this is mostly noise
FTS_BONUS = 0.02


def _any_word(q: str):
    return func.to_tsquery("russian", func.replace(cast(func.plainto_tsquery("russian", q), Text), "&", "|"))


def _ids(db, q: str, mode: str, meeting_id: str | None, limit: int) -> list[int]:
    if mode == "exact":
        # All words first (what people expect from "exact"); any word if that finds nothing.
        for tsq in (func.websearch_to_tsquery("russian", q), _any_word(q)):
            stmt = select(Segment.id).where(Segment.tsv.op("@@")(tsq)).order_by(func.ts_rank_cd(Segment.tsv, tsq).desc()).limit(limit)
            if meeting_id:
                stmt = stmt.where(Segment.meeting_id == meeting_id)
            if ids := list(db.scalars(stmt)):
                return ids
        return []

    qv = ml.embed([q], query=True)[0]
    tsq = _any_word(q)
    sim = 1 - Segment.embedding.cosine_distance(qv)
    ft = func.ts_rank_cd(Segment.tsv, tsq)
    # Normalise word-match rank by the best one among candidates (window max), so FTS_BONUS is a fraction.
    ft_norm = func.coalesce(ft / func.nullif(func.max(ft).over(), 0), literal(0.0, Float))
    stmt = (
        select(Segment.id, (sim + FTS_BONUS * ft_norm).label("score"))
        .where(Segment.embedding.is_not(None), or_(sim >= MIN_SIMILARITY, Segment.tsv.op("@@")(tsq)))
        .order_by(sim.desc())
        .limit(200)
    )
    if meeting_id:
        stmt = stmt.where(Segment.meeting_id == meeting_id)
    rows = db.execute(stmt).all()
    return [sid for sid, _ in sorted(rows, key=lambda r: r.score, reverse=True)[:limit]]


def search(db, q: str, mode: str = "smart", meeting_id: str | None = None, limit: int = 12) -> list[dict]:
    q = q.strip()
    if not q:
        return []
    top = _ids(db, q, mode, meeting_id, limit)
    if not top:
        return []
    rows = db.execute(
        select(Segment, Meeting.title, Meeting.created_at, Speaker.name, Speaker.idx)
        .join(Meeting, Meeting.id == Segment.meeting_id)
        .join(Speaker, Speaker.id == Segment.speaker_id)
        .where(Segment.id.in_(top))
    ).all()
    by_id = {r[0].id: r for r in rows}
    return [
        {
            "meeting_id": s.meeting_id, "meeting_title": title, "date": created.isoformat(),
            "seg": s.idx, "start": s.start, "text": s.text_clean, "speaker": name, "color": idx % 5,
        }
        for s, title, created, name, idx in (by_id[i] for i in top if i in by_id)
    ]
