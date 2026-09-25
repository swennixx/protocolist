import os
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Computed, DateTime, ForeignKey, Index, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Minimal .env loader (KEY=VALUE lines) so the API and the worker share one config file.
_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env):
    for _line in open(_env, encoding="utf-8"):
        _k, _, _v = _line.strip().partition("=")
        if _k and not _k.startswith("#") and _v:
            os.environ.setdefault(_k, _v)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://protocolist:protocolist@localhost:5433/protocolist")
DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
EMBED_DIM = 384  # intfloat/multilingual-e5-small

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class Meeting(Base):
    __tablename__ = "meetings"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    filename: Mapped[str] = mapped_column(String(300))
    # uploading -> queued -> processing -> done | error
    status: Mapped[str] = mapped_column(String(20), default="uploading", index=True)
    stage: Mapped[int] = mapped_column(default=0)  # index of the next stage to run
    stage_progress: Mapped[float] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text)
    size: Mapped[int] = mapped_column(default=0)
    received: Mapped[int] = mapped_column(default=0)
    language: Mapped[str | None] = mapped_column(String(8))  # None = auto
    num_speakers: Mapped[int | None]
    duration: Mapped[float] = mapped_column(default=0)
    has_audio: Mapped[bool] = mapped_column(default=True)
    peaks: Mapped[list] = mapped_column(default=list)
    # Stage checkpoints, so a restarted job resumes instead of redoing work.
    asr: Mapped[dict | None]
    turns: Mapped[list | None]
    summary: Mapped[list] = mapped_column(default=list)
    topics: Mapped[list] = mapped_column(default=list)
    timings: Mapped[dict] = mapped_column(default=dict)

    speakers: Mapped[list["Speaker"]] = relationship(order_by="Speaker.idx", cascade="all, delete-orphan")
    segments: Mapped[list["Segment"]] = relationship(order_by="Segment.idx", cascade="all, delete-orphan")
    decisions: Mapped[list["Decision"]] = relationship(order_by="Decision.id", cascade="all, delete-orphan")
    tasks: Mapped[list["Task"]] = relationship(order_by="Task.id", cascade="all, delete-orphan")


class Speaker(Base):
    __tablename__ = "speakers"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int]
    label: Mapped[str] = mapped_column(String(40))  # diarization label, e.g. SPEAKER_01
    name: Mapped[str] = mapped_column(String(100))


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int]
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"))
    start: Mapped[float]
    end: Mapped[float]
    text: Mapped[str] = mapped_column(Text)  # verbatim
    text_clean: Mapped[str] = mapped_column(Text)
    original_clean: Mapped[str | None] = mapped_column(Text)  # set when a user edits the text
    words: Mapped[list] = mapped_column(default=list)
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('russian', text_clean)", persisted=True))
    embedding = mapped_column(Vector(EMBED_DIM), nullable=True)

    __table_args__ = (
        Index("ix_segments_tsv", "tsv", postgresql_using="gin"),
        Index("ix_segments_embedding", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


class Decision(Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    seg: Mapped[int]


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id", ondelete="SET NULL"))
    owner_name: Mapped[str | None] = mapped_column(String(100))
    due: Mapped[date | None]
    seg: Mapped[int]
    done: Mapped[bool] = mapped_column(default=False)


class Setting(Base):
    __tablename__ = "settings"
    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    data: Mapped[dict] = mapped_column(default=dict)


DEFAULT_SETTINGS = {
    "model": "large-v3-turbo",
    "lang": "auto",
    "llm": "qwen3:8b",
    "template": "short",
    "glossary": [],
    "deleteAudio": False,
    "diarization": "auto",  # auto = pyannote when available; clustering = built-in, faster
}


def get_settings(db) -> dict:
    row = db.get(Setting, 1)
    return {**DEFAULT_SETTINGS, **(row.data if row else {})}
