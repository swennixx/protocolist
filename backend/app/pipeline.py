"""Processing pipeline. Each stage checkpoints to the DB, so a restarted job resumes where it stopped."""

import logging
import os
import time
from datetime import date

from sqlalchemy import delete, update

from . import llm, ml
from .db import DATA_DIR, Decision, Meeting, Segment, Session, Speaker, Task, get_settings
from .text import build_segments, verify_names

STAGES = ["prepare", "asr", "diarize", "analyze", "index"]
log = logging.getLogger("pipeline")


def meeting_dir(mid: str) -> str:
    return os.path.join(DATA_DIR, mid)


def paths(mid: str) -> dict[str, str]:
    d = meeting_dir(mid)
    return {"upload": os.path.join(d, "upload"), "wav": os.path.join(d, "audio.wav"), "play": os.path.join(d, "audio.m4a")}


class Reporter:
    """Writes stage progress to the DB at most a few times per second (the UI reads it over SSE)."""

    def __init__(self, mid: str):
        self.mid, self.last = mid, 0.0

    def __call__(self, frac: float):
        now = time.monotonic()
        if now - self.last > 0.4 or frac >= 1:
            self.last = now
            with Session.begin() as db:
                db.execute(update(Meeting).where(Meeting.id == self.mid).values(stage_progress=round(min(frac, 1), 3)))


def run(mid: str) -> None:
    with Session() as db:
        stage = db.get(Meeting, mid).stage
    for i in range(stage, len(STAGES)):
        t0 = time.monotonic()
        globals()[f"stage_{STAGES[i]}"](mid, Reporter(mid))
        with Session.begin() as db:
            m = db.get(Meeting, mid)
            m.stage, m.stage_progress = i + 1, 0
            m.timings = {**m.timings, STAGES[i]: round(time.monotonic() - t0, 1)}
    with Session.begin() as db:
        m = db.get(Meeting, mid)
        m.status = "done"
        if get_settings(db)["deleteAudio"]:
            for p in paths(mid).values():
                if os.path.exists(p):
                    os.remove(p)
            m.has_audio = False


def stage_prepare(mid: str, progress: Reporter) -> None:
    p = paths(mid)
    duration = ml.prepare_audio(p["upload"], p["wav"], p["play"])
    progress(0.7)
    pk = ml.peaks(ml.load_wav(p["wav"]))
    with Session.begin() as db:
        m = db.get(Meeting, mid)
        m.duration, m.peaks = duration, pk
    os.remove(p["upload"])  # the AAC copy is kept for playback


def stage_asr(mid: str, progress: Reporter) -> None:
    with Session() as db:
        m = db.get(Meeting, mid)
        s = get_settings(db)
        lang = m.language or (None if s["lang"] == "auto" else s["lang"])
    prompt = ("Термины: " + ", ".join(s["glossary"]) + ".") if s["glossary"] else None
    llm.unload_all()
    result = ml.transcribe(paths(mid)["wav"], s["model"], lang, prompt, progress)
    with Session.begin() as db:
        m = db.get(Meeting, mid)
        m.asr, m.language = result, result["language"]


def stage_diarize(mid: str, progress: Reporter) -> None:
    with Session() as db:
        m = db.get(Meeting, mid)
        num, words = m.num_speakers, m.asr["words"]
    turns, backend = ml.diarize(paths(mid)["wav"], num, progress, words)
    segs = build_segments(words, turns)

    with Session.begin() as db:
        m = db.get(Meeting, mid)
        m.turns = turns
        m.timings = {**m.timings, "diarization": backend}
        db.execute(delete(Segment).where(Segment.meeting_id == mid))
        db.execute(delete(Speaker).where(Speaker.meeting_id == mid))
        # Number speakers in order of first appearance: «Спикер 1» speaks first.
        order = list(dict.fromkeys(s["speaker"] for s in segs))
        speakers = {label: Speaker(meeting_id=mid, idx=i, label=label, name=f"Спикер {i + 1}") for i, label in enumerate(order)}
        db.add_all(speakers.values())
        db.flush()
        db.add_all(
            Segment(
                meeting_id=mid, idx=i, speaker_id=speakers[s["speaker"]].id, start=s["start"], end=s["end"],
                text=s["text"], text_clean=s["text_clean"],
                words=[{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2)} for w in s["words"]],
            )
            for i, s in enumerate(segs)
        )


def stage_analyze(mid: str, progress: Reporter) -> None:
    with Session() as db:
        m = db.get(Meeting, mid)
        s = get_settings(db)
        if s["llm"] == "off":
            return
        labels = {sp.id: sp.name for sp in m.speakers}
        lines = [f"[{seg.idx}] {labels[seg.speaker_id]}: {seg.text_clean}" for seg in m.segments]
        turns = [(labels[seg.speaker_id], seg.text_clean) for seg in m.segments]
        day = m.created_at.astimezone().date()  # local date: «до пятницы» is counted from it
    progress(0.02)
    r = llm.analyze(s["llm"], s["template"], day, lines, progress)

    n = len(lines)
    valid = lambda seg: seg if isinstance(seg, int) and 0 <= seg < n else 0  # noqa: E731
    with Session.begin() as db:
        m = db.get(Meeting, mid)
        by_name = {sp.name.lower(): sp for sp in m.speakers}
        # The LLM only proposes names; verify_names keeps those the transcript supports.
        # The LLM only lists names it heard; which speaker owns which name is decided from the transcript.
        proposals = [{"label": sp.name, "name": n} for n in r["people"] for sp in m.speakers]
        verified = verify_names(proposals, turns)
        log.info("speaker names: heard %s, verified %s", r["people"], verified)
        for label, name in verified.items():
            sp = by_name.get(label.lower())
            if sp and sp.name.startswith("Спикер"):
                sp.name = name[:100]
        by_name = {sp.name.lower(): sp for sp in m.speakers} | by_name  # old labels still resolve
        m.summary = [x for x in r["summary"] if x.strip()]
        m.topics = [{"title": t["title"], "seg": valid(t["seg"])} for t in r["topics"]]
        db.execute(delete(Decision).where(Decision.meeting_id == mid))
        db.execute(delete(Task).where(Task.meeting_id == mid))
        db.add_all(Decision(meeting_id=mid, text=d["text"], seg=valid(d["seg"])) for d in r["decisions"])
        for t in r["tasks"]:
            owner = by_name.get((t["owner"] or "").lower())
            db.add(Task(
                meeting_id=mid, text=t["text"], seg=valid(t["seg"]), owner_id=owner.id if owner else None,
                owner_name=None if owner else t["owner"], due=_parse_date(t["due"]),
            ))


def _parse_date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def stage_index(mid: str, progress: Reporter) -> None:
    with Session() as db:
        segs = db.query(Segment).filter(Segment.meeting_id == mid).order_by(Segment.idx).all()
        names = {sp.id: sp.name for sp in db.get(Meeting, mid).speakers}
    # ponytail: one vector per speaker turn; add overlapping windows if long monologues hurt recall.
    vecs = ml.embed([f"{names[s.speaker_id]}: {s.text_clean}" for s in segs])
    progress(0.9)
    with Session.begin() as db:
        for s, v in zip(segs, vecs):
            db.execute(update(Segment).where(Segment.id == s.id).values(embedding=v))
