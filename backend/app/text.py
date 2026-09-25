"""Pure text helpers: filler removal, word-to-speaker alignment, export formats."""

import re

FILLERS = r"э+|э-э|эм+|ммм*|ну|вот|как бы|типа|короче|значит|собственно|так сказать|в общем"
_filler_re = re.compile(rf"(?<![\w-])(?:{FILLERS})(?![\w-]),?\s*", re.IGNORECASE)


def clean(text: str) -> str:
    """Remove filler words, keep meaning. Verbatim text is stored separately."""
    out = _filler_re.sub("", text)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    out = re.sub(r",+([.!?])", r"\1", out)
    return out[:1].upper() + out[1:] if re.search(r"\w", out) else text.strip()


def speaker_at(turns: list[dict], start: float, end: float) -> str | None:
    """Speaker with the largest overlap with [start, end]; nearest turn if none overlaps."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > best_overlap:
            best, best_overlap = t["speaker"], overlap
    if best is None and turns:
        mid = (start + end) / 2
        best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]
    return best


def build_segments(words: list[dict], turns: list[dict], max_pause: float = 1.5, max_words: int = 90) -> list[dict]:
    """Group ASR words into speaker turns.

    A new segment starts when the speaker changes, after a long pause,
    or when a segment grows too long and a sentence ends.
    """
    segments: list[dict] = []
    for w in words:
        spk = speaker_at(turns, w["start"], w["end"]) or "SPEAKER_00"
        cur = segments[-1] if segments else None
        split = (
            cur is None
            or cur["speaker"] != spk
            or w["start"] - cur["end"] > max_pause
            or (len(cur["words"]) >= max_words and cur["words"][-1]["word"].rstrip().endswith((".", "?", "!")))
        )
        if split:
            cur = {"speaker": spk, "start": w["start"], "end": w["end"], "words": []}
            segments.append(cur)
        cur["words"].append(w)
        cur["end"] = w["end"]
    for s in segments:
        s["text"] = "".join(w["word"] for w in s["words"]).strip()
        s["text_clean"] = clean(s["text"])
    return [s for s in segments if s["text"]]


def _ts(sec: float, sep: str) -> str:
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_srt(segments: list[dict]) -> str:
    return "\n".join(
        f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['speaker']}: {s['text']}\n"
        for i, s in enumerate(segments, 1)
    )


def to_vtt(segments: list[dict]) -> str:
    body = "\n".join(
        f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n<v {s['speaker']}>{s['text']}\n" for s in segments
    )
    return "WEBVTT\n\n" + body


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def _asks(text: str, name: str) -> bool:
    """«Игорь, сколько тебе нужно?»: the name opens the turn or a sentence, is followed by a comma
    and by a question of 3+ words — a question expects that person to answer next.
    Instructions («Марина, сделай макет.») do not count: after them someone else often speaks.
    A name at the very end of a turn is usually a word ASR timings pushed across a speaker change."""
    stem = re.escape(name[:-1] if len(name) >= 4 else name)  # Игорь / Игорю / Игоря
    return re.search(rf"(?:^|[.!?]\s+){stem}\w{{0,2}},(?:\s+[^\s.!?]+){{3,}}[^.!?]*\?", text, re.IGNORECASE) is not None


def verify_names(proposals: list[dict], turns: list[tuple[str, str]]) -> dict[str, str]:
    """Keep only speaker-name guesses the transcript itself supports.

    turns: (speaker label, text) in order. A guess «label is Name» gains support when someone else
    asks Name a question and `label` answers next (+1), or when `label` introduces themselves as Name (+2).
    It loses support when Name is asked and a different person answers (-1).
    Each label and each name ends up with at most one partner: the best-scored one.
    """
    scored = []
    for p in proposals:
        label, name = p["label"], p["name"].strip()
        if not name or len(name) > 40 or name.lower().startswith("спикер"):
            continue
        score = 0
        for i, (spk, text) in enumerate(turns):
            if spk == label and re.search(rf"(меня зовут|я\s*[—–-]?)\s+{re.escape(name)}(?!\w)", text, re.IGNORECASE):
                score += 2
            elif spk != label and i + 1 < len(turns) and _asks(text, name):
                nxt = turns[i + 1][0]
                score += 1 if nxt == label else -1 if nxt != spk else 0
        if score > 0:
            scored.append((score, label, name))
    accepted: dict[str, str] = {}
    for score, label, name in sorted(scored, reverse=True):
        rivals = [s for s, lab, n in scored if (lab == label) != (n == name) and (lab == label or n == name)]
        if label not in accepted and name not in accepted.values() and all(score > s for s in rivals):
            accepted[label] = name
    return accepted
