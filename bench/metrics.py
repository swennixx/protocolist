"""WER / CER and DER, dependency-free. Self-check: python bench/metrics.py"""

import re

import numpy as np


def normalize(text: str) -> str:
    """Lowercase, ё→е, digits spelled out, no punctuation — the form Golos references use."""
    from num2words import num2words

    t = text.lower().replace("ё", "е")
    t = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='ru')} ", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def edits(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token lists."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus WER: total word edits / total reference words."""
    e = sum(edits(normalize(r).split(), normalize(h).split()) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(normalize(r).split()) for r in refs))


def cer(refs: list[str], hyps: list[str]) -> float:
    e = sum(edits(list(normalize(r)), list(normalize(h))) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(normalize(r)) for r in refs))


def der(ref: list[dict], hyp: list[dict], collar: float = 0.25, step: float = 0.01) -> dict:
    """Diarization error rate for non-overlapping speech.

    ref/hyp: [{"speaker", "start", "end"}]. Frames within `collar` of a reference boundary are
    not scored (standard practice: boundaries are fuzzy even for human annotators).
    Speakers are matched one-to-one to maximise overlap (Hungarian algorithm).
    """
    from scipy.optimize import linear_sum_assignment

    end = max(t["end"] for t in ref + hyp)
    n = int(np.ceil(end / step)) + 1

    def frames(turns):
        labels = sorted({t["speaker"] for t in turns})
        arr = np.full(n, -1)
        for t in turns:
            arr[int(t["start"] / step):int(t["end"] / step)] = labels.index(t["speaker"])
        return arr, len(labels)

    r, nr = frames(ref)
    h, nh = frames(hyp)
    scored = np.ones(n, bool)
    for t in ref:
        for b in (t["start"], t["end"]):
            scored[max(0, int((b - collar) / step)):int((b + collar) / step)] = False
    r, h = r[scored], h[scored]
    speech = r >= 0
    overlap = np.zeros((nr, nh))
    np.add.at(overlap, (r[speech & (h >= 0)], h[speech & (h >= 0)]), 1)
    rows, cols = linear_sum_assignment(-overlap)
    correct = overlap[rows, cols].sum()
    missed = int((speech & (h < 0)).sum())
    false_alarm = int((~speech & (h >= 0)).sum())
    confusion = int((speech & (h >= 0)).sum() - correct)
    total = int(speech.sum())
    return {
        "der": round((missed + false_alarm + confusion) / total, 4),
        "missed": round(missed / total, 4),
        "false_alarm": round(false_alarm / total, 4),
        "confusion": round(confusion / total, 4),
        "speakers_ref": nr,
        "speakers_hyp": nh,
    }


if __name__ == "__main__":
    assert normalize("Релиз — 9 октября, Ёлка!") == "релиз девять октября елка"
    assert edits("a b c".split(), "a x c d".split()) == 2
    assert abs(wer(["раз два три четыре"], ["раз два три"]) - 0.25) < 1e-9
    ref = [{"speaker": "A", "start": 0, "end": 10}, {"speaker": "B", "start": 10, "end": 20}]
    perfect = [{"speaker": "x", "start": 0, "end": 10}, {"speaker": "y", "start": 10, "end": 20}]
    assert der(ref, perfect)["der"] == 0
    merged = [{"speaker": "x", "start": 0, "end": 20}]  # both people as one speaker: half is confusion
    d = der(ref, merged)
    assert abs(d["confusion"] - 0.5) < 0.01 and d["missed"] == 0, d
    print("metrics ok")
