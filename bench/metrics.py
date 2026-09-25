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


def der(ref: list[dict], hyp: list[dict], collar: float = 0.25, step: float = 0.01, uem: tuple[float, float] | None = None) -> dict:
    """Diarization error rate as in NIST md-eval, overlapping speech included.

    ref/hyp: [{"speaker", "start", "end"}]; turns of different speakers may overlap.
    Frames within `collar` of a reference boundary are not scored (boundaries are fuzzy even for
    human annotators); `uem` limits scoring to (start, end). Speakers are matched one-to-one to
    maximise overlap (Hungarian algorithm). In a frame with R reference and H hypothesis speakers:
    missed = max(0, R − H), false alarm = max(0, H − R), confusion = min(R, H) − correctly matched.
    """
    from scipy.optimize import linear_sum_assignment

    end = uem[1] if uem else max(t["end"] for t in ref + hyp)
    n = int(np.ceil(end / step)) + 1

    def matrix(turns):
        labels = sorted({t["speaker"] for t in turns})
        m = np.zeros((max(1, len(labels)), n), bool)
        for t in turns:
            m[labels.index(t["speaker"]), int(t["start"] / step):int(t["end"] / step)] = True
        return m, len(labels)

    r, nr = matrix(ref)
    h, nh = matrix(hyp)
    scored = np.ones(n, bool)
    if uem:
        scored[: int(uem[0] / step)] = False
    for t in ref:
        for b in (t["start"], t["end"]):
            scored[max(0, int((b - collar) / step)):int((b + collar) / step)] = False
    r, h = r[:, scored], h[:, scored]
    rc, hc = r.sum(0), h.sum(0)
    overlap = r.astype(np.int32) @ h.T.astype(np.int32)
    rows, cols = linear_sum_assignment(-overlap)
    correct = overlap[rows, cols].sum()
    total = int(rc.sum())
    missed = int(np.maximum(0, rc - hc).sum())
    false_alarm = int(np.maximum(0, hc - rc).sum())
    confusion = int(np.minimum(rc, hc).sum() - correct)
    return {
        "der": round((missed + false_alarm + confusion) / total, 4),
        "missed": round(missed / total, 4),
        "false_alarm": round(false_alarm / total, 4),
        "confusion": round(confusion / total, 4),
        "overlap_share": round(float((rc > 1).sum() / max(1, (rc > 0).sum())), 4),
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
    # Overlap: both speak 5–10 s; a single-speaker hypothesis misses one of them there.
    ov_ref = [{"speaker": "A", "start": 0, "end": 10}, {"speaker": "B", "start": 5, "end": 20}]
    d = der(ov_ref, perfect, collar=0)
    assert abs(d["missed"] - 5 / 25) < 0.01 and d["false_alarm"] == 0, d
    print("metrics ok")
