"""Diarization on real meetings: AMI Meeting Corpus test set (English; diarization does not depend
on the language). Reference speaker turns: BUTSpeechFIT/AMI-diarization-setup (only_words).

    python bench/ami.py [--minutes 15] [--cool 1.0]

Downloads (≈180 MB) into bench/data/ami — see the curl commands in README. Scores the first
`--minutes` of each meeting with standard DER: 0.25 s collar, overlapping speech included.
`--cool` sleeps that share of the working time after each meeting to keep the laptop quiet.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from metrics import der  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "ami")
MEETINGS = ["ES2004a", "IS1009a", "TS3003a", "EN2002a"]


def read_rttm(path: str) -> list[dict]:
    turns = []
    for line in open(path):
        f = line.split()
        start, dur = float(f[3]), float(f[4])
        turns.append({"speaker": f[7], "start": start, "end": start + dur})
    return turns


def main() -> None:
    from app import ml
    from app.text import build_segments

    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--cool", type=float, default=1.0)
    ap.add_argument("--backends", nargs="+", default=None, help="clustering and/or pyannote (default: all available)")
    args = ap.parse_args()
    limit = args.minutes * 60

    backends = args.backends or ["clustering"] + (["pyannote"] if ml.diarization_backend() == "pyannote" else [])
    path = os.path.join(HERE, "results", "ami.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    tmp = tempfile.mkdtemp()
    for m in MEETINGS:
        t_meeting = time.perf_counter()
        wav = os.path.join(tmp, f"{m}.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(DATA, f"{m}.wav"), "-t", str(limit),
                        "-ac", "1", "-ar", "16000", wav], check=True)
        uem_start, uem_end = map(float, open(os.path.join(DATA, f"{m}.uem")).read().split()[2:4])
        uem = (uem_start, min(uem_end, limit))
        ref = [t for t in read_rttm(os.path.join(DATA, f"{m}.rttm")) if t["start"] < uem[1]]
        ref = [{**t, "end": min(t["end"], uem[1])} for t in ref]
        n_ref = len({t["speaker"] for t in ref})

        cache = os.path.join(DATA, f"{m}.{int(limit)}s.asr.json")  # ASR is the slow part: compute once
        if os.path.exists(cache):
            words = json.load(open(cache))
        else:
            words = ml.transcribe(wav, "large-v3-turbo", "en", None, lambda f: None)["words"]
            json.dump(words, open(cache, "w"))

        res = results.setdefault(m, {"minutes": round(uem[1] / 60, 1), "speakers": n_ref})
        for backend in backends:
            for label, num in (("auto", None), ("oracle_count", n_ref)):
                t0 = time.perf_counter()
                turns, _ = ml.diarize(wav, num, lambda f: None, words, backend=backend)
                spent = time.perf_counter() - t0
                raw = der(ref, turns, uem=uem)  # the diarizer itself (pyannote output may overlap)
                segs = build_segments(words, turns)  # what the user sees: words assigned to speakers
                shown = der(ref, [{"speaker": s["speaker"], "start": s["start"], "end": s["end"]} for s in segs], uem=uem)
                res[f"{backend}_{label}"] = {"diarizer": raw, "transcript": shown, "seconds": round(spent, 1)}
                print(f"{m} {backend:10} {label:12} DER {raw['der']:.3f} (transcript {shown['der']:.3f})"
                      f" speakers {raw['speakers_hyp']}/{n_ref} overlap {raw['overlap_share']:.2f} {spent:.0f}s", flush=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(results, open(path, "w"), indent=1)
        time.sleep((time.perf_counter() - t_meeting) * args.cool)


if __name__ == "__main__":
    main()
