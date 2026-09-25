"""Diarization on the synthetic 4-speaker meeting (bench/make_synthetic.py): DER with and without
the speaker count, plus WER of the whole pipeline text.

    python bench/diarization.py

The synthetic voices are one TTS voice with different pitch — harder to separate than real people
in timbre, easier in acoustics (no noise, no overlap). Treat the numbers as a regression check,
not as a real-world estimate.
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from metrics import der, wer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(name: str = "sfera") -> None:
    from app import ml
    from app.text import build_segments

    ref = json.load(open(os.path.join(HERE, "data", f"{name}.json"), encoding="utf-8"))
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    ml.prepare_audio(os.path.join(HERE, "data", f"{name}.wav"), wav, os.path.join(tmp, "a.m4a"))

    t0 = time.perf_counter()
    asr = ml.transcribe(wav, "large-v3-turbo", "ru", None, lambda f: None)
    asr_time = time.perf_counter() - t0
    out = {
        "audio_seconds": round(ref["turns"][-1]["end"], 1),
        "asr_seconds": round(asr_time, 1),
        "wer": round(wer([" ".join(t["text"] for t in ref["turns"])], ["".join(w["word"] for w in asr["words"])]), 4),
    }
    for label, num in (("auto", None), ("oracle_count", ref["speakers"])):
        t0 = time.perf_counter()
        turns, backend = ml.diarize(wav, num, lambda f: None, asr["words"])
        spent = time.perf_counter() - t0
        # Score what the user sees: speaker turns after aligning words to diarization.
        segs = build_segments(asr["words"], turns)
        hyp = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"]} for s in segs]
        out[label] = {**der(ref["turns"], hyp), "backend": backend, "seconds": round(spent, 1)}
        print(label, out[label], flush=True)
    print("wer", out["wer"], "asr", out["asr_seconds"], "s")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "diarization.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
