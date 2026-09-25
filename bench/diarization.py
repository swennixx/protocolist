"""Diarization on the synthetic meetings (bench/make_synthetic.py): DER for pyannote and the built-in
clustering, with and without the speaker count, plus WER of the whole pipeline text.

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


def main(names=("sfera", "client")) -> None:
    from app import ml
    from app.text import build_segments

    backends = ["clustering"] + (["pyannote"] if ml.diarization_backend() == "pyannote" else [])
    out = {}
    for name in names:
        ref = json.load(open(os.path.join(HERE, "data", f"{name}.json"), encoding="utf-8"))
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "a.wav")
        ml.prepare_audio(os.path.join(HERE, "data", f"{name}.wav"), wav, os.path.join(tmp, "a.m4a"))
        t0 = time.perf_counter()
        asr = ml.transcribe(wav, "large-v3-turbo", "ru", None, lambda f: None)
        res = {
            "audio_seconds": round(ref["turns"][-1]["end"], 1),
            "speakers": ref["speakers"],
            "asr_seconds": round(time.perf_counter() - t0, 1),
            "wer": round(wer([" ".join(t["text"] for t in ref["turns"])], ["".join(w["word"] for w in asr["words"])]), 4),
        }
        for backend in backends:
            for label, num in (("auto", None), ("oracle_count", ref["speakers"])):
                t0 = time.perf_counter()
                turns, _ = ml.diarize(wav, num, lambda f: None, asr["words"], backend=backend)
                spent = time.perf_counter() - t0
                # Score what the user sees: speaker turns after aligning words to diarization.
                segs = build_segments(asr["words"], turns)
                hyp = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"]} for s in segs]
                res[f"{backend}_{label}"] = {**der(ref["turns"], hyp), "seconds": round(spent, 1)}
                print(name, backend, label, res[f"{backend}_{label}"], flush=True)
        print(name, "wer", res["wer"], flush=True)
        out[name] = res
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "diarization.json"), "w"), indent=1)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("sfera", "client"))
