"""Speech recognition quality and speed on real Russian speech (SberDevices Golos test sets).

    python bench/asr.py --n 200 [--models tiny small medium large-v3-turbo large-v3]

Downloads: see README (bench section). Results: bench/results/asr.json
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from metrics import cer, normalize, wer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "golos_farfield": "data/golos/sberdevices_golos_100h_farfield/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet",
    "golos_crowd": "data/golos/sberdevices_golos_10h_crowd/data/test-00000-of-00003-cad0cfaddbc8fa71.parquet",
}


def load_set(path: str, n: int) -> list[tuple[np.ndarray, str]]:
    table = pq.read_table(os.path.join(HERE, path))
    texts = table.column("transcription").to_pylist()
    valid = [i for i, t in enumerate(texts) if t and t.strip()]  # a few rows have no reference
    idx = [valid[int(k)] for k in np.linspace(0, len(valid) - 1, n)]  # spread over the file: many speakers
    out = []
    for i in idx:
        row = table.slice(int(i), 1).to_pylist()[0]
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, sr
        out.append((audio, row["transcription"]))
    return out


def main() -> None:
    import mlx_whisper

    from app.ml import MLX_REPOS

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=list(MLX_REPOS))
    ap.add_argument("--duty", type=float, default=1.0, help="share of time to keep the GPU busy; 0.5 = pause as long as each file took (quieter fans)")
    ap.add_argument("--redo", action="store_true", help="recompute results that already exist")
    args = ap.parse_args()

    path = os.path.join(HERE, "results", "asr.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    data = {name: load_set(p, args.n) for name, p in SETS.items()}
    for model in args.models:
        repo = MLX_REPOS[model]
        todo = [name for name in data if args.redo or model not in results.get(name, {})]
        if not todo:
            continue
        mlx_whisper.transcribe(data["golos_crowd"][0][0], path_or_hf_repo=repo, language="ru")  # load weights, not timed
        for name in todo:
            items = data[name]
            hyps, spent = [], 0.0
            for audio, _ in items:
                t0 = time.perf_counter()
                r = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language="ru", condition_on_previous_text=False)
                took = time.perf_counter() - t0
                spent += took  # RTF counts only work, not the pauses below
                hyps.append(r["text"])
                if args.duty < 1:
                    time.sleep(took * (1 / args.duty - 1))
            refs = [ref for _, ref in items]
            audio_sec = sum(len(a) for a, _ in items) / 16000
            worst = sorted(zip(refs, hyps), key=lambda p: -wer([p[0]], [p[1]]))[:3]
            results.setdefault(name, {})[model] = {
                "wer": round(wer(refs, hyps), 4),
                "cer": round(cer(refs, hyps), 4),
                "rtf": round(spent / audio_sec, 4),
                "utterances": len(items),
                "audio_minutes": round(audio_sec / 60, 1),
                "examples": [{"ref": r, "hyp": normalize(h)} for r, h in worst],
            }
            print(f"{name:15} {model:15} WER {results[name][model]['wer']:.3f}  CER {results[name][model]['cer']:.3f}  RTF {results[name][model]['rtf']:.3f}", flush=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(results, open(path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
