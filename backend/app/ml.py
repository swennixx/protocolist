"""Model wrappers. Models load lazily once per worker process."""

import os
import platform
import subprocess
import wave
from functools import cache
from typing import Callable

import numpy as np

Progress = Callable[[float], None]
SR = 16000
USE_MLX = platform.system() == "Darwin" and platform.machine() == "arm64"

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


# --- audio ---

def prepare_audio(src: str, wav: str, playback: str) -> float:
    """Decode any audio/video into 16 kHz mono WAV for models and AAC for the player."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
         "-af", "loudnorm=I=-20:TP=-2", "-c:a", "pcm_s16le", wav],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", playback],
        check=True,
    )
    with wave.open(wav) as f:
        return f.getnframes() / f.getframerate()


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def peaks(audio: np.ndarray, bars: int = 180) -> list[float]:
    """RMS per bar, normalised to 0..1, for the waveform in the UI."""
    chunks = np.array_split(audio, bars)
    rms = np.array([float(np.sqrt(np.mean(c**2))) if len(c) else 0 for c in chunks])
    top = rms.max() or 1
    return [round(float(x / top), 3) for x in rms]


# --- speech recognition ---

def transcribe(wav: str, model: str, language: str | None, prompt: str | None, progress: Progress) -> dict:
    """Returns {"language", "words": [{"word", "start", "end", "prob"}]}."""
    if USE_MLX:
        import sys

        import mlx_whisper

        mt = sys.modules["mlx_whisper.transcribe"]  # the package re-exports a function with the same name

        class Bar(mt.tqdm.tqdm):  # mlx-whisper reports progress only through tqdm
            def update(self, n=1):
                super().update(n)
                if self.total:
                    progress(self.n / self.total)

        orig, mt.tqdm.tqdm = mt.tqdm.tqdm, Bar
        try:
            r = mlx_whisper.transcribe(
                wav, path_or_hf_repo=MLX_REPOS[model], language=language, initial_prompt=prompt,
                # No hallucination_silence_threshold: on bench/data/client it dropped two real turns (WER 0.6 % → 5.2 %).
                word_timestamps=True, verbose=False, condition_on_previous_text=False,
            )
        finally:
            mt.tqdm.tqdm = orig
        words = [w for s in r["segments"] for w in s.get("words", [])]
        return {
            "language": r["language"],
            "words": [{"word": w["word"], "start": w["start"], "end": w["end"], "prob": w["probability"]} for w in words],
        }

    fw = _faster_whisper(model)
    segments, info = fw.transcribe(wav, language=language, initial_prompt=prompt, word_timestamps=True, vad_filter=True)
    words = []
    for s in segments:
        words += [{"word": w.word, "start": w.start, "end": w.end, "prob": w.probability} for w in s.words or []]
        progress(min(1, s.end / info.duration))
    return {"language": info.language, "words": words}


@cache
def _faster_whisper(model: str):
    from faster_whisper import WhisperModel

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    return WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")


# --- diarization ---
# Preferred: the full pyannote pipeline (gated on Hugging Face: accept its terms, set HF_TOKEN).
# Fallback: speech windows from ASR word timings -> WeSpeaker voice embeddings (open model) -> clustering.

DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
SPEAKER_EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_EMBED = 0.5  # s, shorter phrases get no embedding
MIN_RELIABLE = 1.5  # s, only phrases this long take part in clustering
MIN_SILHOUETTE = 0.15  # below this the recording is treated as one speaker
SHORT_MARGIN = 0.15  # a short phrase leaves its neighbour's speaker only on a clear voice match


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@cache
def _pipeline():
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.environ.get("HF_TOKEN") or None)
    pipe.to(_device())
    return pipe


@cache
def _speaker_model():
    from pyannote.audio import Model

    model = Model.from_pretrained(SPEAKER_EMBEDDING, token=os.environ.get("HF_TOKEN") or None)
    return model.eval().to(_device())


def pyannote_downloaded() -> bool:
    """Cheap check for the API: is the gated pipeline in the local Hugging Face cache?"""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(DIARIZATION_MODEL, "config.yaml"), str)


@cache  # decided once per worker process; restart the worker after getting access to pyannote
def diarization_backend() -> str:
    """'pyannote' when the gated pipeline is reachable, else 'clustering'."""
    try:
        _pipeline()
        return "pyannote"
    except Exception:
        return "clustering"


def diarize(wav: str, num_speakers: int | None, progress: Progress, words: list[dict], backend: str | None = None) -> tuple[list[dict], str]:
    """Returns (turns, backend). Turns: [{"speaker", "start", "end"}].
    backend: None picks pyannote when available; "clustering" / "pyannote" force one (bench)."""
    audio = load_wav(wav)
    if (backend or diarization_backend()) == "pyannote":
        return _diarize_pyannote(audio, num_speakers, progress), "pyannote"
    return _diarize_clustering(audio, num_speakers, progress, words), "clustering"


def _diarize_pyannote(audio: np.ndarray, num_speakers: int | None, progress: Progress) -> list[dict]:
    import torch

    steps = {"segmentation": 0.0, "speaker_counting": 0.3, "embeddings": 0.35, "discrete_diarization": 0.95}

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        base = steps.get(step_name)
        if base is None:
            return
        span = {"segmentation": 0.3, "embeddings": 0.6}.get(step_name, 0.05)
        frac = (completed / total) if total and completed is not None else 1
        progress(min(1, base + span * frac))

    out = _pipeline()({"waveform": torch.from_numpy(audio)[None], "sample_rate": SR}, num_speakers=num_speakers, hook=hook)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    return [{"speaker": spk, "start": round(t.start, 3), "end": round(t.end, 3)} for t, _, spk in ann.itertracks(yield_label=True)]


FRAME = 0.02  # s, energy frame for voice activity


def voiced_frames(audio: np.ndarray) -> np.ndarray:
    """Boolean mask per 20 ms frame: louder than a tenth of typical speech level."""
    n = int(FRAME * SR)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(1)) + 1e-9
    return rms >= 0.1 * np.percentile(rms, 90)


def silences(voiced: np.ndarray, min_len: float = 0.25) -> list[float]:
    """Centres of pauses at least `min_len` long."""
    out, run = [], 0
    for i, v in enumerate(np.append(voiced, True)):
        if not v:
            run += 1
        elif run:
            if run * FRAME >= min_len:
                out.append((i - run / 2) * FRAME)
            run = 0
    return out


def phrases(words: list[dict], pauses: list[float] = (), gap: float = 0.3) -> list[tuple[float, float]]:
    """Split speech into phrases at pauses and sentence ends: speaker changes almost always fall there.

    Whisper word timings often stretch over short pauses, so pauses found in the signal
    (between the midpoints of two neighbouring words) split phrases too.
    """
    pauses = sorted(pauses)
    out: list[list[float]] = []
    prev = None
    for w in words:
        split = (
            prev is None
            or w["start"] - prev["end"] > gap
            or prev["word"].rstrip().endswith((".", "?", "!"))
            or _any_between(pauses, (prev["start"] + prev["end"]) / 2, (w["start"] + w["end"]) / 2)
        )
        if split:
            out.append([w["start"], w["end"]])
        else:
            out[-1][1] = w["end"]
        prev = w
    return [(s, e) for s, e in out if e > s]


def _any_between(sorted_xs: list[float], lo: float, hi: float) -> bool:
    import bisect

    i = bisect.bisect_right(sorted_xs, lo)
    return i < len(sorted_xs) and sorted_xs[i] < hi


def merge_turns(spans: list[tuple[float, float]], labels: list[int], max_gap: float = 0.5) -> list[dict]:
    """Neighbouring phrases of one speaker become one turn — but not across a real pause:
    that silence is not speech (on AMI, bridging pauses made false alarms 25–39 % of speech)."""
    turns: list[dict] = []
    for (s, e), lab in zip(spans, labels):
        spk = f"SPEAKER_{lab:02d}"
        if turns and turns[-1]["speaker"] == spk and s - turns[-1]["end"] <= max_gap:
            turns[-1]["end"] = round(e, 3)
        else:
            turns.append({"speaker": spk, "start": round(s, 3), "end": round(e, 3)})
    return turns


def cluster(emb: np.ndarray, num_speakers: int | None, max_speakers: int = 10) -> np.ndarray:
    """Agglomerative clustering (average linkage, cosine). Without a known count, pick the count with
    the best silhouette; a weak best silhouette means everyone is one speaker."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score

    if len(emb) < 3:
        return np.zeros(len(emb), dtype=int)
    z = linkage(emb, method="average", metric="cosine")
    if num_speakers:
        return fcluster(z, num_speakers, "maxclust") - 1
    best, best_k = -1.0, 1
    for k in range(2, min(max_speakers, len(emb) - 1) + 1):
        lab = fcluster(z, k, "maxclust")
        if len(set(lab)) < 2:
            continue
        score = silhouette_score(emb, lab, metric="cosine")
        if score > best:
            best, best_k = score, k
    if best < MIN_SILHOUETTE:
        return np.zeros(len(emb), dtype=int)
    return fcluster(z, best_k, "maxclust") - 1


def label_phrases(emb: np.ndarray, spans: list[tuple[float, float]], voiced_len: np.ndarray, num_speakers: int | None) -> list[int]:
    """Cluster reliable (long) phrases, then label the rest.

    Short phrases carry little voice: they take the label of the nearer neighbouring phrase
    unless their own embedding clearly points to another speaker.
    """
    ok = ~np.isnan(emb).any(1)
    reliable = ok & (voiced_len >= MIN_RELIABLE)
    if reliable.sum() < 3:
        reliable = ok
    labels = np.full(len(emb), -1)
    if not reliable.any():
        return [0] * len(emb)
    labels[reliable] = cluster(emb[reliable], num_speakers)
    cents = np.stack([emb[labels == k].mean(0) for k in range(labels.max() + 1)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)

    def neighbour(i: int) -> int:
        prev = next((j for j in range(i - 1, -1, -1) if labels[j] >= 0), None)
        nxt = next((j for j in range(i + 1, len(labels)) if labels[j] >= 0), None)
        if prev is None or (nxt is not None and spans[nxt][0] - spans[i][1] < spans[i][0] - spans[prev][1]):
            return int(labels[nxt])
        return int(labels[prev])

    for i in np.flatnonzero(~reliable):  # in time order, so earlier short phrases can serve as neighbours
        n = neighbour(i)
        if ok[i]:
            sims = cents @ emb[i]
            best = int(np.argmax(sims))
            labels[i] = best if sims[best] - sims[n] > SHORT_MARGIN else n
        else:
            labels[i] = n
    order = list(dict.fromkeys(labels.tolist()))  # «Спикер 1» is whoever speaks first
    return [order.index(k) for k in labels.tolist()]


def _diarize_clustering(audio: np.ndarray, num_speakers: int | None, progress: Progress, words: list[dict]) -> list[dict]:
    import torch

    voiced = voiced_frames(audio)
    spans = phrases(words, silences(voiced))
    if not spans:
        return []
    n = int(FRAME * SR)
    model, dim, rows, lengths = _speaker_model(), None, [], []
    with torch.inference_mode():
        for i, (s, e) in enumerate(spans):
            # Whisper stretches word timings into pauses; embed only the voiced frames of the phrase.
            f0, f1 = int(s / FRAME), int(e / FRAME) + 1
            idx = np.flatnonzero(voiced[f0:f1]) + f0
            clip = audio[(idx[:, None] * n + np.arange(n)).ravel()] if len(idx) else audio[:0]
            lengths.append(len(clip) / SR)
            if len(clip) < MIN_EMBED * SR:
                rows.append(None)
                continue
            # Long phrases: average of ≤10 s pieces keeps memory flat.
            pieces = [clip[j:j + 10 * SR] for j in range(0, len(clip), 10 * SR) if len(clip[j:j + 10 * SR]) >= MIN_EMBED * SR]
            vec = np.mean([model(torch.from_numpy(p)[None, None].to(_device())).cpu().numpy()[0] for p in pieces], 0)
            rows.append(vec / (np.linalg.norm(vec) + 1e-9))
            dim = len(vec)
            if i % 20 == 0:
                progress(0.9 * i / len(spans))
    if dim is None:
        return merge_turns(spans, [0] * len(spans))
    emb = np.stack([r if r is not None else np.full(dim, np.nan) for r in rows])
    return merge_turns(spans, label_phrases(emb, spans, np.array(lengths), num_speakers))


# --- embeddings ---

EMBED_MODEL = "intfloat/multilingual-e5-small"


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    prefix = "query: " if query else "passage: "  # e5 is trained with these prefixes
    vecs = _embedder().encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]
