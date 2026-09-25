"""Search quality on the synthetic meeting: full-text vs vector vs hybrid, several embedding models.

Uses the reference transcript (bench/data/sfera.json), so ASR errors do not blur retrieval numbers.
Full-text ranking runs in Postgres exactly as in the app. Needs the DB from docker-compose.

    python bench/retrieval.py [model ...]
"""

import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "0"  # the bench may download candidate models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import engine  # noqa: E402
from app.search import FTS_BONUS  # noqa: E402

RRF_K = 60

HERE = os.path.dirname(os.path.abspath(__file__))

# question -> indices of reference turns that answer it
QUERIES = {
    "когда выпускаем приложение": [8],
    "почему сдвинули релиз": [4, 6, 7, 8],
    "сколько денег на рекламу": [12],
    "кто делает новый онбординг": [10, 11],
    "проблемы с API часов": [1],
    "сколько времени нужно на доработку синхронизации": [7],
    "доступ к магазину приложений": [14, 15],
    "когда стартует рекламная кампания": [9],
    "что решили с блогерами": [13],
    "сколько экранов в онбординге": [10, 11],
    "до какого числа отправить сборку на проверку": [15],
    "придётся переснимать ролики": [4],
    "платежи готовы?": [1],
    "показать функцию как бету": [5],
}

MODELS = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "deepvk/USER-bge-m3": ("", ""),
}


def fts_scores(docs: list[str], q: str) -> dict[int, float]:
    tsq = "to_tsquery('russian', replace(plainto_tsquery('russian', :q)::text, '&', '|'))"
    with engine.connect() as c:
        rows = c.execute(
            text(
                f"select i, ts_rank_cd(to_tsvector('russian', d), {tsq}) r "
                "from unnest(cast(:docs as text[])) with ordinality as t(d, i) "
                f"where to_tsvector('russian', d) @@ {tsq}"
            ),
            {"q": q, "docs": docs},
        ).all()
    return {int(i) - 1: float(r) for i, r in rows}


def fuse(sims: np.ndarray, fts: dict[int, float], beta: float) -> list[int]:
    """Cosine similarity plus a bonus for word matches, scaled to the best word match."""
    top = max(fts.values(), default=1) or 1
    score = sims + beta * np.array([fts.get(i, 0) / top for i in range(len(sims))])
    return list(np.argsort(-score))


def fts_rank(docs: list[str], q: str, any_word: bool = True) -> list[int]:
    """any_word=True mirrors the app: OR of the query's lexemes, ranked by ts_rank_cd."""
    tsq = "to_tsquery('russian', replace(plainto_tsquery('russian', :q)::text, '&', '|'))" if any_word else "websearch_to_tsquery('russian', :q)"
    with engine.connect() as c:
        rows = c.execute(
            text(
                f"select i, ts_rank_cd(to_tsvector('russian', d), {tsq}) r "
                "from unnest(cast(:docs as text[])) with ordinality as t(d, i) "
                f"where to_tsvector('russian', d) @@ {tsq} order by r desc"
            ),
            {"q": q, "docs": docs},
        ).all()
    return [int(i) - 1 for i, _ in rows]


def rrf(*rankings: list[int], weights: tuple[float, ...] | None = None) -> list[int]:
    score: dict[int, float] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0) + w / (RRF_K + r + 1)
    return sorted(score, key=score.get, reverse=True)


def metrics(results: dict[str, list[int]]) -> dict[str, float]:
    r1 = r3 = mrr = 0.0
    for q, ranking in results.items():
        gold = set(QUERIES[q])
        first = next((r for r, i in enumerate(ranking[:10]) if i in gold), None)
        r1 += first == 0
        r3 += first is not None and first < 3
        mrr += 1 / (first + 1) if first is not None else 0
    n = len(results)
    return {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3), "mrr@10": round(mrr / n, 3)}


def main() -> None:
    from sentence_transformers import SentenceTransformer

    ref = json.load(open(os.path.join(HERE, "data", "sfera.json"), encoding="utf-8"))
    docs = [f"{t['speaker']}: {t['text']}" for t in ref["turns"]]
    texts = [t["text"] for t in ref["turns"]]
    fts_all = {q: fts_rank(texts, q, any_word=False) for q in QUERIES}
    fts = {q: fts_rank(texts, q) for q in QUERIES}
    out = {"fts_all_words": metrics(fts_all), "fts": metrics(fts)}
    print(f"{'fts (all words)':45} {out['fts_all_words']}")
    print(f"{'fts (any word)':45} {out['fts']}")
    for name in sys.argv[1:] or MODELS:
        qp, dp = MODELS.get(name, ("", ""))
        model = SentenceTransformer(name, device="cpu")
        t0 = time.time()
        d = model.encode([dp + x for x in docs], normalize_embeddings=True)
        qv = model.encode([qp + q for q in QUERIES], normalize_embeddings=True)
        ms = (time.time() - t0) / (len(docs) + len(QUERIES)) * 1000
        vec = {q: list(np.argsort(-(d @ v))) for q, v in zip(QUERIES, qv)}
        fs = {q: fts_scores(texts, q) for q in QUERIES}
        sims = {q: d @ v for q, v in zip(QUERIES, qv)}
        hyb = {q: fuse(sims[q], fs[q], FTS_BONUS) for q in QUERIES}
        print(f"   rrf: {metrics({q: rrf(fts[q], vec[q]) for q in QUERIES})}")
        for b in (0.02, 0.05, 0.1, 0.2):
            print(f"   score fusion beta {b}: {metrics({q: fuse(sims[q], fs[q], b) for q in QUERIES})}")
        out[name] = {"vector": metrics(vec), "hybrid": metrics(hyb), "ms_per_text": round(ms, 1)}
        if "-v" in os.environ.get("BENCH_FLAGS", ""):
            for q in QUERIES:
                print(f"   {q[:40]:42} vec {vec[q][:3]} fts {fts[q][:3]} hyb {hyb[q][:3]} gold {QUERIES[q]}")
        print(f"{name:45} vector {out[name]['vector']}  hybrid {out[name]['hybrid']}  {ms:.1f} ms/text")
        del model
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
