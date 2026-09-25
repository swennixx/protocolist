"""Markdown tables for the README from bench/results/*.json.

    python bench/report.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def load(name: str) -> dict:
    path = os.path.join(R, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def main() -> None:
    out = []
    asr = load("asr.json")
    if asr:
        models = list(next(iter(asr.values())))
        n = next(iter(next(iter(asr.values())).values()))["utterances"]
        out += [
            f"**Распознавание речи** — Whisper на тестовых частях [Golos](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield) (SberDevices), по {n} фраз из каждой: farfield — дальний микрофон, как в переговорке; crowd — фразы, записанные людьми на свои устройства. WER — доля ошибок в словах (меньше — лучше), CER — в буквах. RTF — время обработки / длительность записи (0,1 = минута записи за 6 секунд).",
            "",
            "| Модель | WER farfield | WER crowd | CER farfield | RTF farfield |",
            "|---|---|---|---|---|",
        ]
        for m in models:
            f, c = asr["golos_farfield"].get(m), asr.get("golos_crowd", {}).get(m)
            if not f:
                continue
            rtf = f["rtf"]
            bold = "**" if m == "large-v3-turbo" else ""
            out.append(f"| {bold}{m}{bold} | {pct(f['wer'])} | {pct(c['wer']) if c else '—'} | {pct(f['cer'])} | {rtf:.3f}".replace(".", ",") + " |")
        if any(m not in asr.get("golos_crowd", {}) for m in asr["golos_farfield"]):
            out += ["", "«—» — ещё не посчитано: `python bench/asr.py --models large-v3 --duty 0.5`."]
        t, l = asr["golos_farfield"].get("large-v3-turbo"), asr["golos_farfield"].get("large-v3")
        if t and l:
            speedup = str(round(l["rtf"] / t["rtf"], 1)).replace(".", ",")
            out += ["", f"По умолчанию стоит large-v3-turbo: на farfield WER {pct(t['wer'])} против {pct(l['wer'])} у large-v3, а работает в {speedup} раза быстрее."]
        d = load("diarization.json")
        if d:
            rtf = str(round(d["asr_seconds"] / d["audio_seconds"], 2)).replace(".", ",")
            out += ["", f"RTF в таблице завышен для всех моделей: каждая фраза Golos длится 2–5 с, а Whisper обрабатывает окно в 30 с. На сплошной записи встречи turbo работает с RTF {rtf} ({d['audio_seconds']:.0f} с записи за {d['asr_seconds']:.0f} с, вместе с загрузкой модели)."]
        out.append("")

    d = load("diarization.json")
    if d:
        a, o = d["auto"], d["oracle_count"]
        out += [
            f"**Диаризация** — синтетическая встреча из 16 реплик 4 спикеров ({d['audio_seconds']:.0f} с, `bench/make_synthetic.py`), встроенный алгоритм, DER с допуском 0,25 с на границах:",
            "",
            "| Число спикеров | Найдено | DER | Из них путаница спикеров |",
            "|---|---|---|---|",
            f"| определяется автоматически | {a['speakers_hyp']} из {a['speakers_ref']} | {pct(a['der'])} | {pct(a['confusion'])} |",
            f"| задано пользователем | {o['speakers_hyp']} из {o['speakers_ref']} | {pct(o['der'])} | {pct(o['confusion'])} |",
            "",
            f"WER всего пайплайна на этой записи — {pct(d['wer'])}. Синтетика — один голос TTS с разным тоном: это проверка на регрессии, а не оценка на реальных встречах.",
            "",
        ]

    s = load("retrieval.json")
    if s:
        e5 = s.get("intfloat/multilingual-e5-small", {})
        out += [
            "**Поиск** — 14 вопросов своими словами к той же встрече, у каждого известен правильный ответ (`bench/retrieval.py`). Recall@1 — правильная реплика на первом месте, Recall@3 — в первой тройке.",
            "",
            "| Способ | Recall@1 | Recall@3 | MRR |",
            "|---|---|---|---|",
            f"| Полнотекстовый, все слова запроса | {pct(s['fts_all_words']['recall@1'])} | {pct(s['fts_all_words']['recall@3'])} | {s['fts_all_words']['mrr@10']:.2f} |".replace("0.", "0,"),
            f"| Полнотекстовый, любое слово | {pct(s['fts']['recall@1'])} | {pct(s['fts']['recall@3'])} | {s['fts']['mrr@10']:.2f} |".replace("0.", "0,"),
        ]
        for name, label in (("intfloat/multilingual-e5-small", "e5-small"), ("intfloat/multilingual-e5-base", "e5-base"),
                            ("intfloat/multilingual-e5-large", "e5-large"), ("deepvk/USER-bge-m3", "USER-bge-m3")):
            if name in s:
                v = s[name]["vector"]
                out.append(f"| Векторный, {label} ({str(s[name]['ms_per_text']).replace('.', ',')} мс/текст) | {pct(v['recall@1'])} | {pct(v['recall@3'])} | {v['mrr@10']:.2f} |".replace("0.", "0,"))
        if e5:
            h = e5["hybrid"]
            out.append(f"| **Векторный e5-small + бонус за слова (в приложении)** | {pct(h['recall@1'])} | {pct(h['recall@3'])} | {h['mrr@10']:.2f} |".replace("0.", "0,"))
        if e5:
            others = [s[n] for n in ("intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "deepvk/USER-bge-m3") if n in s]
            best_other = max(o["vector"]["mrr@10"] for o in others) if others else 0
            slow = [o["ms_per_text"] / e5["ms_per_text"] for o in others]
            if others and best_other <= e5["vector"]["mrr@10"] + 0.01:
                out += ["", f"Модели крупнее e5-small на этом наборе не выиграли, а работают в {min(slow):.0f}–{max(slow):.0f} раз медленнее, поэтому в приложении стоит e5-small. Набор маленький: вывод предварительный."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
