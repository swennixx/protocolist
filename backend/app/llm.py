"""Local LLM through Ollama: meeting analysis and archive Q&A."""

import json
import os
from datetime import date
from typing import Iterator

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = 32768
WINDOW, OVERLAP = 40, 5  # transcript lines per extraction call
MAP_CHARS = 18000  # ~6k tokens: longer transcripts get a map-reduce overview

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

TEMPLATES = {
    "short": "3–5 пунктов, по одному предложению, только главное",
    "full": "6–10 пунктов: ход обсуждения, аргументы сторон, цифры и итоги",
    "protocol": "5–8 пунктов в стиле официального протокола: повестка, обсуждали, постановили",
}


def _obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


STR, INT, OPT_STR = {"type": "string"}, {"type": "integer"}, {"type": ["string", "null"]}

OVERVIEW_SCHEMA = _obj(
    summary={"type": "array", "items": STR},
    topics={"type": "array", "items": _obj(title=STR, seg=INT)},
    people={"type": "array", "items": STR},
)

MAP_SCHEMA = _obj(
    points={"type": "array", "items": STR},
    topics={"type": "array", "items": _obj(title=STR, seg=INT)},
    people={"type": "array", "items": STR},
)

REDUCE_SCHEMA = _obj(
    summary={"type": "array", "items": STR},
    topics={"type": "array", "items": _obj(title=STR, seg=INT)},
)

ITEMS_SCHEMA = _obj(
    items={
        "type": "array",
        "items": _obj(seg=INT, quote=STR, kind={"type": "string", "enum": ["decision", "task"]}, text=STR, owner=OPT_STR, due=OPT_STR),
    }
)

OVERVIEW_PROMPT = """Ты — секретарь, который пишет протокол рабочей встречи на русском языке.
Ниже стенограмма: каждая реплика начинается с номера в квадратных скобках и метки спикера.

Верни JSON:
- summary: краткое содержание — {template}. Только факты из стенограммы.
- topics: 3–8 тем в порядке обсуждения, короткие названия (2–4 слова); seg — номер реплики, где тема началась.
- people: имена людей, которые прозвучали в разговоре, в именительном падеже, как их называют («Игорь», «Анна Сергеевна»). Только имена из стенограммы.

Стенограмма:
{transcript}"""

MAP_PROMPT = """Ты — секретарь, который пишет протокол рабочей встречи на русском языке.
Ниже часть {part} из {parts} стенограммы одной встречи; каждая реплика начинается с номера и метки спикера.

Верни JSON только по этой части:
- points: 3–6 главных фактов части по одному предложению — что обсуждали, к чему пришли, цифры, сроки, аргументы сторон.
- topics: темы этой части в порядке обсуждения, короткие названия (2–4 слова); seg — номер реплики, где тема началась.
- people: имена людей, которые прозвучали в этой части, в именительном падеже.

Только факты из стенограммы.

Часть стенограммы:
{transcript}"""

REDUCE_PROMPT = """Ты — секретарь, который пишет протокол рабочей встречи на русском языке.
Ниже заметки по частям одной встречи в хронологическом порядке и темы-кандидаты с номерами реплик.

Верни JSON по всей встрече:
- summary: {template}. Объедини повторы, сохрани цифры и сроки, ничего не добавляй от себя.
- topics: {topics} в порядке обсуждения; похожие темы объедини; seg бери из кандидатов — номер реплики, где тема началась.

Заметки:
{notes}

Темы-кандидаты:
{candidates}"""

ITEMS_PROMPT = """Дата встречи: {date} ({weekday}). Ниже фрагмент стенограммы рабочей встречи, каждая реплика с номером.
Пройди по фрагменту реплика за репликой и выпиши КАЖДОЕ место, где:
- принято решение (с чем-то согласились: «принимаем», «утверждаем», «переносим», «договорились», «хорошо, тогда…») — kind=decision;
- кому-то поручили задачу или кто-то пообещал сделать сам («подготовь», «сделай», «я запрошу», «скину», «запустим») — kind=task.
Если в одной реплике и решение, и поручение — выпиши оба пунктами отдельно.
Для каждого: seg — номер реплики; quote — короткая цитата из неё; text — суть одним предложением (для задачи — в инфинитиве: «Подготовить прогноз…»); owner — имя исполнителя, если прозвучало, иначе метка спикера, который взял задачу на себя, иначе null; due — срок в формате ГГГГ-ММ-ДД, вычисленный от даты встречи («до пятницы», «к пятому», «сегодня», «в понедельник»), иначе null.
Ничего не выдумывай. Если таких мест нет — пустой список.

Фрагмент:
{transcript}"""


def _ctx(text: str) -> int:
    """Context just big enough for the prompt (~3 chars per token in Russian) plus the answer:
    a smaller KV cache leaves memory for Whisper on 16 GB machines."""
    need = len(text) // 3 + 3000
    return min(NUM_CTX, -(-need // 4096) * 4096)


def unload_all() -> None:
    """Free the GPU memory Ollama holds before Whisper runs."""
    try:
        for m in httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5).json().get("models", []):
            httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass  # Ollama not running: nothing to free


def _post(model: str, prompt: str, schema: dict) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "options": {"num_ctx": _ctx(prompt), "temperature": 0.1},
    }
    if model.startswith("qwen3"):
        body["think"] = False  # thinking doubled the time without better extraction on our bench
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=1800)
    except httpx.ConnectError as e:
        raise RuntimeError("Ollama не запущена: откройте приложение Ollama и нажмите «Повторить» — обработка продолжится с анализа.") from e
    if r.status_code == 404:
        raise RuntimeError(f"Модель {model} не скачана: выполните `ollama pull {model}` и нажмите «Повторить».")
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def pack(lines: list[str], limit: int) -> list[list[str]]:
    """Consecutive lines grouped into chunks of at most `limit` characters (a longer line stays alone)."""
    chunks: list[list[str]] = []
    size = 0
    for line in lines:
        if not chunks or size + len(line) + 1 > limit:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line) + 1
    return chunks


def _dedupe(names: list[str]) -> list[str]:
    """Names in first-seen order, case-insensitive; speaker labels («Спикер 1») are not names."""
    seen, out = set(), []
    for n in names:
        key = n.strip().lower()
        if key and key not in seen and not key.startswith("спикер"):
            seen.add(key)
            out.append(n.strip())
    return out


def _notes(parts: list[dict]) -> tuple[str, str]:
    notes = "\n\n".join(f"{p['label']}:\n" + "\n".join(f"- {x}" for x in p["points"]) for p in parts)
    candidates = "\n".join(f"[{t['seg']}] {t['title']}" for p in parts for t in p["topics"])
    return notes, candidates


def _reduce(model: str, template: str, parts: list[dict], depth: int = 0) -> dict:
    """Fold notes of consecutive parts into the final summary. When the notes themselves are too
    long (many hours of talk), neighbouring parts are first folded into intermediate notes, at least
    two at a time, so every level halves the count; three levels cover any realistic meeting."""
    notes, candidates = _notes(parts)
    if len(notes) + len(candidates) > MAP_CHARS and len(parts) > 1 and depth < 3:
        groups, size = [[]], 0
        for p in parts:
            n = sum(len(x) + 3 for x in p["points"]) + sum(len(t["title"]) + 8 for t in p["topics"])
            if groups[-1] and size + n > MAP_CHARS:
                groups.append([])
                size = 0
            groups[-1].append(p)
            size += n
        if len(groups) == len(parts):  # every part is too big to share a prompt: pair them anyway
            groups = [parts[i:i + 2] for i in range(0, len(parts), 2)]
        folded = []
        for g in groups:
            if len(g) == 1:
                folded.append(g[0])
                continue
            n, c = _notes(g)
            r = _post(model, REDUCE_PROMPT.format(template="6–10 пунктов: главное из этих частей, с цифрами и сроками", topics="темы этих частей",
                                                  notes=n, candidates=c), REDUCE_SCHEMA)
            label = f"{g[0]['label'].split(' (')[0]} — {g[-1]['label'].split(' (')[0]}"
            folded.append({"label": label, "points": r["summary"], "topics": r["topics"]})
        return _reduce(model, template, folded, depth + 1)
    return _post(model, REDUCE_PROMPT.format(template=template, topics="3–8 тем", notes=notes, candidates=candidates), REDUCE_SCHEMA)


def _tidy_topics(topics: list[dict], limit: int = 8) -> list[dict]:
    """In meeting order, one topic per starting turn, at most `limit` spread evenly over the meeting:
    models asked for 3–8 topics still return a dozen after merging long meetings."""
    by_seg: dict[int, dict] = {}
    for t in sorted(topics, key=lambda t: t["seg"]):
        by_seg.setdefault(t["seg"], t)
    out = list(by_seg.values())
    if len(out) > limit:
        out = [out[round(i * (len(out) - 1) / (limit - 1))] for i in range(limit)]
    return out


def overview(model: str, template: str, lines: list[str], progress=lambda f: None) -> dict:
    """Summary, topics and names. Short meetings: one call. Long ones: map-reduce — notes per part
    (the model stays attentive on a short text and the prompt fits the context), then one fold."""
    tpl = TEMPLATES.get(template, TEMPLATES["short"])
    chunks = pack(lines, MAP_CHARS)
    if len(chunks) == 1:
        r = _post(model, OVERVIEW_PROMPT.format(template=tpl, transcript="\n".join(lines)), OVERVIEW_SCHEMA)
        return {"summary": r["summary"], "topics": _tidy_topics(r["topics"]), "people": _dedupe(r["people"])}
    parts, people = [], []
    for i, chunk in enumerate(chunks, 1):
        r = _post(model, MAP_PROMPT.format(part=i, parts=len(chunks), transcript="\n".join(chunk)), MAP_SCHEMA)
        first, last = chunk[0].split("]")[0].lstrip("["), chunk[-1].split("]")[0].lstrip("[")
        parts.append({"label": f"Часть {i} (реплики {first}–{last})", "points": r["points"], "topics": r["topics"]})
        people += r["people"]
        progress(0.9 * i / len(chunks))
    final = _reduce(model, tpl, parts)
    return {"summary": final["summary"], "topics": _tidy_topics(final["topics"]), "people": _dedupe(people)}


def analyze(model: str, template: str, day: date, lines: list[str], progress=lambda f: None) -> dict:
    """Two focused passes work better for 7–8B models than one big prompt:
    an overview of the whole meeting, then decisions and tasks window by window."""
    over = overview(model, template, lines, lambda f: progress(0.4 * f))
    progress(0.4)
    items, seen = [], set()
    starts = list(range(0, max(len(lines) - OVERLAP, 1), WINDOW - OVERLAP))
    for n, i in enumerate(starts):
        chunk = lines[i:i + WINDOW]
        found = _post(model, ITEMS_PROMPT.format(date=day.isoformat(), weekday=WEEKDAYS[day.weekday()], transcript="\n".join(chunk)), ITEMS_SCHEMA)
        for it in found["items"]:
            key = (it["kind"], it["seg"], it["text"].lower()[:40])
            if key not in seen:
                seen.add(key)
                items.append(it)
        progress(0.4 + 0.6 * (n + 1) / len(starts))
    return {
        **over,
        "decisions": [{"text": it["text"], "seg": it["seg"]} for it in items if it["kind"] == "decision"],
        "tasks": [{"text": it["text"], "owner": it["owner"], "due": it["due"], "seg": it["seg"]} for it in items if it["kind"] == "task"],
    }


ANSWER_SYSTEM = """Ты отвечаешь на вопросы по архиву рабочих встреч. Используй ТОЛЬКО приведённые фрагменты.
В каждом фрагменте найденная реплика отмечена стрелкой →, рядом — соседние реплики для контекста.
После каждого утверждения ставь ссылку на фрагмент в квадратных скобках, например [2] или [1][3].
Если во фрагментах нет ответа, так и скажи одной фразой: «В записанных встречах нет информации по этому вопросу.» — и ничего не придумывай.
Отвечай по-русски, кратко: 1–4 предложения."""


def answer(model: str, question: str, fragments: list[str]) -> Iterator[str]:
    context = "\n\n".join(f"[{i}] {f}" for i, f in enumerate(fragments, 1))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Фрагменты:\n{context}\n\nВопрос: {question}"},
        ],
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.1},
    }
    if model.startswith("qwen3"):
        body["think"] = False
    with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=body, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if text := chunk.get("message", {}).get("content"):
                    yield text
