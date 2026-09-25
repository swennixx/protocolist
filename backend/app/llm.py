"""Local LLM through Ollama: meeting analysis and archive Q&A."""

import json
import os
from datetime import date
from typing import Iterator

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = 32768
WINDOW, OVERLAP = 40, 5  # transcript lines per extraction call

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


def analyze(model: str, template: str, day: date, lines: list[str], progress=lambda f: None) -> dict:
    """Two focused passes work better for 7–8B models than one big prompt:
    an overview of the whole meeting, then decisions and tasks window by window."""
    # ponytail: overview sees the whole transcript at once; ~3 h meetings overflow NUM_CTX and need map-reduce.
    overview = _post(model, OVERVIEW_PROMPT.format(template=TEMPLATES.get(template, TEMPLATES["short"]), transcript="\n".join(lines)), OVERVIEW_SCHEMA)
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
        **overview,
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
