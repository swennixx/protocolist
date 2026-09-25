"""Map-reduce overview logic with the model replaced by a fake: no Ollama needed."""

from app import llm


def fake_llm(calls):
    def post(model, prompt, schema):
        calls.append((schema, prompt))
        if schema is llm.MAP_SCHEMA:
            first = int(prompt.split("Часть стенограммы:\n[")[1].split("]")[0])
            return {"points": [f"факт с реплики {first}"], "topics": [{"title": f"тема {first}", "seg": first}], "people": ["Анна", "анна ", f"Гость{first}"]}
        if schema is llm.REDUCE_SCHEMA:
            points = [line[2:] for line in prompt.split("Заметки:\n")[1].split("\n\nТемы-кандидаты")[0].splitlines() if line.startswith("- ")]
            return {"summary": points, "topics": [{"title": "итог", "seg": 0}]}
        if schema is llm.OVERVIEW_SCHEMA:
            return {"summary": ["коротко"], "topics": [], "people": []}
        return {"items": []}
    return post


def lines(n, width=90):
    return [f"[{i}] Спикер {i % 3 + 1}: " + "слово " * (width // 6) for i in range(n)]


def test_pack_keeps_order_and_limit():
    ls = lines(50)
    chunks = llm.pack(ls, 1000)
    assert [x for c in chunks for x in c] == ls
    assert all(sum(len(x) + 1 for x in c) <= 1000 for c in chunks)


def test_short_meeting_is_one_call(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_llm(calls))
    r = llm.overview("m", "short", lines(10))
    assert r["summary"] == ["коротко"] and [c[0] for c in calls] == [llm.OVERVIEW_SCHEMA]


def test_long_meeting_maps_every_line_once_then_reduces(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_llm(calls))
    monkeypatch.setattr(llm, "MAP_CHARS", 2000)
    ls = lines(60)
    r = llm.overview("m", "short", ls)
    maps = [p for s, p in calls if s is llm.MAP_SCHEMA]
    assert len(maps) == len(llm.pack(ls, 2000)) > 1
    mapped = [x for p in maps for x in p.split("Часть стенограммы:\n")[1].splitlines()]
    assert mapped == ls  # every line in exactly one part, in order
    assert calls[-1][0] is llm.REDUCE_SCHEMA
    assert r["summary"] == [f"факт с реплики {p.split('Часть стенограммы:')[1].split('[')[1].split(']')[0]}" for p in maps]
    firsts = [p.split("Часть стенограммы:\n[")[1].split("]")[0] for p in maps]
    assert r["people"] == ["Анна"] + [f"Гость{f}" for f in firsts]  # union, case-insensitive dedupe, order kept


def test_very_long_meeting_reduces_in_levels(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_llm(calls))
    monkeypatch.setattr(llm, "MAP_CHARS", 300)  # notes of all parts overflow one reduce prompt
    ls = lines(40, width=60)
    r = llm.overview("m", "short", ls)
    reduces = [p for s, p in calls if s is llm.REDUCE_SCHEMA]
    assert len(reduces) > 1
    assert all(len(p.split("Заметки:\n")[1]) < 2000 for p in reduces)
    n_maps = sum(1 for s, _ in calls if s is llm.MAP_SCHEMA)
    assert len(r["summary"]) == n_maps  # nothing lost on the way up


def test_reduce_terminates_when_every_part_exceeds_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_llm(calls))
    monkeypatch.setattr(llm, "MAP_CHARS", 10)  # nothing fits anywhere: pairing and the depth cap must stop it
    r = llm.overview("m", "short", lines(30, width=30))
    n_maps = sum(1 for s, _ in calls if s is llm.MAP_SCHEMA)
    n_reduces = sum(1 for s, _ in calls if s is llm.REDUCE_SCHEMA)
    assert n_reduces < n_maps  # each level at least halves the parts
    assert len(r["summary"]) == n_maps


def test_topics_are_ordered_unique_and_capped():
    topics = [{"title": f"t{i}", "seg": i} for i in (9, 1, 1, 5, 3, 12, 7, 20, 15, 30, 25, 0, 2)]
    out = llm._tidy_topics(topics)
    segs = [t["seg"] for t in out]
    assert segs == sorted(set(segs)) and len(out) == 8
    assert segs[0] == 0 and segs[-1] == 30  # the start and the end of the meeting stay covered
    assert llm._dedupe(["Спикер 1", "Игорь", "игорь", " Анна "]) == ["Игорь", "Анна"]
